from typing import Type, Dict, Callable, Optional, Any, Union
from pydantic import BaseModel


# ============================================================================
# Shared execution logic
# ============================================================================

def execute_stage_with_chain(
    chain,
    template: str,
    output_class: Optional[Type[BaseModel]] = None,
    output_name: Optional[str] = None,
    post_transforms: Optional[Dict[str, Callable]] = None,
    preproc_func: Optional[Callable] = None,
    prev_fields: Optional[Dict[str, Any]] = None
):
    """Execute a single stage with a chain.

    This is shared logic used by both normal and compiled execution modes.

    Args:
        chain: The PromptChain to execute with
        template: The prompt template (can be pre-rendered or contain {{variables}})
        output_class: Optional Pydantic model for structured output
        output_name: Optional name for storing the output in prev_fields
        post_transforms: Optional dict of post-processing transforms
        preproc_func: Optional preprocessing function
        prev_fields: Optional dict to update chain.prev_fields before execution

    Returns:
        Updated PromptChain after executing this stage
    """
    # Update prev_fields if provided
    if prev_fields:
        chain = chain.set_prev_fields({**chain.prev_fields, **prev_fields})

    # Run preprocessing if defined
    if preproc_func:
        preproc_result = preproc_func(chain)

        # If preproc returns a chain, short-circuit (skip prompt execution)
        if hasattr(preproc_result, 'prev_fields'):  # It's a PromptChain
            return preproc_result

        # If preproc returns a dict, add to prev_fields for template rendering
        if isinstance(preproc_result, dict):
            chain = chain.set_prev_fields({**chain.prev_fields, **preproc_result})

    # Execute the prompt
    chain = (
        chain
        .prompt(template)
        .with_structure(output_class)
        .generate()
    )

    # Apply post-processing transforms if specified
    if post_transforms:
        chain = chain.post_last(**post_transforms)
    elif output_name:
        # Default: store result under output_name
        chain = chain.post_last(**{output_name: lambda x: x})

    return chain


# ============================================================================
# Decorators
# ============================================================================

def register_output(output_class: Type[BaseModel]):
    """Decorator to register the output class for a PromptModule"""
    def decorator(cls):
        cls._output_class = output_class
        return cls
    return decorator


def register_prompt(
    template: str,
    post_last: Optional[Dict[str, Callable]] = None,
    preproc: Optional[Callable] = None
):
    """Decorator to register a prompt template on a Pydantic model

    Usage:
        @register_prompt("Generate a job for {{sector}}")
        @pipeline.register_stage("job")
        class JobRole(BaseModel):
            title: str

        # With post-processing
        @register_prompt(
            "Decide if we should include sector",
            post_last={"has_sector": lambda x: x.include_sector}
        )
        @pipeline.register_stage("has_sector")
        class SectorDecision(BaseModel):
            include_sector: bool

        # With preprocessing
        @register_prompt(
            "Select sector: {{sectors_list}}",
            preproc=lambda chain: {"sectors_list": "..."},
            post_last={"sector": lambda x: transform(x)}
        )
        @pipeline.register_stage("sector")
        class SectorSelection(BaseModel):
            chosen_sector: str
    """
    def decorator(cls):
        cls._prompt_template = template
        cls._output_class = cls  # Self-referential for consistency
        if post_last is not None:
            cls._post_last = post_last  # Optional post-processing transforms
        if preproc is not None:
            cls._preproc = preproc  # Optional preprocessing function
        return cls
    return decorator


class PromptModule:
    output_name: str = None  # Must be set by subclass or decorator
    _output_class: Type[BaseModel] = None  # Set by @register_output decorator

    def __init__(self, chain):
        self.chain = chain

    def preproc(self) -> Optional[Dict[str, Any]]:
        """
        Optional preprocessing step. Can return:
        - None: continue with normal flow
        - dict: additional fields to add to prev_fields for template rendering
        - PromptChain: early return (bypass forward/generate/post_last)
        """
        return None

    def forward(self, **kwargs) -> str:  # noqa: ARG002
        """
        Returns the prompt template string.
        kwargs contains any fields returned from preproc()
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement forward()")

    def post_last(self) -> Optional[Dict[str, Callable]]:
        """
        Optional post-processing of the generated output.
        Returns a dict mapping field names to lambda functions that transform the output.
        """
        return None


    def execute(self):
        """Execute this module's full pipeline"""
        # Run preprocessing
        preproc_result = self.preproc()

        # If preproc returns a chain, short-circuit
        if hasattr(preproc_result, 'prev_fields'):  # It's a PromptChain
            return preproc_result

        # Build additional fields for template
        template_fields = preproc_result if isinstance(preproc_result, dict) else {}

        # Add template fields temporarily if needed
        if template_fields:
            self.chain = self.chain.set_prev_fields({**self.chain.prev_fields, **template_fields})

        # Get prompt template
        prompt_template = self.forward(**template_fields)

        # Execute chain: prompt -> structure -> generate
        self.chain = (
            self.chain
            .prompt(prompt_template)
            .with_structure(self._output_class)
            .generate()
        )

        # Apply post-processing if defined
        post_transforms = self.post_last()
        if post_transforms:
            self.chain = self.chain.post_last(**post_transforms)
        else:
            # Default: store output under output_name
            if self.output_name:
                self.chain = self.chain.post_last(**{self.output_name: lambda x: x})

        return self.chain


class Pipeline:
    """
    Pipeline that executes a sequence of PromptModules.

    Usage:
        pipeline = Pipeline()

        @pipeline.register_stage("sector")
        @register_output(SectorSelection)
        class SectorModule(PromptModule):
            def forward(self, **kwargs):
                return "Select a sector..."

        result = pipeline(chain).init(seed_paragraph="...").set_output(UserProfile).execute()
    """

    def __init__(self):
        self.stages = []  # List of stage specifications (module_class or dict with conditional info)

    def register_stage(
        self,
        output_name: Optional[str] = None,
        when: Optional[str] = None,
        post_last: Optional[Dict[str, Callable]] = None,
    ):
        """Decorator to register a stage in the pipeline

        Args:
            output_name: Optional name for the stage output
            when: Optional condition field name - stage only executes if this prev_fields value is truthy
                  (Deprecated: use register_when() decorator instead for better composability)
        """
        def decorator(module_class):
            # Set output_name on the class if provided
            if output_name:
                module_class.output_name = output_name

            # Ensure output_name is set
            if not hasattr(module_class, 'output_name') or module_class.output_name is None:
                raise AttributeError(
                    f"{module_class.__name__} must have 'output_name' attribute. "
                    f"Set it in the class or pass it to @register_stage()"
                )

            # Attach post_last transforms if provided
            if post_last is not None:
                module_class._post_last = post_last

            # Store either the module class directly or wrapped with condition
            if when:
                self.stages.append({
                    'module_class': module_class,
                    'condition': when
                })
            else:
                self.stages.append(module_class)
            return module_class
        return decorator

    def register_when(self, condition: str):
        """Decorator to mark a stage as conditional

        Must be stacked on top of register_stage():
            @pipeline.register_when("should_execute")
            @pipeline.register_stage("my_stage")
            class MyStage(BaseModel): ...

        Args:
            condition: Field name in prev_fields to check (must be bool)
        """
        def decorator(module_class):
            # Wrap the class in a dict with condition info
            # This will be stored in stages when register_stage() is called
            # We need to modify how the stage is stored in the stages list

            # Find this class in stages and wrap it
            for i, stage_spec in enumerate(self.stages):
                if stage_spec == module_class or (isinstance(stage_spec, dict) and stage_spec.get('module_class') == module_class):
                    # Wrap it
                    self.stages[i] = {
                        'module_class': module_class,
                        'condition': condition
                    }
                    break

            return module_class
        return decorator

    def register_loop(self, output_name: str, length: Union[int, str], per_iter: Optional[Callable] = None):
        """Decorator to mark a stage as looped N times

        Must be stacked on top of register_stage():
            @pipeline.register_loop("questions", length=3)
            @pipeline.register_stage("question")
            class MyStage(BaseModel): ...

        Args:
            output_name: Where to store the list of results (e.g., "questions")
            length: Fixed int or field reference (e.g., "num_questions")
            per_iter: Optional callable (int, PromptChain) -> Dict for per-iteration fields
        """
        def decorator(module_class):
            # Validate: class must already have register_stage() applied
            if not hasattr(module_class, 'output_name'):
                raise AttributeError(
                    f"{module_class.__name__} must have register_stage() applied before register_loop()"
                )

            # Validate per_iter signature if provided
            if per_iter is not None and not callable(per_iter):
                raise TypeError("per_iter must be a callable")

            # Store loop config on the class
            module_class._loop_config = {
                'output_name': output_name,
                'length': length,
                'per_iter': per_iter
            }

            return module_class
        return decorator

    def __call__(self, chain):
        """Initialize pipeline with a PromptChain"""
        return PipelineExecutor(self.stages, chain)
    
    def register_prompt(self, template: str,
            post_last: Optional[Dict[str, Callable]] = None,
            preproc: Optional[Callable] = None):
        return register_prompt(template, post_last, preproc)
    


class PipelineExecutor:
    """Executes a pipeline of modules on a chain"""

    def __init__(self, stages, chain):
        self.stages = stages
        self.chain = chain
        self._output_class = None
        self._output_name = None

    def init(self, **fields):
        """Initialize prev_fields with given values"""
        self.chain = self.chain.set_prev_fields({**self.chain.prev_fields, **fields})
        return self

    def pset(self, **fields):
        """Set additional fields in prev_fields"""
        self.chain = self.chain.set_prev_fields({**self.chain.prev_fields, **fields})
        return self

    def set_output(self, output_class=None, name=None, **kwargs):
        """Set the final output class to construct from prev_fields

        Can be used in two ways:
        1. set_output(UserProfile, "profile") - positional args
        2. set_output(profile=UserProfile) - keyword arg with name=class

        Args:
            output_class: The Pydantic model to construct (positional style)
            name: Optional name for the output field (positional style)
            **kwargs: Keyword argument where key=name, value=class (keyword style)
        """
        # Handle keyword argument style: set_output(profile=UserProfile)
        if kwargs:
            if len(kwargs) > 1:
                raise ValueError("set_output() accepts only one keyword argument")
            if output_class is not None:
                raise ValueError("Cannot use both positional and keyword arguments")
            name, output_class = next(iter(kwargs.items()))

        if output_class is None:
            raise ValueError("Must provide output_class either positionally or as keyword argument")

        self._output_class = output_class
        self._output_name = name
        return self

    def execute(self):
        """Execute all stages in sequence"""
        # Execute each module
        for stage_spec in self.stages:
            # Check if this is a conditional stage
            if isinstance(stage_spec, dict):
                module_class = stage_spec['module_class']
                condition = stage_spec['condition']

                # Define execution function for the stage
                def execute_stage(chain):
                    return self._execute_module(module_class, chain)

                # Use chain.when() to conditionally execute
                self.chain = self.chain.when(condition, true_func=execute_stage)
            else:
                # Regular stage - execute unconditionally
                module_class = stage_spec
                self.chain = self._execute_module(module_class, self.chain)

            self.chain.print_last()

        # If output class is specified, construct it from prev_fields
        if self._output_class:
            output_name = self._output_name or self._output_class.__name__.lower()
            self.chain = self.chain.post_chain_class(self._output_class, output_name)

        # Return the final chain with the output accessible via attribute
        return self.chain

    def _execute_module(self, module_class, chain):
        """Execute a single module and return updated chain"""
        # Check if this is a decorated Pydantic model (new style) or PromptModule (old style)
        if hasattr(module_class, '_prompt_template'):
            # New style: Pydantic model with @register_prompt decorator
            template = module_class._prompt_template
            output_class = module_class._output_class
            output_name = module_class.output_name
            post_transforms = getattr(module_class, '_post_last', None)
            preproc_func = getattr(module_class, '_preproc', None)

            # Use shared execution logic
            return execute_stage_with_chain(
                chain=chain,
                template=template,
                output_class=output_class,
                output_name=output_name,
                post_transforms=post_transforms,
                preproc_func=preproc_func
            )
        else:
            # Old style: PromptModule subclass
            module = module_class(chain)
            return module.execute()

    def __rshift__(self, other):
        """Support operator-based chaining with >> operator

        Usage:
            executor >> init(data=value)  # Returns PipelineExecutor
            executor >> set_output(result=Model)  # Returns PipelineExecutor
            executor >> execute  # Executes and returns PromptChain
            executor >> next_pipeline  # Chains to next pipeline
        """
        if isinstance(other, Pipeline):
            # Pipeline case: call the pipeline with the current chain
            # If output_class is set, execute and materialize it before continuing
            if self._output_class:
                # Execute current stages
                for stage_spec in self.stages:
                    # Check if this is a conditional stage
                    if isinstance(stage_spec, dict):
                        module_class = stage_spec['module_class']
                        condition = stage_spec['condition']

                        # Define execution function for the stage
                        def execute_stage(chain):
                            return self._execute_module(module_class, chain)

                        # Use chain.when() to conditionally execute
                        self.chain = self.chain.when(condition, true_func=execute_stage)
                    else:
                        # Regular stage - execute unconditionally
                        module_class = stage_spec
                        self.chain = self._execute_module(module_class, self.chain)

                    self.chain.print_last()

                # Materialize the output object and add to prev_fields
                output_name = self._output_name or self._output_class.__name__.lower()
                self.chain = self.chain.post_chain_class(self._output_class, output_name)

            return other(self.chain)
        elif callable(other):
            # Callable case (wrappers, functions, etc.): apply to this executor
            return other(self)
        else:
            raise TypeError(f"unsupported operand type(s) for >>: 'PipelineExecutor' and '{type(other).__name__}'")


# ============================================================================
# Callable Wrappers for Operator-Based Chaining
# ============================================================================

class InitWrapper:
    """Wrapper for init() to support operator chaining"""
    def __init__(self, **fields):
        self.fields = fields

    def __call__(self, executor):
        """Apply init() to a PipelineExecutor"""
        return executor.init(**self.fields)


class PsetWrapper:
    """Wrapper for pset() to support operator chaining"""
    def __init__(self, **fields):
        self.fields = fields

    def __call__(self, executor):
        """Apply pset() to a PipelineExecutor"""
        return executor.pset(**self.fields)


class SetOutputWrapper:
    """Wrapper for set_output() to support operator chaining"""
    def __init__(self, output_class=None, name=None, **kwargs):
        self.output_class = output_class
        self.name = name
        self.kwargs = kwargs

    def __call__(self, executor):
        """Apply set_output() to a PipelineExecutor"""
        if self.kwargs:
            return executor.set_output(**self.kwargs)
        return executor.set_output(self.output_class, self.name)

class ApplyWrapper:
    def __init__(self, func):
        self.func = func
    def __call__(self, executor):
        """Apply execute() to a PipelineExecutor"""
        return self.func(executor)

def apply(func):
    return ApplyWrapper(func)


class ExecuteWrapper:
    """Wrapper for execute to support operator chaining"""
    def __call__(self, executor):
        """Apply execute() to a PipelineExecutor"""
        return executor.execute()



# Create singleton callable instances
def init(**fields):
    """Create an InitWrapper for operator chaining

    Usage:
        pipe = chain >> pipeline >> init(data=value)
    """
    return InitWrapper(**fields)


def pset(**fields):
    """Create a PsetWrapper for operator chaining

    Usage:
        pipe = executor >> pset(field=value)
    """
    return PsetWrapper(**fields)


def set_output(output_class=None, name=None, **kwargs):
    """Create a SetOutputWrapper for operator chaining

    Usage:
        pipe = executor >> set_output(result=Model)
        pipe = executor >> set_output(Model, "result")
    """
    return SetOutputWrapper(output_class, name, **kwargs)


# Singleton execute object
execute = ExecuteWrapper()

