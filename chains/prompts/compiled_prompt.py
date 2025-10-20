from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Union, Callable, Type
import re

from pydantic import BaseModel
from jinja2 import Environment, meta


@dataclass
class ExecutionNode:
    """Base node in the execution graph"""
    id: str
    dependencies: Set[str] = field(default_factory=set)
    outputs: Set[str] = field(default_factory=set)

    def __hash__(self):
        return hash(self.id)


@dataclass
class PromptNode(ExecutionNode):
    """Node representing a prompt generation stage"""
    template: str = ""
    response_format: Optional[Type[BaseModel]] = None
    post_transforms: Optional[Dict[str, Callable]] = None
    preproc_func: Optional[Callable] = None
    output_name: Optional[str] = None
    condition: Optional[str] = None  # Optional condition field name for conditional execution

    def __post_init__(self):
        """Extract dependencies from the template"""
        if self.template:
            self.dependencies.update(extract_template_dependencies(self.template))


@dataclass
class ConditionalNode(ExecutionNode):
    """Node representing conditional execution (when)"""
    condition: Union[str, Callable] = None
    true_branch: 'ExecutionGraph' = None
    false_branch: 'ExecutionGraph' = None

    def __post_init__(self):
        """Mark condition field as dependency if string"""
        if isinstance(self.condition, str):
            self.dependencies.add(self.condition)


@dataclass
class LoopNode(ExecutionNode):
    """Node representing iterative execution"""
    loop_body: 'ExecutionGraph' = None  # Subgraph to execute N times
    num_iterations: Union[int, str] = None  # Fixed number or field reference
    output_name: str = None  # Where to store list of results
    per_iter_fields: Optional[Callable] = None  # Generate fields per iteration

    def __post_init__(self):
        """Add dependencies"""
        if isinstance(self.num_iterations, str):
            self.dependencies.add(self.num_iterations)  # e.g., "num_questions"
        if self.output_name:
            self.outputs.add(self.output_name)  # e.g., "questions"


@dataclass
class ExecutionGraph:
    """DAG of execution nodes with compilation and optimization"""
    nodes: List[ExecutionNode] = field(default_factory=list)
    node_map: Dict[str, ExecutionNode] = field(default_factory=dict)
    execution_order: List[ExecutionNode] = field(default_factory=list)
    initial_fields: Set[str] = field(default_factory=set)  # Fields provided at start

    def add_node(self, node: ExecutionNode):
        """Add a node to the graph"""
        self.nodes.append(node)
        self.node_map[node.id] = node
        
    def compile(self):
        """Compile the graph: determine execution order"""
        # Topological sort for execution order
        self.execution_order = self._topological_sort()

        # Validate graph
        self._validate()
        
    def _topological_sort(self) -> List[ExecutionNode]:
        """Sort nodes in dependency order"""
        visited = set()
        order = []
        
        def visit(node: ExecutionNode):
            if node.id in visited:
                return
            visited.add(node.id)
            
            # Visit dependencies first
            for dep_field in node.dependencies:
                producer = self._find_producer(dep_field)
                if producer and producer != node:
                    visit(producer)
                    
            order.append(node)
        
        for node in self.nodes:
            visit(node)
            
        return order
    
    def _find_producer(self, field: str) -> Optional[ExecutionNode]:
        """Find the node that produces a given field"""
        # Don't create dependencies for initial fields
        if field in self.initial_fields:
            return None
        for node in self.nodes:
            if field in node.outputs:
                return node
        return None

    def _get_all_dependencies(self, node: ExecutionNode) -> Set[str]:
        """Get all transitive dependencies of a node"""
        deps = set()
        to_process = list(node.dependencies)
        
        while to_process:
            dep_field = to_process.pop()
            producer = self._find_producer(dep_field)
            if producer:
                deps.add(producer.id)
                to_process.extend(producer.dependencies)
                
        return deps
    
    def _validate(self):
        """Validate the graph for cycles and missing dependencies"""
        # Check for cycles using DFS
        WHITE, GRAY, BLACK = 0, 1, 2
        colors = {node.id: WHITE for node in self.nodes}
        
        def has_cycle(node_id: str) -> bool:
            colors[node_id] = GRAY
            node = self.node_map[node_id]
            
            for dep_field in node.dependencies:
                producer = self._find_producer(dep_field)
                if producer:
                    if colors[producer.id] == GRAY:
                        return True  # Back edge found
                    if colors[producer.id] == WHITE and has_cycle(producer.id):
                        return True
                        
            colors[node_id] = BLACK
            return False
        
        for node in self.nodes:
            if colors[node.id] == WHITE:
                if has_cycle(node.id):
                    raise ValueError(f"Cycle detected involving node {node.id}")
                    
        # Check for missing dependencies
        available_fields = self.initial_fields.copy()
        for node in self.execution_order:
            missing = node.dependencies - available_fields
            if missing:
                # Some dependencies might be provided at runtime
                pass  # We'll handle this during execution
            available_fields.update(node.outputs)
    


def extract_template_dependencies(template: str) -> Set[str]:
    """Extract variable dependencies from a Jinja2 template"""
    env = Environment()
    ast = env.parse(template)
    undeclared = meta.find_undeclared_variables(ast)
    
    # Also extract from str() calls and other functions
    # This is a simplified version - could be more sophisticated
    pattern = r'{{\s*(?:str\()?(\w+)(?:\.\w+)*(?:\))?\s*}}'
    matches = re.findall(pattern, template)
    
    dependencies = set(undeclared)
    dependencies.update(matches)
    
    # Remove Jinja2 built-ins
    builtins = {'str', 'len', 'int', 'float', 'list', 'dict', 'range', 'enumerate'}
    dependencies -= builtins
    
    return dependencies


class PipelineCompiler:
    """Compiles a Pipeline into an ExecutionGraph"""
    
    def compile(self, pipeline: 'Pipeline', initial_fields: Dict[str, Any] = None) -> ExecutionGraph:
        """Compile a Pipeline into an optimized execution graph"""
        graph = ExecutionGraph()
        available_fields = set(initial_fields.keys()) if initial_fields else set()
        graph.initial_fields = available_fields.copy()

        for idx, stage_spec in enumerate(pipeline.stages):
            node = self._compile_stage(stage_spec, idx)
            graph.add_node(node)
            available_fields.update(node.outputs)

        graph.compile()
        return graph

    def _compile_stage(self, stage_spec, idx: int) -> ExecutionNode:
        """Compile a single stage into a PromptNode, ConditionalNode, or LoopNode"""
        # Extract module_class (could be wrapped in dict for conditional stages)
        if isinstance(stage_spec, dict):
            module_class = stage_spec['module_class']
            condition = stage_spec['condition']
        else:
            module_class = stage_spec
            condition = None

        # Check for control flow decorators
        if hasattr(module_class, '_loop_config'):
            node = self._compile_loop_stage(module_class, idx)
        else:
            node = self._compile_module(module_class, idx)

        # Apply conditional wrapper if present
        if condition:
            node.condition = condition
            node.dependencies.add(condition)

        return node

    def _compile_module(self, stage_class, idx: int) -> PromptNode:
        """Compile a module class into a PromptNode"""
        # Check if it's a decorated Pydantic model
        if hasattr(stage_class, '_prompt_template'):
            template = stage_class._prompt_template
            output_class = stage_class._output_class
            output_name = getattr(stage_class, 'output_name', None)
            post_transforms = getattr(stage_class, '_post_last', None)
            preproc_func = getattr(stage_class, '_preproc', None)

            node = PromptNode(
                id=f"stage_{idx}_{output_name or stage_class.__name__}",
                template=template,
                response_format=output_class,
                post_transforms=post_transforms,
                preproc_func=preproc_func,
                output_name=output_name
            )

            # Set outputs
            if post_transforms:
                node.outputs.update(post_transforms.keys())
            elif output_name:
                node.outputs.add(output_name)

            return node
        else:
            class_name = getattr(stage_class, '__name__', repr(stage_class))
            raise NotImplementedError(
                f"PromptModule compilation not yet implemented for {class_name}"
            )

    def _compile_loop_stage(self, stage_class, idx: int) -> LoopNode:
        """Compile a loop stage into LoopNode with subgraph

        stage_class may have been decorated with:
        - @pipeline.register_loop() which sets _loop_config (required for this path)
        - @pipeline.register_stage() which sets output_name (required)
        - @pipeline.register_when() which sets _when_condition (optional, handled by caller)
        """
        loop_config = stage_class._loop_config

        # Compile the inner stage as a PromptNode
        # This reads stage_class.output_name (e.g., "question")
        prompt_node = self._compile_module(stage_class, idx)

        # Create subgraph containing just this node (for now)
        loop_body = ExecutionGraph()
        loop_body.add_node(prompt_node)
        loop_body.compile()

        # Create LoopNode wrapper
        # Uses loop_config.output_name (e.g., "questions") not stage_class.output_name
        return LoopNode(
            id=f"loop_{idx}_{loop_config['output_name']}",
            loop_body=loop_body,
            num_iterations=loop_config['length'],  # From register_loop()
            output_name=loop_config['output_name'],  # "questions" from register_loop()
            per_iter_fields=loop_config.get('per_iter')
        )


class ChainCompiler:
    """Compiles a PromptChain into an ExecutionGraph"""
    
    def compile_chain(self, chain) -> ExecutionGraph:
        """Compile a PromptChain into an execution graph"""
        graph = ExecutionGraph()
        
        # This would need to trace through the chain operations
        # For now, simplified implementation
        if chain.curr_prompt:
            node = self._compile_prompt(chain.curr_prompt, 0)
            graph.add_node(node)
            
        graph.compile()
        return graph
    
    def _compile_prompt(self, prompt, idx: int) -> PromptNode:
        """Compile a Prompt into a PromptNode"""
        node = PromptNode(
            id=f"prompt_{idx}",
            template=prompt.template,
            response_format=prompt.response_format
        )
        return node
    
    def _compile_when(self, condition: Union[str, Callable], 
                     true_func: Callable, false_func: Callable,
                     chain, idx: int) -> ConditionalNode:
        """Compile a when() conditional into a ConditionalNode"""
        # Compile both branches
        true_chain = true_func(chain)
        false_chain = false_func(chain) if false_func else chain
        
        true_graph = self.compile_chain(true_chain)
        false_graph = self.compile_chain(false_chain)
        
        node = ConditionalNode(
            id=f"conditional_{idx}",
            condition=condition,
            true_branch=true_graph,
            false_branch=false_graph
        )
        
        return node


# ============================================================================
# Optimized Execution
# ============================================================================

class CompiledExecutor:
    """Executes a compiled graph sequentially"""

    def __init__(self, graph: ExecutionGraph):
        self.graph = graph
        
    def execute(self, chain,
                     initial_fields: Dict[str, Any] = None):
        """Execute the compiled graph sequentially"""
        prev_fields = {**chain.prev_fields, **(initial_fields or {})}

        # Execute each node in topological order
        for node in self.graph.execution_order:
            chain = self._execute_node(node, chain, prev_fields)
            prev_fields = chain.prev_fields

        return chain
    
    def _execute_node(self, node: ExecutionNode,
                          chain,
                          prev_fields: Dict[str, Any]):
        """Execute a single node"""
        if isinstance(node, PromptNode):
            return self._execute_prompt_node(node, chain, prev_fields)
        elif isinstance(node, ConditionalNode):
            return self._execute_conditional_node(node, chain, prev_fields)
        elif isinstance(node, LoopNode):
            return self._execute_loop_node(node, chain, prev_fields)
        else:
            raise ValueError(f"Unknown node type: {type(node)}")
    
    def _execute_prompt_node(self, node: PromptNode,
                                  chain,
                                  prev_fields: Dict[str, Any]):
        """Execute a prompt node"""
        # Check if this node has a condition
        if node.condition:
            condition_value = prev_fields.get(node.condition, False)
            if not condition_value:
                # Condition is false, skip execution
                return chain

        # Import the shared execution logic
        from chains.prompts.prompt_module import execute_stage_with_chain

        # Use shared execution logic (rendering happens inside generate())
        return execute_stage_with_chain(
            chain=chain,
            template=node.template,
            output_class=node.response_format,
            output_name=node.output_name,
            post_transforms=node.post_transforms,
            preproc_func=node.preproc_func,
            prev_fields=prev_fields
        )
    
    def _execute_conditional_node(self, node: ConditionalNode,
                                       chain,
                                       prev_fields: Dict[str, Any]):
        """Execute a conditional node"""
        # Evaluate condition
        if isinstance(node.condition, str):
            condition_value = prev_fields.get(node.condition, False)
        else:
            condition_value = node.condition(chain)

        # Execute appropriate branch
        if condition_value:
            executor = CompiledExecutor(node.true_branch)
        else:
            executor = CompiledExecutor(node.false_branch)

        return executor.execute(chain, prev_fields)

    def _execute_loop_node(self, node: LoopNode,
                                chain,
                                prev_fields: Dict[str, Any]):
        """Execute a loop node - runs subgraph N times, collects results"""
        # Resolve num_iterations
        if isinstance(node.num_iterations, str):
            n = prev_fields[node.num_iterations]  # e.g., prev_fields["num_questions"] = 3
        else:
            n = node.num_iterations

        loop_output_names = sorted({
            output
            for body_node in node.loop_body.nodes
            for output in body_node.outputs
        })

        if not loop_output_names:
            raise ValueError(
                f"Loop body for node {node.id} does not declare any outputs"
            )

        results = []

        # Execute loop body N times
        for i in range(n):
            # 1. Generate per-iteration fields
            iter_prev_fields = dict(prev_fields)
            if node.per_iter_fields:
                iter_fields = node.per_iter_fields(i, chain)
                iter_prev_fields.update(iter_fields)

            # 2. Execute the subgraph
            executor = CompiledExecutor(node.loop_body)
            chain = executor.execute(chain, iter_prev_fields)

            # 3. Collect outputs from this iteration
            if len(loop_output_names) == 1:
                result = chain.prev_fields[loop_output_names[0]]
            else:
                result = {
                    name: chain.prev_fields[name]
                    for name in loop_output_names
                }
            results.append(result)

            # 4. Update prev_fields for next iteration
            prev_fields = chain.prev_fields

        # Store the list in prev_fields
        return chain.set_prev_fields({**prev_fields, node.output_name: results})



class ChainedPipeline:
    """Represents multiple pipelines chained with the >> operator"""
    
    def __init__(self, pipelines: List['Pipeline']):
        self.pipelines = pipelines
        self.compiled_graphs = []
        
    def compile(self) -> List[ExecutionGraph]:
        """Compile all pipelines in the chain"""
        self.compiled_graphs = []
        compiler = PipelineCompiler()
        
        for pipeline in self.pipelines:
            graph = compiler.compile(pipeline)
            self.compiled_graphs.append(graph)
            
        return self.compiled_graphs
    
    def execute(self, chain):
        """Execute the chained pipelines"""
        for graph in self.compiled_graphs:
            executor = CompiledExecutor(graph)
            chain = executor.execute(chain)

        return chain

# Module-level cache for compiled graphs
_compiled_graph_cache = {}

def monkeypatch_pipeline(pipeline_class):
    """Monkey-patch the Pipeline class to support compiled mode"""

    original_call = pipeline_class.__call__

    def compiled_call(self, chain, compiled=True):
        """Initialize pipeline with optional compiled mode"""
        if compiled:
            from chains.prompts.prompt_module import PipelineExecutor
            # Create a compiled executor wrapper
            executor = PipelineExecutor(self.stages, chain)
            executor._compiled = True

            # Use pipeline instance id as cache key
            cache_key = id(self)

            # Override execute to use compiled mode
            def compiled_execute():
                # Check cache first, compile if not cached
                if cache_key not in _compiled_graph_cache:
                    compiler = PipelineCompiler()
                    _compiled_graph_cache[cache_key] = compiler.compile(self, executor.chain.prev_fields)

                # Get compiled graph from cache
                graph = _compiled_graph_cache[cache_key]

                # Run synchronous execution
                compiled_executor = CompiledExecutor(graph)
                result = compiled_executor.execute(executor.chain, executor.chain.prev_fields)

                # Handle set_output() - materialize the output class if specified
                if executor._output_class:
                    output_name = executor._output_name or executor._output_class.__name__.lower()
                    result = result.post_chain_class(executor._output_class, output_name)

                return result

            executor.execute = compiled_execute
            return executor
        else:
            return original_call(self, chain)

    pipeline_class.__call__ = compiled_call
    return pipeline_class


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    # Example of how to use compiled mode
    from chains.prompts.prompt_module import Pipeline
    from chains.prompts.prompt_chain import PromptChain
    
    # Enable compiled mode on Pipeline class
    Pipeline = monkeypatch_pipeline(Pipeline)
    
    # Now pipelines will run in compiled mode by default
    pipeline = Pipeline()
    chain = PromptChain()
    
    # This will compile and optimize the pipeline
    result = pipeline(chain, compiled=True).init(data="test").execute()
    
    print("Compiled execution complete!")
