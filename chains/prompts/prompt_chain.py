from dataclasses import dataclass, field, replace, asdict, is_dataclass
from typing import List, Dict, Union, Any, Optional, Tuple, Type, Callable

from functools import wraps
import inspect
import os
import sys
import json

from chains.msg_chain import MessageChain
from pydantic import BaseModel
from jinja2 import Environment


def chain_method(func):
    """Decorator to convert a function into a chainable method."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        return func(self, *args, **kwargs)

    return wrapper


_jinja_env = Environment()
_jinja_env.globals['str'] = str
_jinja_env.globals['len'] = len
_jinja_env.globals['int'] = int
_jinja_env.globals['float'] = float
_jinja_env.globals['list'] = list
_jinja_env.globals['dict'] = dict
_jinja_env.globals['enumerate'] = enumerate

def replace_strs(s: str, kwargs: Dict[str, Any]) -> str:
    template = _jinja_env.from_string(s)
    return template.render(**kwargs)


@dataclass(frozen=True)
class Prompt:
    template: str
    response_format: Optional[Type[BaseModel]] = None
    pre_tuple: Tuple[Callable] = field(default_factory=tuple)
    post_tuple: Tuple[Callable] = field(default_factory=tuple)
    rendered: Optional[str] = None  # The prompt after replace_strs is applied


@dataclass(frozen=True)
class PromptChain:
    curr_prompt: Optional[Prompt] = None
    prev_prompts: Tuple[Prompt] = field(default_factory=tuple)
    response_list: Tuple[Any] = field(default_factory=tuple)
    prev_fields: Dict[str, Any] = field(default_factory=dict)
    msg_chain_func: Optional[Callable] = None

    def __getattr__(self, name):
        """Allow attribute access to prev_fields for convenience"""
        if name in ('curr_prompt', 'prev_prompts', 'response_list', 'prev_fields', 'msg_chain_func'):
            # These are actual dataclass fields, use default behavior
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        if name in self.prev_fields:
            return self.prev_fields[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}' in prev_fields")

    @chain_method
    def prompt(self, template: str):
        # msg_chain_func = None
        response_format = None
        if self.curr_prompt is not None:
            if self.curr_prompt.response_format is not None:
                response_format = self.curr_prompt.response_format
            # if self.curr_prompt.msg_chain_func is not None:
            # msg_chain_func = self.curr_prompt.msg_chain_func
        prompt = Prompt(
            template=template,
            response_format=response_format,
            # msg_chain_func=msg_chain_func,
        )
        return replace(self, curr_prompt=prompt)

    def set_prev_fields(self, prev_fields: Dict[str, Any]):
        return replace(self, prev_fields=prev_fields)

    @chain_method
    def with_structure(self, response_format: Type[BaseModel]):
        """Set a Pydantic model as the expected response format."""
        curr_prompt = replace(self.curr_prompt, response_format=response_format)
        return replace(self, curr_prompt=curr_prompt)

    @chain_method
    def set_model(self, func: Callable):
        # Store the function that creates the message chain
        return replace(self, msg_chain_func=func)

    @chain_method
    def pipe(self, func: Callable):
        """Apply a function to this chain and return the result.

        The function should take a PromptChain as input and return a PromptChain.
        This enables functional composition of chain operations.

        Example:
            chain.pipe(generate_attributes).pipe(create_stages)
        """
        return func(self)

    @chain_method
    def when(self, condition, true_func: Optional[Callable]=None, false_func: Optional[Callable]=None):
        """Conditionally execute true_func or false_func based on condition
        
        Args:
            condition: Either a callable that takes the chain and returns bool,
                      or a string key to look up in prev_fields (must be bool)
            true_func: Function to call if condition is True
            false_func: Function to call if condition is False (optional, defaults to identity)
        """
        if isinstance(condition, str):
            # Look up the condition in prev_fields
            condition_value = self.prev_fields.get(condition)
            assert isinstance(condition_value, bool), f"prev_fields['{condition}'] must be a bool, got {type(condition_value)}"
            should_execute_true = condition_value
        else:
            # Assume it's a callable
            should_execute_true = condition(self)
        assert true_func is not None or false_func is not None, "At least one of true_func or false_func must be provided"
        
        if false_func is None:
            false_func = lambda c: c
        if true_func is None:
            true_func = lambda c: c
            
        
        return true_func(self) if should_execute_true else false_func(self)

    @chain_method
    def post_last(self, **named_transformations):
        """Apply transformations to the last response and add the results to prev_fields

        Transform functions can accept either:
        - One argument (last_response) - legacy behavior
        - Two arguments (last_response, chain) - new behavior for chain context access
        """
        if not self.response_list:
            return self

        last_response = self.response_list[-1]
        new_fields = {}

        for field_name, transform_func in named_transformations.items():
            call_args = None
            call_kwargs = None

            try:
                sig = inspect.signature(transform_func)
            except (TypeError, ValueError):
                sig = None

            if sig is not None:
                candidate_calls = [
                    ((last_response, self), {}),
                    ((last_response,), {'chain': self}),
                    ((), {'last_response': last_response, 'chain': self}),
                    ((last_response,), {}),
                    ((), {'last_response': last_response}),
                    ((), {'chain': self}),
                    ((), {}),
                ]

                for args, kwargs in candidate_calls:
                    try:
                        sig.bind(*args, **kwargs)
                    except TypeError:
                        continue
                    else:
                        call_args, call_kwargs = args, kwargs
                        break

                if call_args is None and call_kwargs is None:
                    raise TypeError(
                        f"Transform '{field_name}' has an unsupported signature: {sig}"
                    )
            else:
                call_args = (last_response,)
                call_kwargs = {}

            new_fields[field_name] = transform_func(*call_args, **call_kwargs)

        return replace(self, prev_fields={**self.prev_fields, **new_fields})

    @chain_method
    def post_chain(self, transform_func):
        """Apply a transformation to the entire chain and add results to prev_fields"""
        new_fields = transform_func(self)
        return replace(self, prev_fields={**self.prev_fields, **new_fields})
    
    @chain_method
    def post_chain_class(self, class_obj, name):
        """Use  a transformation to the entire chain and add results to prev_fields"""
        obj = class_obj(**self.prev_fields)
        new_fields = {name: obj}
        return replace(self, prev_fields={**self.prev_fields, **new_fields})

    @chain_method
    def print_last(self):
        """Print the last response for debugging purposes"""
        if self.response_list:
            print(f"Last response: {self.response_list[-1]}")
        return self

    @chain_method
    def generate(self):
        # Use the message chain function if provided, otherwise create a default one
        if self.msg_chain_func is not None:
            chain = self.msg_chain_func()
        else:
            # Use MessageChain from the chains library with default model
            chain = MessageChain.get_chain(model="gpt-4o")
            chain = chain.system("You are a helpful assistant.")

        prompt = replace_strs(self.curr_prompt.template, self.prev_fields)

        # Add user message and set structure if needed
        chain = chain.user(prompt)

        if self.curr_prompt.response_format is not None:
            chain = chain.with_structure(self.curr_prompt.response_format)

        # Generate response
        chain = chain.generate()
        print("Step")
        # Get the output from the chain
        output = chain.last_response
        # Add the response to the response_list
        new_response_list = self.response_list + (output,)
        # Store the rendered prompt in the Prompt object
        curr_prompt_with_rendered = replace(self.curr_prompt, rendered=prompt)
        # Add the current prompt (with rendered text) to prev_prompts
        new_prev_prompts = self.prev_prompts + (curr_prompt_with_rendered,)

        return replace(
            self, response_list=new_response_list, prev_prompts=new_prev_prompts
        )

    @chain_method
    def gen_prompt(self, template: str):
        """Generate a response and then use it to create a new prompt in the chain."""
        # First generate the response from the current prompt
        updated_chain = self.generate()

        # Then create a new prompt with the given template on the updated chain
        return updated_chain.prompt(template)

    @chain_method
    def save(self, file_path: str, mode: str = 'both', indent: int = 2):
        """Save chain data to a JSON file.

        Args:
            file_path: Path to save the file
            mode: What to save - 'responses', 'fields', or 'both' (default)
            indent: JSON indentation level (default: 2)
        """
        data = {}



        
        
        def _serialize(value):
            if isinstance(value, BaseModel):
                return value.model_dump()
            if isinstance(value, dict):
                return {k: _serialize(v) for k, v in value.items()}
            if isinstance(value, (list, tuple, set)):
                serialized = [_serialize(v) for v in value]
                return serialized if isinstance(value, list) else list(serialized)
            if is_dataclass(value):
                return asdict(value)
            return value

        # Convert prev_fields to serializable format
        fields = {key: _serialize(value) for key, value in self.prev_fields.items()}
        data['fields'] = fields

        # Convert responses to serializable format
        responses = [_serialize(resp) for resp in self.response_list]
        # Always include prompts for context
        prompts = []
        for prompt,resp in zip(self.prev_prompts, responses):
            prompts.append({
                'template': prompt.template,
                'rendered': prompt.rendered,
                'response_format': prompt.response_format.__name__ if prompt.response_format else None,
                "response": resp
            })
        data['prompts'] = prompts



        with open(file_path, 'w') as f:
            json.dump(data, f, indent=indent)

        return self

    def __rshift__(self, other):
        """Support operator-based chaining with >> operator

        Usage:
            chain >> pipeline  # Returns PipelineExecutor
            chain >> callable  # Calls callable with chain
        """
        # Import here to avoid circular dependency
        from chains.prompts.prompt_module import Pipeline

        if isinstance(other, Pipeline):
            # Pipeline case: call the pipeline with this chain
            return other(self)
        elif callable(other):
            # Callable case: apply the callable to this chain
            return other(self)
        else:
            raise TypeError(f"unsupported operand type(s) for >>: 'PromptChain' and '{type(other).__name__}'")
