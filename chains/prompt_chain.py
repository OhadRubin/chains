from dataclasses import dataclass, field, replace
from typing import List, Dict, Union, Any, Optional, Tuple, Type, Callable

from functools import wraps
import os
import sys

from chains.msg_chain import MessageChain
from pydantic import BaseModel
import re


def chain_method(func):
    """Decorator to convert a function into a chainable method."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        return func(self, *args, **kwargs)

    return wrapper


def replace_strs(s: str, kwargs: Dict[str, Any]) -> str:
    for k, v in kwargs.items():
        s = s.replace("{" + k + "}", str(v))
    return s


@dataclass(frozen=True)
class Prompt:
    template: str
    response_format: Optional[Type[BaseModel]] = None
    pre_tuple: Tuple[Callable] = field(default_factory=tuple)
    post_tuple: Tuple[Callable] = field(default_factory=tuple)


@dataclass(frozen=True)
class PromptChain:
    curr_prompt: Optional[Prompt] = None
    prev_prompts: Tuple[Prompt] = field(default_factory=tuple)
    response_list: Tuple[Any] = field(default_factory=tuple)
    prev_fields: Dict[str, Any] = field(default_factory=dict)
    msg_chain_func: Optional[Callable] = None

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
    def post_last(self, **named_transformations):
        """Apply transformations to the last response and add the results to prev_fields"""
        if not self.response_list:
            return self

        last_response = self.response_list[-1]
        new_fields = {}

        for field_name, transform_func in named_transformations.items():
            # Simply apply the transform function to the last response
            # The transform function should be responsible for handling different types
            new_fields[field_name] = transform_func(last_response)

        return replace(self, prev_fields={**self.prev_fields, **new_fields})

    @chain_method
    def post_chain(self, transform_func):
        """Apply a transformation to the entire chain and add results to prev_fields"""
        new_fields = transform_func(self)
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

        # Extract field names from the template by finding all strings enclosed in curly braces
        field_names = re.findall(r"\{([^}]+)\}", self.curr_prompt.template)
        kwargs = {field: self.prev_fields.get(field, "") for field in field_names}

        prompt = replace_strs(self.curr_prompt.template, kwargs)

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
        # Add the current prompt to prev_prompts
        new_prev_prompts = self.prev_prompts + (self.curr_prompt,)

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
