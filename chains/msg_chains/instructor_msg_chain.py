from dataclasses import dataclass, field, replace
from typing import List, Dict, Union, Any, Optional, Tuple, Type
from openai import OpenAI
from functools import wraps
import os
from pydantic import BaseModel
import anthropic

from chains.utils import calc_cost

import instructor

def chain_method(func):
    """Decorator to convert a function into a chainable method."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        return func(self, *args, **kwargs)

    return wrapper


@dataclass(frozen=True)
class Message:
    content: Union[str, List[Dict[str, str]]]
    role: str
    should_cache: bool = False


@dataclass(frozen=True)
class InstructorMessageChain:
    model_name: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    messages: Tuple[Message] = field(default_factory=tuple)
    system_prompt: Any = None  
    cache_system: bool = False
    metric_list: List[Dict[str, Any]] = field(default_factory=tuple)
    response_list: List[Any] = field(default_factory=tuple)
    verbose: bool = False
    response_format: Optional[Any] = None
    base_url: Optional[str] = None
    max_tokens: int = 4096
    
    
    @chain_method
    def quiet(self):
        self = replace(self, verbose=False)
        return self

    @chain_method
    def verbose(self):
        self = replace(self, verbose=True)
        return self

    @chain_method
    def add_message(
        self,
        content: Union[str, List[Dict[str, str]]],
        role: str,
        should_cache: bool = False,
    ):
        assert not should_cache, "Instructor does not support caching"
        msg = Message(role=role, content=content, should_cache=should_cache)
        return replace(self, messages=self.messages + (msg,))

    @chain_method
    def user(
        self, content: Union[str, List[Dict[str, str]]], should_cache: bool = False
    ):
        return self.add_message(content=content, role="user", should_cache=should_cache)

    @chain_method
    def bot(
        self, content: Union[str, List[Dict[str, str]]], should_cache: bool = False
    ):
        return self.add_message(
            content=content, role="assistant", should_cache=should_cache
        )

    @chain_method
    def system(self, content: str, should_cache: bool = False):
        self = replace(self, system_prompt=content, cache_system=should_cache)
        return self

    @chain_method
    def with_structure(self, response_format: Type[BaseModel]):
        """Set a Pydantic model as the expected response format."""
        self = replace(self, response_format=response_format)
        return self

    def serialize(self) -> list:
        output = []
        if self.system_prompt is not None:
            output.append({"role": "system", "content": self.system_prompt})

        for m in self.messages:
            output.append({"role": m.role, "content": m.content})

        return output

    @staticmethod
    def parse_metrics(resp):
        return dict(
            input_tokens=resp.usage.prompt_tokens,
            output_tokens=resp.usage.completion_tokens,
            total_tokens=resp.usage.total_tokens,
            input_tokens_cache_read=0,  # OpenAI doesn't have cache metrics
            input_tokens_cache_create=0,
        )

    @chain_method
    def generate(self):
        if self.base_url is not None:
            client = OpenAI(base_url=self.base_url, api_key="lm-studio")
            client = instructor.from_openai(
                client,
                # mode=instructor.Mode.JSON,
            )
        elif "claude" in self.model_name:
            client = anthropic.AnthropicBedrock(
                    aws_region="us-west-2",
                )
            client = instructor.from_anthropic(client, mode=instructor.Mode.ANTHROPIC_TOOLS)
        elif "gpt" in self.model_name:
            client = OpenAI()
            client = instructor.from_openai(client)
        else:
            raise ValueError(f"Model {self.model_name} not supported")
        msgs = self.serialize()
        # assert self.response_format
        resp = client.messages.create(
            model=self.model_name,
            max_completion_tokens=self.max_tokens,
            messages=msgs,
            response_model=self.response_format,
        )

        self = replace(
            self,
            metric_list=(None,),
            response_list=self.response_list + (resp,),
        )
        return self

    # genrates and appends the last assistant message into the chain
    @chain_method
    def generate_bot(self):
        self = self.generate()
        self = self.bot(self.response_list[-1])
        return self

    @chain_method
    def emit_last(self):
        return self, self.response_list[-1], self.metric_list[-1]

    @chain_method
    def print_last(self, response=None, metrics=None, mode="response_all"):
        if mode == "response_all":
            if response is None:
                response = self.last_response
                metrics = self.last_metrics
            print(f"{response=}")
            print(f"{metrics=}")
        if mode == "full_completion":
            response = self.last_full_completion
            print(f"{response=}")

        return self

    @property
    def last_response(self):
        return self.response_list[-1]

    @property
    def last_metrics(self):
        return self.metric_list[-1]

    @property
    def last_full_completion(self):
        rev_messages = self.messages[::-1]
        output = []
        for msg in rev_messages:
            if msg.role == "user":
                break
            output.append(msg.content)
        return "".join(output[::-1])

    @chain_method
    def apply(self, func):
        func(self)
        return self

    @chain_method
    def map(self, func):
        return self.apply(func)

    def print_cost(self):
        calc_cost(self.metric_list)


def test_chain1():
    chain1 = InstructorMessageChain()
    chain1 = (
        chain1.user("Hello!")
        .bot("Hi there!")
        .user("How are you?")
        .generate()
        .print_last()
    )


def test_chain2():
    chain2 = InstructorMessageChain()
    chain2 = (
        chain2.user("Come up with a name, respond with a single word")
        .bot("Donny")
        .user("Tell me a story about Donny")
        .generate()
        .print_last()
    )


def test_system():
    chain2 = InstructorMessageChain()
    chain2 = (
        chain2.system("Answer in rhyming words")
        .user("Come up with a name, respond with a single word")
        .bot("Donny")
        .user("Tell me a story about Donny")
        .generate()
        .print_last()
    )


def test_generate_bot():
    chain = InstructorMessageChain()
    chain = (
        chain.system("Answer in rhyming words")
        .user("Come up with a name, respond with a single word")
        .bot("Donny")
        .user("Tell me a story about Donny")
        .generate_bot()
        .user("Repeat your last message in lowercase")
        .generate_bot()
        .print_last()
    )


def test_generate_bot_prefix():
    chain = InstructorMessageChain()
    chain = (
        chain.system("Answer in rhyming words")
        .user("Come up with a name, respond with a single word")
        .user("Tell me a story about Donny")
        .bot("Our story is about Donny")
        .generate_bot()
        .print_last()
        .user("Repeat your last message in lowercase")
        .generate_bot()
        .print_last()
    )
    # prints
    # response=",\nWho wasn't at all bright or brainy,\nHe tripped on his feet,\nWhile crossing the street,\nAnd landed in puddles quite rainy!"
    # ....
    # response="our story is about donny\n\nwho wasn't at all bright or brainy,\nhe tripped on his feet,\nwhile crossing the street,\nand landed in puddles quite rainy!"
    # ...


def test_apply_last():
    chain = InstructorMessageChain()
    chain = (
        chain.system("Answer in rhyming words")
        .user("Come up with a name, respond with a single word")
        .user("Tell me a story about Donny")
        .bot("Our story is about Donny")
        .generate_bot()
        .print_last(mode="full_completion")
        .user("Repeat your last message in lowercase")
        .generate_bot()
        .print_last()
    )
    # prints
    # response=",\nWho wasn't at all bright or brainy,\nHe tripped on his feet,\nWhile crossing the street,\nAnd landed in puddles quite rainy!"
    # ....
    # response="our story is about donny\n\nwho wasn't at all bright or brainy,\nhe tripped on his feet,\nwhile crossing the street,\nand landed in puddles quite rainy!"
    # ...


# test_apply_last()
# test_generate_bot_prefix()


def test_apply1():
    def print_response(chain):
        print("Custom print:", chain.response_list[-1])

    chain = InstructorMessageChain()
    chain = chain.user("Hello").generate().apply(print_response)


def test_apply2():
    resp_list = []

    def append_to(chain):
        resp_list.append(chain.response_list[-1])

    chain = InstructorMessageChain()
    chain = chain.user("Hello").generate().apply(append_to)

    print(resp_list)
    # .bot("Hi there!")


def test_structured_output():
    from pydantic import BaseModel

    class CalendarEvent(BaseModel):
        name: str
        date: str
        participants: list[str]

    chain = InstructorMessageChain()
    chain = (
        chain.system("Extract the event information.")
        .user("Alice and Bob are going to a science fair on Friday.")
        .with_structure(CalendarEvent)
        .generate()
        .print_last()
    )

    event = chain.last_response
    print(f"Event name: {event.name}")
    print(f"Event date: {event.date}")
    print(f"Participants: {', '.join(event.participants)}")

import fire

if __name__ == "__main__":
    fire.Fire()
