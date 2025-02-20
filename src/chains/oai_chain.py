from dataclasses import dataclass, field, replace
from typing import List, Dict, Union, Any, Optional, Tuple
from openai import OpenAI
from functools import wraps
import os
# import persistence
import tenacity


from src.utils import calc_cost


def chain_method(func):
    """Decorator to convert a function into a chainable method."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        return func(self, *args, **kwargs)
    return wrapper

@dataclass(frozen=True)
class Message:
    content: List[Dict[str, str]]
    role: str
    should_cache: bool = False

@dataclass(frozen=True)
class OpenAIMessageChain:
    model_name: str = "gpt-4"
    messages: Tuple[Message] = field(default_factory=tuple)
    system_prompt: Any = None  # Changed from anthropic.NOT_GIVEN
    cache_system: bool = False
    metric_list: List[Dict[str, Any]] = field(default_factory=tuple)
    response_list: List[str] = field(default_factory=tuple)
    verbose: bool = False
    
    @chain_method
    def quiet(self):
        self = replace(self, verbose=False)
        return self
    
    @chain_method
    def verbose(self):
        self = replace(self, verbose=True)
        return self
    
    @chain_method
    def add_message(self, content: str, role: str, should_cache: bool = False):
        assert not should_cache, "OpenAI does not support caching"
        msg = Message(role=role, content=content, should_cache=should_cache)
        return replace(self, messages=self.messages + (msg,))

    @chain_method
    def user(self, content: str, should_cache: bool = False):
        return self.add_message(content=content, role="user", should_cache=should_cache)

    @chain_method
    def bot(self, content: str, should_cache: bool = False):
        return self.add_message(content=content, role="assistant", should_cache=should_cache)
    
    @chain_method
    def system(self, content: str, should_cache: bool = False):
        self = replace(self, system_prompt=content, cache_system=should_cache)
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
            input_tokens_cache_create=0
        )

    @chain_method
    def generate(self):
        client = OpenAI()
        msgs = self.serialize()
        
        response = client.chat.completions.create(
            model=self.model_name,  # or your preferred model
            messages=msgs,
            max_tokens=4096,
            temperature=1.0
        )
        
        resp = response.choices[0].message.content
        self = replace(self, 
                      metric_list=self.metric_list + (self.parse_metrics(response),),
                      response_list=self.response_list + (resp,)
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
        if mode=="full_completion":
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
    chain1 = OpenAIMessageChain()
    chain1 = (
        chain1
        .user("Hello!")
        .bot("Hi there!")
        .user("How are you?")
        .generate().print_last()
    )
    

def test_chain2():
    chain2 = OpenAIMessageChain()
    chain2 = (
        chain2
        .user("Come up with a name, respond with a single word")
        .bot("Donny")
        .user("Tell me a story about Donny")
        .generate().print_last()
    )
    
def test_system():
    chain2 = OpenAIMessageChain()
    chain2 = (
        chain2
        .system("Answer in rhyming words")
        .user("Come up with a name, respond with a single word")
        .bot("Donny")
        .user("Tell me a story about Donny")
        .generate().print_last()
    )
    
def test_generate_bot():
    chain = OpenAIMessageChain()
    chain = (
        chain
        .system("Answer in rhyming words")
        .user("Come up with a name, respond with a single word").bot("Donny")
        .user("Tell me a story about Donny").generate_bot()
        .user("Repeat your last message in lowercase").generate_bot()
        .print_last()
    )
        
    
def test_generate_bot_prefix():
    chain = OpenAIMessageChain()
    chain = (
        chain
        .system("Answer in rhyming words")
        .user("Come up with a name, respond with a single word")
        .user("Tell me a story about Donny").bot("Our story is about Donny").generate_bot().print_last()
        .user("Repeat your last message in lowercase").generate_bot()
        .print_last()
    )
    # prints
    # response=",\nWho wasn't at all bright or brainy,\nHe tripped on his feet,\nWhile crossing the street,\nAnd landed in puddles quite rainy!"
    # ....
    # response="our story is about donny\n\nwho wasn't at all bright or brainy,\nhe tripped on his feet,\nwhile crossing the street,\nand landed in puddles quite rainy!"
    # ...
    
def test_apply_last():
    chain = OpenAIMessageChain()
    chain = (
        chain
        .system("Answer in rhyming words")
        .user("Come up with a name, respond with a single word")
        .user("Tell me a story about Donny").bot("Our story is about Donny").generate_bot().print_last(mode="full_completion")
        .user("Repeat your last message in lowercase").generate_bot()
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
        
    chain = OpenAIMessageChain()
    chain = (
        chain
        .user("Hello")
        .generate()
        .apply(print_response)
    )
def test_apply2():
    resp_list = []
    def append_to(chain):
        resp_list.append(chain.response_list[-1])
    chain = OpenAIMessageChain()
    chain = (
        chain
        .user("Hello")
        .generate()
        .apply(append_to)
    )
    
    print(resp_list)
    # .bot("Hi there!")
    

