from dataclasses import dataclass, field, replace
from typing import List, Dict, Union, Any, Optional, Tuple
from functools import wraps
import os
import tenacity

import os
import google.generativeai as genai
from tenacity import retry, wait_exponential, stop_after_attempt, wait_fixed, wait_random, retry_if_exception
import tempfile
import hashlib
from appdirs import user_cache_dir

CACHE_PATH = user_cache_dir("gemini_chain", "gemini_chain")
os.makedirs(CACHE_PATH, exist_ok=True)


def chain_method(func):
    """Decorator to convert a function into a chainable method."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        return func(self, *args, **kwargs)
    return wrapper

@dataclass(frozen=True)
class Message:
    content: str
    role: str
    should_cache: bool = False


@dataclass(frozen=True)
class GeminiMessageChain:
    messages: Tuple[Message] = field(default_factory=tuple)
    system_prompt: str = ""
    cache_system: bool = False
    metric_list: List[Dict[str, Any]] = field(default_factory=tuple)
    response_list: List[str] = field(default_factory=tuple)
    verbose: bool = False

    
    def cache_hash(self):
        first_msg = self.messages[0]
        to_cache_str = []
        # either we cache the first user message or the system prompt or both
        if self.cache_system:
            to_cache_str.append(f"System: {self.system_prompt}")
        if first_msg.should_cache:
            to_cache_str.append(f"User: {first_msg.content}")
        to_cache_str = "\n".join(to_cache_str)
        cache_hash = hashlib.sha256(to_cache_str.encode()).hexdigest()[:8]
        return cache_hash
    
    def _create_model(self):
        """Create cache from messages marked for caching - only for system and first user message"""
        models = [
            'models/gemini-1.5-pro-001',
            'models/gemini-1.5-pro-002', 
            'models/gemini-1.5-flash-001',
            'models/gemini-1.5-flash-002',
            'models/gemini-1.5-flash-8b',
            'models/gemini-1.5-flash-8b-001',
            'models/gemini-1.5-flash-8b-latest'
        ]
        assert len(self.messages) > 0
        assert self.messages[0].role == "user"        
        
        if self.messages[0].should_cache or self.cache_system:
            cache_hash = self.cache_hash()
            cache_path = os.path.join(CACHE_PATH, f"{cache_hash}.md")
            if self.messages[0].should_cache:
                cached_content = self.messages[0].content
            else:
                cached_content = "system_sentinel"
            
            # if we didn't already created the cache, create a new cache
            if not os.path.exists(cache_path):
                print(f"Cache miss. Creating cache at {cache_path}")
                # Create temporary file and cache
                with open(cache_path, "w") as f:
                    f.write(cached_content)
                
                if self.messages[0].should_cache:
                    docs = [genai.upload_file(path=cache_path)]
                else:
                    docs = None
                cache = genai.caching.CachedContent.create(
                    model=models[3],
                    system_instruction=self.system_prompt if self.cache_system else None,
                    contents=docs,
                )
                cache_name = cache.name
                # Write cache name to file
                
                with open(f"{cache_path}_name", "w") as f:
                    f.write(cache_name)
                    f.flush()
                    
                # Print cache file content
                print(f"Cache content at {cache_path}:")
                with open(cache_path) as f:
                    print(f.read())
            else:
                print(f"Cache hit. Using cache at {cache_path}")
                cache_name = open(f"{cache_path}_name").read()
                cache = genai.caching.CachedContent.get(cache_name)
            model = genai.GenerativeModel.from_cached_content(cache)
            
        else:
            model = genai.GenerativeModel('gemini-1.5-pro')
        return model

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
        messages = self.messages
        
        if messages[0].should_cache:
            messages = messages[1:]
            for m in messages:
                assert not m.should_cache, "We only cache the first user message"
        for m in messages:
            role = "model" if m.role == "assistant" else m.role
            output.append(dict(role=role, parts=[m.content]))
        return output

    @staticmethod
    def parse_metrics(resp):
        return dict(
            total_tokens=getattr(resp, 'total_tokens', 0),
            prompt_tokens=getattr(resp, 'prompt_tokens', 0),
            completion_tokens=getattr(resp, 'completion_tokens', 0)
        )

    @chain_method
    # @tenacity.retry(stop=tenacity.stop_after_attempt(10), wait=tenacity.wait_exponential(multiplier=1, min=4, max=15))
    def generate(self):
        # Create new cache if there are messages to cache
        model = self._create_model()
        
        msgs = self.serialize()
        response = model.generate_content(msgs)
        resp = response.text
        
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
    def print_last(self, response=None, metrics=None, mode="full_completion"):
        
        metrics = self.last_metrics if metrics is None else metrics
        if mode == "response_all":
            response = self.last_response if response is None else response
        if mode=="full_completion":
            response = self.last_full_completion
        print(f"{response=}")
        print(f"{metrics=}")
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
    
    @chain_method
    def set_cache(self, cache):
        """Set the cached content for the chain"""
        return replace(self, cached_content=cache)
    
    
    
    
def test_chain1():
    chain1 = GeminiMessageChain()
    chain1 = (
        chain1
        .user("Hello!")
        .bot("Hi there!")
        .user("How are you?")
        .generate().print_last()
    )
    

def test_chain2():
    chain2 = GeminiMessageChain()
    chain2 = (
        chain2
        .user("Come up with a name, respond with a single word")
        .bot("Donny")
        .user("Tell me a story about Donny")
        .generate().print_last()
    )
    
def test_system():
    chain2 = GeminiMessageChain()
    chain2 = (
        chain2
        .system("Answer in rhyming words")
        .user("Come up with a name, respond with a single word")
        .bot("Donny")
        .user("Tell me a story about Donny")
        .generate().print_last()
    )
    
def test_generate_bot():
    chain = GeminiMessageChain()
    chain = (
        chain
        .system("Answer in rhyming words")
        .user("Come up with a name, respond with a single word").bot("Donny")
        .user("Tell me a story about Donny").generate_bot()
        .user("Repeat your last message in lowercase").generate_bot()
        .print_last()
    )
        
    
def test_generate_bot_prefix():
    chain = GeminiMessageChain()
    chain = (
        chain
        .system("Answer in rhyming words")
        .user("Come up with a name, respond with a single word")
        .user("Tell me a story about Donny").bot("Our story is about Donny the monkey").generate_bot().print_last(mode="full_completion")
        .user("Repeat your last message in lowercase").generate_bot()
        .print_last(mode="full_completion")
    )
    # prints
    # response=",\nWho wasn't at all bright or brainy,\nHe tripped on his feet,\nWhile crossing the street,\nAnd landed in puddles quite rainy!"
    # ....
    # response="our story is about donny\n\nwho wasn't at all bright or brainy,\nhe tripped on his feet,\nwhile crossing the street,\nand landed in puddles quite rainy!"
    # ...
    
def test_apply_last():
    chain = GeminiMessageChain()
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
        
    chain = GeminiMessageChain()
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
    chain = GeminiMessageChain()
    chain = (
        chain
        .user("Hello")
        .generate()
        .apply(append_to)
    )
    
    print(resp_list)
    # .bot("Hi there!")

def test_caching():
    # Read the long file
    with open("/home/ohadr/general_o1/scratch/to_infer.txt") as f:
        long_content = f.read()
    
    (
        GeminiMessageChain()
        .system(long_content, should_cache=True)  # System prompt will be cached
        .user("repeat line 04 in input 3").generate_bot()
        .print_last()
        .user("repeat line 16 in input 4").generate_bot()
        .print_last()
    )
    
import fire
# usage: python gemini_msg_chain.py test_apply2
if __name__ == "__main__":
    fire.Fire()