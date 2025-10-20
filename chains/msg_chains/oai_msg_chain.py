from dataclasses import dataclass, field, replace
import json
from typing import List, Dict, Union, Any, Optional, Tuple, Type
from openai import OpenAI
from functools import wraps
import os
from pydantic import BaseModel


from chains.utils import calc_cost


def chain_method(func):
    """Decorator to convert a function into a chainable method."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        return func(self, *args, **kwargs)
    return wrapper


@dataclass(frozen=True)
class Message:
    role: str
    content: Optional[Union[str, List[Dict[str, str]]]] = None
    tool_calls: Optional[List[Any]] = (
        None  # Can store OpenAI's ToolCall objects or dicts
    )
    tool_call_id: Optional[str] = None  # For role 'tool'
    name: Optional[str] = None  # For tool messages to specify function name
    should_cache: bool = False


@dataclass(frozen=True)
class OpenAIMessageChain:
    model_name: str = "gpt-4o"
    messages: Tuple[Message] = field(default_factory=tuple)
    system_prompt: Any = None  # Changed from anthropic.NOT_GIVEN
    cache_system: bool = False
    metric_list: List[Dict[str, Any]] = field(default_factory=tuple)
    response_list: List[Any] = field(default_factory=tuple)
    verbose: bool = False
    response_format: Optional[Any] = None
    tools_list: Optional[List[Any]] = None
    tools_mapping: Optional[Dict[str, Any]] = None
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
        role: str,
        content: Optional[Union[str, List[Dict[str, Any]], BaseModel]] = None,
        tool_calls: Optional[List[Any]] = None,
        tool_call_id: Optional[str] = None,
        name: Optional[str] = None,
        should_cache: bool = False,
    ):
        assert (
            not should_cache
        ), "OpenAI does not support caching for individual messages in this way"
        # Ensure content is not assigned if it's meant to be None (e.g. assistant message with only tool_calls)
        msg = Message(
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
            name=name,
            should_cache=should_cache,
        )
        return replace(self, messages=self.messages + (msg,))

    @chain_method
    def user(self, content: Union[str, List[Dict[str, str]]], should_cache: bool = False):
        return self.add_message(role="user", content=content, should_cache=should_cache)

    @chain_method
    def bot(
        self,
        content: Optional[Union[str, List[Dict[str, Any]], BaseModel]] = None,
        tool_calls: Optional[List[Any]] = None,
        should_cache: bool = False,
    ):
        return self.add_message(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            should_cache=should_cache,
        )

    @chain_method
    def tool(
        self,
        content: str,
        tool_call_id: str,
        name: Optional[str] = None,
        should_cache: bool = False,
    ):  # content for tool is stringified result
        return self.add_message(
            role="tool",
            content=content,
            tool_call_id=tool_call_id,
            name=name,
            should_cache=should_cache,
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

    @chain_method
    def with_tools(self, tools_list: List, tools_mapping: Dict[str, Any]):
        """Set a Pydantic model as the expected response format."""
        self = replace(self, tools_list=tools_list, tools_mapping=tools_mapping)
        return self

    def serialize(self) -> list:
        output = []
        if self.system_prompt is not None:
            output.append({"role": "system", "content": self.system_prompt})

        for m in self.messages:
            msg_dict = {"role": m.role}

            # Add content if it exists
            if m.content is not None:
                msg_dict["content"] = m.content

            # Add tool_calls for assistant messages
            if m.role == "assistant" and m.tool_calls is not None:
                msg_dict["tool_calls"] = m.tool_calls

            # Add tool_call_id for tool messages
            if m.role == "tool" and m.tool_call_id is not None:
                msg_dict["tool_call_id"] = m.tool_call_id

            # Add name for tool messages
            if m.role == "tool" and m.name is not None:
                msg_dict["name"] = m.name

            output.append(msg_dict)

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
        while True:
            if self.base_url == "https://openrouter.ai/api/v1":
                client = OpenAI(
                    base_url=self.base_url, api_key=os.getenv("OPENROUTER_API_KEY")
                )
            elif self.base_url is not None:
                client = OpenAI(
                    base_url=self.base_url, api_key="lm-studio"
                )  # Or a configurable key for other base URLs
            else:
                client = OpenAI()
            msgs = self.serialize()

            # Prepare common parameters
            api_params = {
                "model": self.model_name,
                "messages": msgs,
                # "temperature": 1.0,
            }

            # Only add max_tokens if it exists
            if self.max_tokens is not None:
                api_params["max_completion_tokens"] = self.max_tokens

            # Only add tools if they exist
            if self.tools_list is not None:
                api_params["tools"] = self.tools_list

            if self.response_format:
                # Use structured output with parse
                response = client.beta.chat.completions.parse(
                    response_format=self.response_format, **api_params
                )
                # For structured output, get the parsed response
                resp = response.choices[0].message.parsed
                msg = response.choices[0].message
            else:
                response = client.chat.completions.create(**api_params)
                msg = response.choices[0].message
                resp = msg.content

            self = replace(
                self,
                metric_list=self.metric_list + (self.parse_metrics(response),),
                response_list=self.response_list + (resp,),
            )

            # Check if the assistant made tool calls
            if msg.tool_calls and len(msg.tool_calls) > 0:
                # Add the assistant message with tool calls to the conversation
                self = self.bot(content=msg.content, tool_calls=msg.tool_calls)

                # Execute each tool call and add the results
                for tool_call in msg.tool_calls:
                    print(f"Tool call: {tool_call}")
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)

                    # Execute the tool function
                    if self.tools_mapping and tool_name in self.tools_mapping:

                        tool_response = self.tools_mapping[tool_name](**tool_args)
                        # Convert tool response to string for the API
                        tool_response_str = (
                            json.dumps(tool_response)
                            if not isinstance(tool_response, str)
                            else tool_response
                        )
                        print(f"Tool response: {tool_response_str}")

                        # Add tool result with proper tool_call_id and function name
                        self = self.tool(
                            content=tool_response_str,
                            tool_call_id=tool_call.id,
                            name=tool_name,
                        )
                    else:
                        # Handle case where tool is not found
                        error_msg = f"Tool '{tool_name}' not found in tools_mapping"
                        self = self.tool(
                            content=error_msg, tool_call_id=tool_call.id, name=tool_name
                        )
            else:
                # No tool calls, we're done
                break

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

def test_structured_output():
    from pydantic import BaseModel

    class CalendarEvent(BaseModel):
        name: str
        date: str
        participants: list[str]

    chain = OpenAIMessageChain(model_name="gpt-4o-2024-08-06")
    chain = (
        chain
        .system("Extract the event information.")
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
