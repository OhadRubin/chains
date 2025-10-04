from dataclasses import dataclass, field, replace
import json
from typing import List, Dict, Union, Any, Optional, Tuple, Type
from openai import AsyncOpenAI
from functools import wraps
import inspect
import os
import base64
import httpx
import mimetypes
from pydantic import BaseModel
from asyncio import Queue

from chains.utils import calc_cost


def serialize_tool_call(tool_call) -> dict:
    """Convert a tool call object to a serializable dictionary."""
    if hasattr(tool_call, "model_dump"):  # Pydantic object
        return tool_call.model_dump()
    elif hasattr(tool_call, "dict"):  # Older pydantic
        return tool_call.dict()
    elif isinstance(tool_call, dict):
        return tool_call
    else:
        # Convert OpenAI ToolCall object to dict manually
        return {
            "id": getattr(tool_call, "id", None),
            "type": getattr(tool_call, "type", "function"),
            "function": {
                "name": (
                    getattr(tool_call.function, "name", "")
                    if hasattr(tool_call, "function")
                    else ""
                ),
                "arguments": (
                    getattr(tool_call.function, "arguments", "")
                    if hasattr(tool_call, "function")
                    else ""
                ),
            },
        }


async def encode_base64_content_from_url(content_url: str) -> str:
    """Asynchronously fetch content from a URL and encode it in base64."""

    async with httpx.AsyncClient() as client:
        response = await client.get(content_url)
        response.raise_for_status()
        result = base64.b64encode(response.content).decode("utf-8")

    return result


async def _resolve_multimodal_args(args: Dict[str, Any]) -> Dict[str, Any]:
    """Convert any URL fields in tool arguments to base64-encoded strings."""

    resolved = {}
    for key, value in args.items():
        if isinstance(value, list):
            resolved[key] = [
                (
                    await encode_base64_content_from_url(v)
                    if isinstance(v, str) and v.startswith("http")
                    else v
                )
                for v in value
            ]
        elif isinstance(value, str) and value.startswith("http"):
            resolved[key] = await encode_base64_content_from_url(value)
        else:
            resolved[key] = value
    return resolved


async def _encode_to_data_uri(source: str, mime_type: Optional[str] = None) -> str:
    """Encode a local file or remote URL to a base64 data URI."""

    if source.startswith("http"):
        async with httpx.AsyncClient() as client:
            response = await client.get(source)
            response.raise_for_status()
            content = response.content
            if not mime_type:
                mime_type = response.headers.get("content-type")
    else:
        with open(source, "rb") as f:
            content = f.read()
        if not mime_type:
            mime_type = mimetypes.guess_type(source)[0]

    mime_type = mime_type or "application/octet-stream"
    encoded = base64.b64encode(content).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


async def _resolve_multimodal_output(output: Any) -> Any:
    """Convert MCP multimodal content to OpenAI message format."""

    if isinstance(output, dict) and "content" in output:
        # Handle MCP multimodal content structure
        content_items = output["content"]

        # If there's only text content, return just the text
        text_items = [item for item in content_items if item.get("type") == "text"]
        image_items = [item for item in content_items if item.get("type") == "image"]

        if len(content_items) == 1 and content_items[0].get("type") == "text":
            return content_items[0].get("text", "")

        # For multimodal content, we'll return structured data that the chain can handle
        result = []

        for item in content_items:
            if item.get("type") == "text" and item.get("text"):
                result.append({"type": "text", "text": item["text"]})
            elif item.get("type") == "image" and item.get("data"):
                # Convert to OpenAI format
                mime_type = item.get("mimeType", "image/png")
                data_uri = f"data:{mime_type};base64,{item['data']}"
                result.append({"type": "image_url", "image_url": {"url": data_uri}})

        return {"multimodal_content": result}

    elif isinstance(output, str):
        if output.startswith("http") or os.path.exists(output):
            return await _encode_to_data_uri(output)
        return output
    elif isinstance(output, list):
        return [await _resolve_multimodal_output(v) for v in output]
    elif isinstance(output, dict):
        return {k: await _resolve_multimodal_output(v) for k, v in output.items()}
    else:
        return output


import json

def is_json_serializable(obj):
    """Check if an object is JSON serializable."""
    try:
        json.dumps(obj)
        return True
    except (TypeError, ValueError):
        return False


def chain_method(func):
    """Decorator to convert a function into a chainable method that supports
    both synchronous and asynchronous functions."""

    if inspect.iscoroutinefunction(func):
        @wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            return await func(self, *args, **kwargs)

        return async_wrapper
    else:
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
    
    def serialize(self, ser_tool_calls: bool = False):
        msg_dict = {"role": self.role}

        # Add content if it exists
        if self.content is not None:
            msg_dict["content"] = self.content

        # Add tool_calls for assistant messages
        if self.role == "assistant" and self.tool_calls is not None:
            if not ser_tool_calls:
                msg_dict["tool_calls"] = self.tool_calls
            else:
                msg_dict["tool_calls"] = [serialize_tool_call(tool_call) for tool_call in self.tool_calls]

        # Add tool_call_id for tool messages
        if self.role == "tool" and self.tool_call_id is not None:
            msg_dict["tool_call_id"] = self.tool_call_id

        # Add name for tool messages
        if self.role == "tool" and self.name is not None:
            msg_dict["name"] = self.name
        return msg_dict


import uuid
def generate_session_id():
    return str(uuid.uuid4())

@dataclass(frozen=True)
class OpenAIAsyncMessageChain:
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
    session_id: str = field(default_factory=generate_session_id)
    stream_queue: Optional[Queue] = None

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
        if self.stream_queue is not None:
            m = msg.serialize(ser_tool_calls=True)
            m["session_id"] = self.session_id
            self.stream_queue.put_nowait(m)
        return replace(self, messages=self.messages + (msg,))

    @chain_method
    def user(self, content: Union[str, List[Dict[str, str]]], should_cache: bool = False):
        return self.add_message(role="user", content=content, should_cache=should_cache)

    @chain_method
    def user_image_url(self, prompt: str, image_urls: List[str]):
        """Send a user message with a text prompt and one or more image URLs."""
        content = [{"type": "text", "text": prompt}] + [
            {"type": "image_url", "image_url": {"url": url}} for url in image_urls
        ]
        return self.user(content)

    @chain_method
    async def user_image_base64(self, prompt: str, image_urls: List[str]):
        """Send a user message with image content encoded in base64."""
        encoded_images = [
            await encode_base64_content_from_url(url) for url in image_urls
        ]
        content = [{"type": "text", "text": prompt}] + [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img}"},
            }
            for img in encoded_images
        ]
        return self.user(content)

    @chain_method
    async def user_image_file(self, prompt: str, image_paths: List[str]):
        """Send a user message with local image files encoded as base64 data URIs."""
        encoded_images = [await _encode_to_data_uri(path) for path in image_paths]
        content = [{"type": "text", "text": prompt}] + [
            {
                "type": "image_url",
                "image_url": {"url": data_uri},
            }
            for data_uri in encoded_images
        ]
        return self.user(content)

    @chain_method
    def user_audio_url(self, prompt: str, audio_urls: List[str]):
        """Send a user message with a text prompt and one or more audio URLs."""
        content = [{"type": "text", "text": prompt}] + [
            {"type": "audio_url", "audio_url": {"url": url}} for url in audio_urls
        ]
        return self.user(content)

    @chain_method
    async def user_audio_base64(
        self, prompt: str, audio_urls: List[str], mime_type: str = "audio/ogg"
    ):
        """Send a user message with audio content encoded in base64."""
        encoded_audio = [
            await encode_base64_content_from_url(url) for url in audio_urls
        ]
        content = [{"type": "text", "text": prompt}] + [
            {
                "type": "audio_url",
                "audio_url": {"url": f"data:{mime_type};base64,{audio}"},
            }
            for audio in encoded_audio
        ]
        return self.user(content)

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
        if self.stream_queue is not None:
            m = {"role": "system", "content": content}
            m["session_id"] = self.session_id
            self.stream_queue.put_nowait(m)
        return self

    @chain_method
    def with_structure(self, response_format: Type[BaseModel]):
        """Set a Pydantic model as the expected response format."""
        self = replace(self, response_format=response_format)
        return self

    @chain_method
    def with_tools(self, tools_list: List, tools_mapping: Dict[str, Any]):
        """Set a Pydantic model as the expected response format."""
        if self.stream_queue is not None:
            m = {"role": "tool_spec", "content": json.dumps(tools_list)}
            m["session_id"] = self.session_id
            self.stream_queue.put_nowait(m)
        self = replace(self, tools_list=tools_list, tools_mapping=tools_mapping)
        return self

    def serialize(self) -> list:
        output = []
        if self.system_prompt is not None:
            output.append({"role": "system", "content": self.system_prompt})

        for m in self.messages:
            output.append(m.serialize())

        return output

    @staticmethod
    def parse_metrics(resp):
        try:
            return dict(
                input_tokens=resp.usage.prompt_tokens,
                output_tokens=resp.usage.completion_tokens,
                total_tokens=resp.usage.total_tokens,
                input_tokens_cache_read=0,  # OpenAI doesn't have cache metrics
                input_tokens_cache_create=0
            )
        except Exception as e:
            # print(f"Error parsing metrics: {e}")
            return dict(
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                input_tokens_cache_read=0,  # OpenAI doesn't have cache metrics
                input_tokens_cache_create=0
            )

    @chain_method
    async def generate(self):
        while True:
            if self.base_url == "https://openrouter.ai/api/v1":
                client = AsyncOpenAI(
                    base_url=self.base_url, api_key=os.getenv("OPENROUTER_API_KEY")
                )
            elif self.base_url is not None:
                client = AsyncOpenAI(
                    base_url=self.base_url, api_key="lm-studio"
                )  # Or a configurable key for other base URLs
            else:
                client = AsyncOpenAI()
            msgs = self.serialize()

            # Prepare common parameters
            api_params = {
                "model": self.model_name,
                "messages": msgs,
                "max_tokens": self.max_tokens,
                "temperature": 1.0,
            }

            # Only add tools if they exist
            if self.tools_list is not None:
                api_params["tools"] = self.tools_list

            if self.response_format:
                # Use structured output with parse
                response = await client.beta.chat.completions.parse(
                    response_format=self.response_format, **api_params
                )
                # For structured output, get the parsed response
                resp = response.choices[0].message.parsed
                msg = response.choices[0].message
            else:
                response = await client.chat.completions.create(**api_params)
                msg = response.choices[0].message
                resp = msg.content

            self = replace(
                self,
                metric_list=self.metric_list + (self.parse_metrics(response),),
                response_list=self.response_list + (resp,),
            )
            if msg.tool_calls and len(msg.tool_calls) > 0:
                self = await self.handle_tool_calls(msg)
            else:
                break
            
        return self

    async def handle_tool_calls(self, msg):
        # Add the assistant message with tool calls to the conversation
        self = self.bot(content=msg.content, tool_calls=msg.tool_calls)
        if msg.content and self.verbose:
            print(f"<bot_thinking>{msg.content}</bot_thinking>")
        skip_tool_calls = False

        # Execute each tool call and add the results
        for tool_call in msg.tool_calls:
            if self.verbose:
                print(f"<tool_call>{json.dumps(serialize_tool_call(tool_call))}</tool_call>")
            tool_name = tool_call.function.name

            if tool_call.function.arguments:
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                    tool_args = await _resolve_multimodal_args(tool_args)
                except Exception as e:
                    skip_tool_calls = True
            else:
                tool_args = {}

            # Execute the tool function
            if self.tools_mapping and tool_name in self.tools_mapping:
                if not skip_tool_calls:

                    tool_response = await self.tools_mapping[tool_name](**tool_args)
                    tool_response = await _resolve_multimodal_output(tool_response)
                else:
                    tool_response = "Error: Tool arguments are not valid JSON"

                # Handle multimodal content specially
                if (
                    isinstance(tool_response, dict)
                    and "multimodal_content" in tool_response
                ):
                    # Add multimodal content as a user message so the LLM can see images
                    multimodal_content = tool_response["multimodal_content"]
                    if self.verbose:
                        print("<tool_response_multimodal>Tool response contained multimodal content</tool_response_multimodal>")

                    # Add tool result as text first
                    text_parts = [
                        item["text"]
                        for item in multimodal_content
                        if item.get("type") == "text"
                    ]
                    tool_text = (
                        " ".join(text_parts)
                        if text_parts
                        else f"Tool {tool_name} executed successfully"
                    )
                    if self.verbose:
                        print(f"<tool_text>{tool_text}</tool_text>")

                    self = self.tool(
                        content=tool_text,
                        tool_call_id=tool_call.id,
                        name=tool_name,
                    )

                    # Then add the multimodal content as a user message
                    if any(
                        item.get("type") == "image_url"
                        for item in multimodal_content
                    ):
                        user_prompt = f"Here's the result from {tool_name}:"
                        self = self.user(
                            [{"type": "text", "text": user_prompt}]
                            + multimodal_content
                        )
                else:
                    # Convert tool response to string for the API
                    tool_response_str = (
                        json.dumps(tool_response)
                        if not isinstance(tool_response, str)
                        else tool_response
                    )
                    if self.verbose:
                        print(f"<tool_response>{tool_response_str}</tool_response>")

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
        return self

    # genrates and appends the last assistant message into the chain
    @chain_method
    async def generate_bot(self):
        self = await self.generate()
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

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary, excluding non-serializable fields."""


        # Convert messages to serializable format
        serialized_messages = []
        for msg in self.messages:
            msg_dict = {
                "role": msg.role,
                "tool_call_id": msg.tool_call_id,
                "name": msg.name,
                "should_cache": msg.should_cache,
            }

            # Handle content carefully - only include if serializable
            if msg.content is not None and is_json_serializable(msg.content):
                msg_dict["content"] = msg.content
            else:
                msg_dict["content"] = None

            # Convert tool_calls to serializable format (dict instead of ToolCall objects)
            if msg.tool_calls:
                msg_dict["tool_calls"] = []
                for tool_call in msg.tool_calls:
                    if hasattr(tool_call, "model_dump"):  # Pydantic object
                        msg_dict["tool_calls"].append(tool_call.model_dump())
                    elif hasattr(tool_call, "dict"):  # Older pydantic
                        msg_dict["tool_calls"].append(tool_call.dict())
                    elif isinstance(tool_call, dict):
                        msg_dict["tool_calls"].append(tool_call)
                    else:
                        # Convert OpenAI ToolCall object to dict manually
                        msg_dict["tool_calls"].append(
                            {
                                "id": getattr(tool_call, "id", None),
                                "type": getattr(tool_call, "type", "function"),
                                "function": {
                                    "name": (
                                        getattr(tool_call.function, "name", "")
                                        if hasattr(tool_call, "function")
                                        else ""
                                    ),
                                    "arguments": (
                                        getattr(tool_call.function, "arguments", "{}")
                                        if hasattr(tool_call, "function")
                                        else "{}"
                                    ),
                                },
                            }
                        )
            else:
                msg_dict["tool_calls"] = None

            serialized_messages.append(msg_dict)

        # Only include serializable response_list items
        serializable_responses = []
        for response in self.response_list:
            if is_json_serializable(response):
                serializable_responses.append(response)
            else:
                # Convert to string representation for non-serializable objects
                serializable_responses.append(str(response))

        return {
            "model_name": self.model_name,
            "messages": serialized_messages,
            "system_prompt": (
                self.system_prompt
                if is_json_serializable(self.system_prompt)
                else str(self.system_prompt) if self.system_prompt is not None else None
            ),
            "cache_system": self.cache_system,
            "metric_list": list(self.metric_list),
            "response_list": serializable_responses,
            "base_url": self.base_url,
            "max_tokens": self.max_tokens,
            # Skip verbose, response_format and tools_mapping as they're not needed
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OpenAIAsyncMessageChain":
        """Deserialize from dictionary."""
        # Convert messages back to Message objects
        messages = []
        for msg_data in data.get("messages", []):
            msg = Message(
                role=msg_data["role"],
                content=msg_data.get("content"),
                tool_calls=msg_data.get("tool_calls"),  # Keep as dicts
                tool_call_id=msg_data.get("tool_call_id"),
                name=msg_data.get("name"),
                should_cache=msg_data.get("should_cache", False),
            )
            messages.append(msg)

        return cls(
            model_name=data.get("model_name", "gpt-4o"),
            messages=tuple(messages),
            system_prompt=data.get("system_prompt"),
            cache_system=data.get("cache_system", False),
            metric_list=tuple(data.get("metric_list", [])),
            response_list=tuple(data.get("response_list", [])),
            verbose=data.get("verbose", False),
            base_url=data.get("base_url"),
            max_tokens=data.get("max_tokens", 4096),
            # response_format and tools_mapping will be None
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        import json

        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "OpenAIAsyncMessageChain":
        """Deserialize from JSON string."""
        import json

        return cls.from_dict(json.loads(json_str))


def test_chain1():
    chain1 = OpenAIAsyncMessageChain()
    chain1 = (
        chain1
        .user("Hello!")
        .bot("Hi there!")
        .user("How are you?")
        .generate().print_last()
    )


def test_chain2():
    chain2 = OpenAIAsyncMessageChain()
    chain2 = (
        chain2
        .user("Come up with a name, respond with a single word")
        .bot("Donny")
        .user("Tell me a story about Donny")
        .generate().print_last()
    )

def test_system():
    chain2 = OpenAIAsyncMessageChain()
    chain2 = (
        chain2
        .system("Answer in rhyming words")
        .user("Come up with a name, respond with a single word")
        .bot("Donny")
        .user("Tell me a story about Donny")
        .generate().print_last()
    )

def test_generate_bot():
    chain = OpenAIAsyncMessageChain()
    chain = (
        chain
        .system("Answer in rhyming words")
        .user("Come up with a name, respond with a single word").bot("Donny")
        .user("Tell me a story about Donny").generate_bot()
        .user("Repeat your last message in lowercase").generate_bot()
        .print_last()
    )


def test_generate_bot_prefix():
    chain = OpenAIAsyncMessageChain()
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
    chain = OpenAIAsyncMessageChain()
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
        
    chain = OpenAIAsyncMessageChain()
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
    chain = OpenAIAsyncMessageChain()
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

    chain = OpenAIAsyncMessageChain(model_name="gpt-4o-2024-08-06")
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


def test_serialization():
    """Test that serialization and deserialization work correctly."""
    # Create a chain with various message types
    chain = OpenAIAsyncMessageChain(model_name="gpt-4o")
    chain = (
        chain.system("You are a helpful assistant")
        .user("Hello!")
        .bot("Hi there! How can I help you?")
        .user("What's 2+2?")
    )

    # Add some mock data to test serialization
    chain = replace(
        chain,
        response_list=("Hi there! How can I help you?", "2+2 equals 4"),
        metric_list=(
            {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            {"input_tokens": 8, "output_tokens": 3, "total_tokens": 11},
        ),
    )

    # Test serialization
    json_str = chain.to_json()
    print("Serialized chain:")
    print(json_str[:200] + "..." if len(json_str) > 200 else json_str)

    # Test deserialization
    restored_chain = OpenAIAsyncMessageChain.from_json(json_str)

    # Verify the data matches
    assert restored_chain.model_name == chain.model_name
    assert restored_chain.system_prompt == chain.system_prompt
    assert len(restored_chain.messages) == len(chain.messages)
    assert restored_chain.response_list == chain.response_list
    assert restored_chain.metric_list == chain.metric_list

    print("✅ Serialization test passed!")


def test_serialization_debug():
    """Debug serialization issues by testing each field individually."""
    import json

    # Create a minimal chain
    chain = OpenAIAsyncMessageChain(model_name="gpt-4o")
    chain = chain.system("Test").user("Hello")

    print("Testing individual fields:")

    # Test each field individually
    test_data = {
        "model_name": chain.model_name,
        "system_prompt": chain.system_prompt,
        "cache_system": chain.cache_system,
        "verbose": chain.verbose,
        "base_url": chain.base_url,
        "max_tokens": chain.max_tokens,
    }

    for key, value in test_data.items():
        try:
            json.dumps({key: value})
            print(f"✅ {key}: OK")
        except Exception as e:
            print(f"❌ {key}: {type(value)} - {e}")

    # Test messages
    try:
        messages_data = []
        for i, msg in enumerate(chain.messages):
            msg_dict = {
                "role": msg.role,
                "content": msg.content,
                "tool_calls": msg.tool_calls,
                "tool_call_id": msg.tool_call_id,
                "name": msg.name,
                "should_cache": msg.should_cache,
            }
            try:
                json.dumps(msg_dict)
                print(f"✅ Message {i}: OK")
            except Exception as e:
                print(f"❌ Message {i}: {e}")
                # Test each field in the message
                for field, val in msg_dict.items():
                    try:
                        json.dumps({field: val})
                        print(f"  ✅ {field}: OK")
                    except Exception as fe:
                        print(f"  ❌ {field}: {type(val)} - {fe}")
    except Exception as e:
        print(f"❌ Messages: {e}")

    # Test tuples
    try:
        json.dumps(list(chain.metric_list))
        print("✅ metric_list: OK")
    except Exception as e:
        print(f"❌ metric_list: {e}")

    try:
        json.dumps(list(chain.response_list))
        print("✅ response_list: OK")
    except Exception as e:
        print(f"❌ response_list: {e}")


async def test_image_serialization():
    """Test image handling with serialization."""
    # Create initial chain with image
    chain = OpenAIAsyncMessageChain(model_name="gpt-4o")
    chain = await chain.user_image_file(
        "Describe this image in detail.",
        [
            "/Users/ohadr/chains/a_solid_black_silhouette_of_a_a_man_and_woman_holding_hands__-shading__sky_2061071959.png"
        ],
    )

    # Get initial description
    chain = await chain.generate_bot()
    print("\nInitial description:")
    print(chain.last_response)

    # Serialize the chain
    json_str = chain.to_json()
    print("\nSerialized chain (truncated):")
    print(json_str[:200] + "..." if len(json_str) > 200 else json_str)

    # Deserialize and ask follow-up
    restored_chain = OpenAIAsyncMessageChain.from_json(json_str)
    restored_chain = restored_chain.user(
        "What is the woman holding? Answer in one word."
    )
    restored_chain = await restored_chain.generate_bot()
    print("\nFollow-up answer about what she's holding:")
    print(restored_chain.last_response)

    # Ask about the type
    restored_chain = restored_chain.user("What type is it?")
    restored_chain = await restored_chain.generate_bot()
    print("\nFollow-up about the type:")
    print(restored_chain.last_response)

    print("\n✅ Image serialization test completed!")


if __name__ == "__main__":
    import asyncio

    async def main():
        await test_image_serialization()

    asyncio.run(main())
