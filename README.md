# Message Chain Framework

A unified, chainable interface for working with multiple LLM providers (Anthropic Claude, Google Gemini, and OpenAI).

## Install

```bash
pip install anthropic "google-generativeai>=0.3.0" openai tenacity appdirs
```

## Quick Start

```python
from chains.chain import MessageChain

# Create a chain for your preferred model
chain = MessageChain.get_chain(model="claude-3-5-sonnet")

# Build a conversation
result = (chain
    .system("You are a helpful assistant.")
    .user("What is the capital of France?")
    .generate_bot()  # Generate response and add to chain
    .user("And what about Germany?")
    .generate_bot()
)

# Get the last response
print(result.last_response)

# Print cost metrics
result.print_cost()
```

## Key Features

- **Immutable API**: Each method returns a new instance for clean chaining
- **Multiple Providers**: Unified interface for Claude, Gemini, and OpenAI
- **Caching**: Support for reducing costs with Claude and Gemini
- **Metrics**: Track token usage and costs
- **Custom Operations**: Apply functions with `.apply()` or `.map()`
- **Structured Output**: Generate Pydantic models directly from prompts
- **Single Chain Workflows**: One chain flows through all operations with shared state
- **MCP Integration**: Connect to Model Context Protocol (MCP) servers for tool support
- **Prompt Pipelines**: Build complex multi-stage prompt workflows with `chains.prompts`

## Basic Methods

```python
chain = (chain
    .system("System instructions")       # Set system prompt
    .user("User message")                # Add user message
    .bot("Assistant message")            # Add assistant message
    .generate()                          # Generate response
    .generate_bot()                      # Generate + add as bot message
    .quiet()/.verbose()                  # Toggle verbosity
    .apply(custom_function)              # Run custom function on chain
)

# Access data
response = chain.last_response
metrics = chain.last_metrics
full_text = chain.last_full_completion
```

## Structured Output with Pydantic

Generate structured data directly from prompts using `.with_structure()`:

```python
from pydantic import BaseModel, Field
from typing import List

class Attribute(BaseModel):
    name: str = Field(..., description="Name of the attribute")
    description: str = Field(..., description="Description of the attribute")
    importance_rank: int = Field(..., description="Importance ranking")

class AttributeList(BaseModel):
    attributes: List[Attribute] = Field(..., description="List of attributes")

# Generate structured output
result = (
    MessageChain.get_chain(model="gpt-4o")
    .system("You are a helpful assistant.")
    .user("List 5 quality attributes for a good blog post")
    .with_structure(AttributeList)  # ← Key method for structured output
    .generate()
    .print_last()
)

# Access structured data
attributes = result.last_response  # This is an AttributeList object
for attr in attributes.attributes:
    print(f"{attr.name}: {attr.description}")
```

## MCP Integration

The framework includes support for Model Context Protocol (MCP) servers, enabling LLMs to access external tools and data sources. This allows you to create powerful AI agents that can interact with real systems.

### Features

- **Tool Discovery**: Automatically discover tools from MCP servers
- **Async Tool Execution**: Execute tools with retry mechanisms and proper error handling  
- **Multi-Server Support**: Connect to multiple MCP servers simultaneously
- **Seamless Integration**: Tools appear as native functions to the LLM

### Usage

```python
# See examples/mcp_chat.py for a complete implementation
from chains.mcp_utils import Configuration, Server, create_tool_functions
from chains.msg_chains.oai_msg_chain_async import OpenAIAsyncMessageChain

# Initialize MCP servers
servers = [
    Server("minecraft-controller", {
        "command": "npx",
        "args": ["tsx", "path/to/minecraft-mcp-server.ts"]
    })
]

# Initialize and connect
for server in servers:
    await server.initialize()

# Create tool functions
tool_schemas, tool_mapping = await create_tool_functions(servers)

# Create chain with tools
chain = await (
    OpenAIAsyncMessageChain(model_name="gpt-4")
    .with_tools(tool_schemas, tool_mapping)
    .system("You are an AI assistant with access to external tools.")
)

# Use tools naturally in conversation
chain = await chain.user("Take a screenshot in Minecraft").generate_bot()
```

### Available Examples

- `examples/mcp_chat.py`: Interactive chat with MCP tool support
- `examples/hello.py`: Simple MCP server example

### Command Line Usage

```bash
# Run interactive chat with tools
python examples/mcp_chat.py --model "gpt-4" --msg "walk forward in minecraft"

# Use different models and endpoints
python examples/mcp_chat.py --model "google/gemini-flash-1.5" --base-url "https://openrouter.ai/api/v1"
```

## Caching

```python
# Cache system prompt or first message to reduce costs
chain = chain.system("Long prompt...", should_cache=True)
chain = chain.user("Complex instructions...", should_cache=True)
```

## Provider-Specific Features

- **Claude**: Ephemeral caching, anthropic.NOT_GIVEN support
- **Gemini**: File-based caching, role name adaptation
- **OpenAI**: Standard ChatGPT/GPT-4 interface

## Prompt Pipelines (chains.prompts)

The `chains.prompts` module provides a powerful framework for building complex multi-stage prompt workflows with:

- **PromptChain**: Immutable chain for building sequences of prompts with template rendering
- **Pipeline**: Decorator-based system for organizing multi-stage workflows
- **Conditional Execution**: Execute stages based on runtime conditions
- **Loop Support**: Repeat stages N times with per-iteration fields
- **Compiled Execution**: Optional graph-based compilation for optimized execution

### Quick Example

```python
from chains.prompts import PromptChain, Pipeline, register_prompt
from pydantic import BaseModel

# Define stages with decorators
pipeline = Pipeline()

@register_prompt("Generate a {{sector}} sector description")
@pipeline.register_stage("sector_desc")
class SectorDescription(BaseModel):
    description: str
    key_points: list[str]

# Execute the pipeline
chain = PromptChain()
result = (
    chain
    >> pipeline
    >> init(sector="technology")
    >> execute
)

print(result.sector_desc)
```

For more details on prompt pipelines, see the examples in the `chains/prompts/` directory.
