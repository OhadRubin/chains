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

## Advanced: Single Chain Workflows

The most powerful pattern is using **one chain that flows through all operations** with shared variables:

```python
from chains.prompt_chain import PromptChain
from chains.msg_chain import MessageChain

def generate_attributes(chain):
    """Phase 1: Generate attributes using the single chain"""
    return (
        chain
        .prompt("Generate {n_attributes} quality attributes for a \"{target_goal}\"")
        .with_structure(AttributeList)
        .generate()
        .post_last(attributes_str=lambda x: x.att_into_str())  # ← Save for later use
    )

def create_stages(chain):
    """Phase 2: Use previous results in the same chain"""
    return (
        chain
        .prompt(
            "Create {n_stages} development stages for \"{target_goal}\" using:\n"
            "{attributes_str}"  # ← Automatically available from Phase 1
        )
        .with_structure(DevModel)
        .generate()
    )

# Single chain flows through all operations
final_result = (
    PromptChain()
    .set_prev_fields({
        "target_goal": "blog post about AI safety",
        "n_attributes": "8",
        "n_stages": "5"
    })
    .set_model(lambda: MessageChain.get_chain(model="gpt-4o"))
    .pipe(generate_attributes)
    .pipe(create_stages)
)

# Access any response from the chain
attributes = final_result.response_list[0]  # First operation result
stages = final_result.response_list[1]      # Second operation result
```

## Single Chain Benefits

### 🔗 **Shared Variables**
```python
# Variables set once are available everywhere
chain.set_prev_fields({"target_goal": "blog post", "n_items": "5"})
# Now use {target_goal} and {n_items} in any .prompt() call
```

### 📦 **Automatic State Capture**
```python
# .post_last() saves structured results for later operations
.post_last(summary=lambda x: x.summarize())
# Now {summary} is available in subsequent prompts
```

### 🌊 **Linear Flow**
```python
# Clean pipeline of operations
chain.pipe(step1).pipe(step2).pipe(step3)
# Each step can use results from all previous steps
```

### 📊 **Complete Traceability**
```python
# Access all intermediate results
final_chain.response_list  # List of all generated responses
final_chain.prev_fields    # All shared variables
```

## Framework Patterns

The MessageChain framework enables powerful patterns for complex AI workflows:

### 1. **Single Chain State Management**
One chain maintains all context instead of manually passing data between separate chains.

### 2. **Structured State Persistence**
Use `.post_last()` to extract and store structured data for use in later operations.

### 3. **Variable Interpolation**
Set variables once with `.set_prev_fields()`, use anywhere with `{variable_name}`.

### 4. **Pipelined Operations**
Chain operations with `.pipe(function)` where each function transforms the chain.

See `run_simple.py` for a complete example showcasing these patterns.

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