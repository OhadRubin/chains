# Message Chain Framework

A unified, chainable interface for working with multiple LLM providers (Anthropic Claude, Google Gemini, and OpenAI).

## Install

```bash
pip install anthropic "google-generativeai>=0.3.0" openai tenacity appdirs
```

## Quick Start

```python
from src.chain import MessageChain

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