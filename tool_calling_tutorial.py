"""

# Tool Calling Tutorial

This is a complete implementation of the OpenRouter tool calling tutorial that demonstrates how to use function calling with LLMs to search the Project Gutenberg library.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up your OpenRouter API key:**
   Since you already have the `OPENROUTER_API_KEY` environment variable set, you're ready to go! If you need to verify it's set, you can check with:
   ```bash
   echo $OPENROUTER_API_KEY
   ```

## Running the Code

Simply run the Python script:

```bash
python tool_calling_tutorial.py
```

## What the Script Does

The script demonstrates two different approaches to tool calling:

1. **Sequential Tool Calling Example**: Shows the step-by-step process of making API calls, handling tool requests, and getting the final response.

2. **Agentic Loop Example**: Demonstrates a more automated approach where the script continues making API calls and tool calls until the task is complete.

Both examples use the Project Gutenberg API to search for books based on author names or keywords.

## Expected Output

You should see output similar to:

```
=== Sequential Tool Calling Example ===
User query: What are the titles of some James Joyce books?
Making first API call...
Processing tool calls...
Calling tool: search_gutenberg_books with args: {'search_terms': ['James Joyce']}
Making second API call for final response...
Final response:
Here are some books by James Joyce:
* Ulysses
* Dubliners
* A Portrait of the Artist as a Young Man
...

=== Agentic Loop Example ===
...
```

## Troubleshooting

If you encounter any issues:

1. Make sure your `OPENROUTER_API_KEY` environment variable is set
2. Ensure you have an active internet connection
3. Check that all dependencies are installed correctly
4. Verify your OpenRouter API key has sufficient credits
"""

import json
import os
import requests
from openai import OpenAI

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable is not set")

MODEL = (
    "google/gemini-2.0-flash-001"  # You can use any model that supports tool calling
)

# Initialize OpenAI client with OpenRouter
openai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)


def search_gutenberg_books(search_terms):
    """Search for books in the Project Gutenberg library based on specified search terms"""
    search_query = " ".join(search_terms)
    url = "https://gutendex.com/books"
    response = requests.get(url, params={"search": search_query})

    simplified_results = []
    for book in response.json().get("results", []):
        simplified_results.append(
            {
                "id": book.get("id"),
                "title": book.get("title"),
                "authors": book.get("authors"),
            }
        )

    return simplified_results


# Tool definition for the LLM
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_gutenberg_books",
            "description": "Search for books in the Project Gutenberg library based on specified search terms",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_terms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of search terms to find books in the Gutenberg library (e.g. ['dickens', 'great'] to search for books by Dickens with 'great' in the title)",
                    }
                },
                "required": ["search_terms"],
            },
        },
    }
]

# Tool mapping for local function calls
TOOL_MAPPING = {"search_gutenberg_books": search_gutenberg_books}


def sequential_tool_calling_example():
    """Example of sequential tool calling as shown in the tutorial"""
    print("=== Sequential Tool Calling Example ===")

    task = "What are the titles of some James Joyce books?"

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": task,
        },
    ]

    # First API call
    print(f"User query: {task}")
    print("Making first API call...")

    request_1 = {"model": MODEL, "tools": tools, "messages": messages}

    response_1 = openai_client.chat.completions.create(**request_1)
    response_1_message = response_1.choices[0].message

    # Append the response to the messages array so the LLM has the full context
    messages.append(response_1_message.dict())

    # Process the requested tool calls
    print("Processing tool calls...")
    for tool_call in response_1_message.tool_calls:
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)
        print(f"Calling tool: {tool_name} with args: {tool_args}")

        tool_response = TOOL_MAPPING[tool_name](**tool_args)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": json.dumps(tool_response),
            }
        )

    # Second API call to get the final result
    print("Making second API call for final response...")
    request_2 = {"model": MODEL, "messages": messages, "tools": tools}

    response_2 = openai_client.chat.completions.create(**request_2)

    print("Final response:")
    print(response_2.choices[0].message.content)
    print()


def agentic_loop_example():
    """Example of an agentic loop as shown in the tutorial"""
    print("=== Agentic Loop Example ===")

    task = "Find me some books by Charles Dickens and tell me about them"

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": task,
        },
    ]

    def call_llm(msgs):
        resp = openai_client.chat.completions.create(
            model=MODEL, tools=tools, messages=msgs
        )
        msgs.append(resp.choices[0].message.dict())
        return resp

    def get_tool_response(response):
        tool_call = response.choices[0].message.tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)

        print(f"Calling tool: {tool_name} with args: {tool_args}")

        # Look up the correct tool locally, and call it with the provided arguments
        tool_result = TOOL_MAPPING[tool_name](**tool_args)

        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": tool_name,
            "content": json.dumps(tool_result),
        }

    print(f"User query: {task}")
    print("Starting agentic loop...")

    max_iterations = 5  # Prevent infinite loops
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        print(f"Iteration {iteration}:")

        resp = call_llm(messages)

        if resp.choices[0].message.tool_calls is not None:
            print("Tool call requested, processing...")
            messages.append(get_tool_response(resp))
        else:
            print("No more tool calls needed.")
            break

    print("Final response:")
    print(messages[-1]["content"])
    print()


def main():
    """Main function to run both examples"""
    try:
        # Run the sequential example first
        sequential_tool_calling_example()

        # Run the agentic loop example
        agentic_loop_example()

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to:")
        print("1. Install required packages: pip install openai requests")
        print("2. Set your OpenRouter API key in the OPENROUTER_API_KEY variable")
        print("3. Check your internet connection")


if __name__ == "__main__":
    main()
