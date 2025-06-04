#!/usr/bin/env python3

import os
import sys
import json
import requests
from chains.msg_chains.oai_msg_chain import OpenAIMessageChain

# --- Tool Definition: search_gutenberg_books ---
def search_gutenberg_books(search_terms: list):
    """Search for books in the Project Gutenberg library based on specified search terms"""
    print(f"[Tool Call: search_gutenberg_books, Args: {search_terms}]")
    search_query = " ".join(search_terms)
    url = "https://gutendex.com/books"
    try:
        response = requests.get(url, params={"search": search_query})
        response.raise_for_status() # Raise an exception for HTTP errors
        results = response.json().get("results", [])
        simplified_results = []
        for book in results:
            simplified_results.append(
                {
                    "id": book.get("id"),
                    "title": book.get("title"),
                    "authors": [author.get("name") for author in book.get("authors", [])],
                }
            )
        output = simplified_results[:3] # Return only first 3 results
        print(f"[Tool Response: {json.dumps(output)}]")
        return output
    except requests.exceptions.RequestException as e:
        error_message = f"Error calling Gutenberg API: {e}"
        print(f"[Tool Error: {error_message}]")
        return {"error": error_message}
    except json.JSONDecodeError as e:
        error_message = f"Error decoding JSON from Gutenberg API: {e}"
        print(f"[Tool Error: {error_message}]")
        return {"error": error_message}


# Tool schema for OpenAI API
OPENAI_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_gutenberg_books",
            "description": "Search for books in the Project Gutenberg library based on specified search terms (e.g., author, title keywords).",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_terms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of search terms (e.g. ['dickens', 'great expectations'] or ['james joyce']).",
                    }
                },
                "required": ["search_terms"],
            },
        },
    }
]

# Tool mapping for local function calls
TOOLS_MAPPING = {"search_gutenberg_books": search_gutenberg_books}

def main():
    """Interactive ChatGPT CLI with tool calling using the chains library"""

    api_key_name = "OPENAI_API_KEY"
    model_name = "gpt-4o" # Default to OpenAI

    # --- Get API Endpoint (optional) ---
    api_endpoint = None
    if len(sys.argv) > 1:
        api_endpoint = sys.argv[1]
        print(f"Using API endpoint: {api_endpoint}")
        if "openrouter.ai" in api_endpoint:
            api_key_name = "OPENROUTER_API_KEY"
            # Suggest a model known to work well with tools on OpenRouter
            # You might want to make this configurable or use a default from oai_msg_chain
            model_name = "google/gemini-flash-1.5" 
            print(f"Using OpenRouter. Ensure {api_key_name} is set. Model set to {model_name}.")
        else:
            print(f"Ensure {api_key_name} is set for custom endpoint.")


    # --- Get API Key ---
    api_key = os.getenv(api_key_name)
    if not api_key:
        print(f"Error: {api_key_name} environment variable not set.", file=sys.stderr)
        return 1
    
    # --- Create the Chain ---
    try:
        if api_endpoint:
            chain = OpenAIMessageChain(model_name=model_name, base_url=api_endpoint)
        else:
            chain = OpenAIMessageChain(model_name=model_name) # Uses default OpenAI endpoint

        chain = (
            chain
            .with_tools(OPENAI_TOOLS_SCHEMA, TOOLS_MAPPING)
            .system(
                "You are a helpful assistant that can search for books using the 'search_gutenberg_books' tool. "
                "When asked to find books, use the tool. Provide concise answers based on the tool's output."
            )
        )

    except Exception as e:
        print(f"Chain Initialization Error: {e}", file=sys.stderr)
        return 1

    print(f"Interactive Tool-Calling Chat CLI (Model: {chain.model_name})")
    print("Type 'exit' or 'quit' to end. Tool calls will be shown in [brackets].")
    print("--------------------------------------------------------------------")

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            if not user_input:
                continue

            # The generate() method now handles the agentic tool calling loop
            chain = chain.user(user_input).generate()

            if chain.last_response:
                print(f"Bot: {chain.last_response}")
            else:
                # This case should be less common now that generate handles the loop
                # but good to have as a fallback.
                # Check if there were tool calls but no final text content.
                last_msg_obj = chain.messages[-1] if chain.messages else None
                if last_msg_obj and last_msg_obj.role == "assistant" and last_msg_obj.tool_calls:
                     print("Bot: (Made tool calls, awaiting next step or final response)")
                elif last_msg_obj and last_msg_obj.role == "tool":
                    print(f"Bot: (Processed tool response: {last_msg_obj.content})")
                else:
                    print("Bot: No textual response generated, but processing might have occurred.")


        except KeyboardInterrupt:
            print("Goodbye!")
            break
        except Exception as e:
            print(f"Error during interaction: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            continue
    return 0

if __name__ == "__main__":
    sys.exit(main()) 