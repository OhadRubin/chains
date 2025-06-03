#!/usr/bin/env python3

import os
import sys
from chains.msg_chains.oai_msg_chain import OpenAIMessageChain


def main():
    """Interactive ChatGPT CLI using the chains library"""

    # --- Get API Key ---
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.", file=sys.stderr)
        return 1

    # --- Get API Endpoint (optional) ---
    api_endpoint = None
    if len(sys.argv) > 1:
        api_endpoint = sys.argv[1]
        print(f"Using API endpoint: {api_endpoint}")

    # --- Create the Chain ---
    try:
        if api_endpoint:
            chain = OpenAIMessageChain(model_name="gpt-4o", base_url=api_endpoint)
        else:
            chain = OpenAIMessageChain(model_name="gpt-4o")

        # Set system prompt
        chain = chain.system(
            "You are a helpful assistant that provides concise answers."
        )

    except Exception as e:
        print(f"Chain Error: {e}", file=sys.stderr)
        return 1

    print("Interactive ChatGPT CLI (Type 'exit' or 'quit' to end)")
    print("------------------------------------------------------")

    while True:
        try:
            # Prompt for user input
            user_input = input("\nYou: ").strip()

            # Check for exit command
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            if not user_input:
                continue

            # Send message to the API
            chain = chain.user(user_input).generate_bot()

            # Display response
            if chain.last_response:
                print(f"Bot: {chain.last_response}")
            else:
                print("Bot: No response generated.")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            continue

    return 0


if __name__ == "__main__":
    sys.exit(main())
