import argparse
import asyncio

import sys
from dataclasses import dataclass
from chains.msg_chains.oai_msg_chain_async import (
    OpenAIAsyncMessageChain as OpenAIMessageChain,
)
from chains.mcp_utils import (
    Configuration,
    Server,
    create_tool_functions,
)


@dataclass
class ChatSessionConfig:
    """Configuration for chat session."""

    servers: list["Server"]
    api_key: str
    model_name: str = "google/gemini-flash-1.5"
    base_url: str = "https://openrouter.ai/api/v1"
    initial_message: str | None = None
    constant_msg: str | None = None

async def cleanup_servers(servers: list[Server]) -> None:
    """Clean up all servers properly."""
    for server in reversed(servers):
        try:
            await server.cleanup()
        except Exception as e:
            print(f"Warning during final cleanup: {e}")


async def initialize_servers(servers: list[Server]) -> bool:
    for server in servers:
        try:
            await server.initialize()
        except Exception as e:
            print(f"Failed to initialize server: {e}")
            await cleanup_servers(servers)
            return False
    return True


async def handle_interactive_session(
    chain: OpenAIMessageChain, initial_message: str | None = None, constant_msg: str | None = None
) -> OpenAIMessageChain:
    # Send initial message if provided
    if initial_message:
        print(f"You: {initial_message}")
        chain = await chain.user(initial_message).generate_bot()
        print(f"Assistant: {chain.last_response}")

    print("Chat session started. Type 'quit' or 'exit' to end.")

    while True:
        try:
            if constant_msg is not None:
                user_input = constant_msg
            else:
                user_input = input("You: ").strip()
                if user_input.lower() in ["quit", "exit"]:
                    print("\nExiting...")
                    break

            if not user_input:
                continue

            # Use the new async-aware method
            chain = await chain.user(user_input).generate_bot()
            print(f"Assistant: {chain.last_response}")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error during interaction: {e}")
            continue

    return chain


async def run_chat_session(config: ChatSessionConfig) -> None:
    """Main chat session handler using functional paradigm.

    Args:
        config: Chat session configuration
    """
    try:
        # Initialize servers
        if not await initialize_servers(config.servers):
            return

        # Create tool functions and schemas
        tool_schemas, tool_mapping = await create_tool_functions(config.servers)

        # Initialize the chain
        chain = (
            OpenAIMessageChain(
                model_name=config.model_name,
                base_url=config.base_url,
                verbose=True,
            )
            .with_tools(tool_schemas, tool_mapping)
            
            .system(
"""You are a *very* ambitious minecraft player.
Your goal is to find and aquire dirt, wood, stone, iron and diamonds. All in your quest to kill the Ender dragon.
Follow Minecraft progression - wood first for tools, then stone, then dig deep for iron and diamonds.
You are autonomous and you can do anything you want.
I suggest making rotations of plus/minus 45 degrees at a time.
Craft wooden tools before trying to mine harder materials like stone or terracotta (remember that they take a while to mine).
Look for surface stone exposures, caves, or ravines rather than digging through hard blocks with bare hands
Don't call multiple tools at once.
"""
            )
        )

        # Handle interactive session with optional initial message
        chain = await handle_interactive_session(chain, config.initial_message, config.constant_msg)

    finally:
        await cleanup_servers(config.servers)


async def main() -> None:
    """Initialize and run the chat session."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="MCP Client with OpenAI Message Chain")
    parser.add_argument(
        "--model",
        # default="google/gemini-flash-1.5",
        default="gpt-4.1-nano",
        help="Model name to use (default: google/gemini-flash-1.5)",
    )
    parser.add_argument(
        "--base-url",
        # default="https://openrouter.ai/api/v1",
        default=None,
        help="API base URL (default: https://openrouter.ai/api/v1)",
    )
    parser.add_argument(
        "--msg",
        default=None,
        help="An optional first message to send to the assistant",
    )
    parser.add_argument(
        "--constant-msg",
        default=None,
        help="An optional constant message to send to the assistant",
    )

    args = parser.parse_args()

    config = Configuration()
    try:
        server_config = config.load_config("servers_config.json")
    except FileNotFoundError:
        server_config = {
            "mcpServers": {
                "echo": {"command": "python", "args": ["/Users/ohadr/chains/hello.py"]}
            }
        }
        server_config = {
            "mcpServers": {
                "minecraft-controller_stdio": {
                    "command": "npx",
                    "args": [
                        "tsx",
                        "/Users/ohadr/scrape_lm_copy/minecraft-web-client/minecraft-mcp-server.ts",
                        "--transport",
                        "stdio",
                    ],
                    "env": {
                    "NODE_NO_WARNINGS": "1"
                }
                },
            }
        }

    servers = [
        Server(name, srv_config)
        for name, srv_config in server_config["mcpServers"].items()
    ]

    chat_config = ChatSessionConfig(
        servers=servers,
        api_key=config.llm_api_key,
        model_name=args.model,
        base_url=args.base_url,
        initial_message=args.msg,
        constant_msg=args.constant_msg,
    )

    await run_chat_session(chat_config)


# python simple_client.py --msg "walk forwards in minecraft"
# python simple_client.py --model "google/gemma-3-12b" --base-url "http://localhost:1234/v1" --msg "walk forwards in minecraft"
# python simple_client.py --model "google/gemma-3-12b" --base-url "http://localhost:1234/v1" --msg "what's the weather in seattle?"
# python simple_client.py --model "google/gemma-3-12b" --base-url "http://localhost:1234/v1" --msg "what's the weather in tel aviv?"
if __name__ == "__main__":
    asyncio.run(main())
