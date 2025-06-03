import argparse
import asyncio
import json
import logging
import os
import shutil
import sys
from contextlib import AsyncExitStack
from typing import Any

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from chains.msg_chains.oai_msg_chain_async import (
    OpenAIAsyncMessageChain as OpenAIMessageChain,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - line %(lineno)d - %(message)s",
)


class Configuration:
    """Manages configuration and environment variables for the MCP client."""

    def __init__(self) -> None:
        """Initialize configuration with environment variables."""
        self.load_env()
        self.api_key = os.getenv("OPENROUTER_API_KEY")

    @staticmethod
    def load_env() -> None:
        """Load environment variables from .env file."""
        load_dotenv()

    @staticmethod
    def load_config(file_path: str) -> dict[str, Any]:
        """Load server configuration from JSON file.

        Args:
            file_path: Path to the JSON configuration file.

        Returns:
            Dict containing server configuration.

        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            JSONDecodeError: If configuration file is invalid JSON.
        """
        with open(file_path, "r") as f:
            return json.load(f)

    @property
    def llm_api_key(self) -> str:
        """Get the LLM API key.

        Returns:
            The API key as a string.

        Raises:
            ValueError: If the API key is not found in environment variables.
        """
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")
        return self.api_key


class Server:
    """Manages MCP server connections and tool execution."""

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name: str = name
        self.config: dict[str, Any] = config
        self.stdio_context: Any | None = None
        self.session: ClientSession | None = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.exit_stack: AsyncExitStack = AsyncExitStack()

    async def initialize(self) -> None:
        """Initialize the server connection."""
        command = (
            shutil.which("npx")
            if self.config["command"] == "npx"
            else self.config["command"]
        )
        if command is None:
            raise ValueError("The command must be a valid string and cannot be None.")

        server_params = StdioServerParameters(
            command=command,
            args=self.config["args"],
            env=(
                {**os.environ, **self.config["env"]} if self.config.get("env") else None
            ),
        )
        try:
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.session = session
        except Exception as e:
            logging.error(f"Error initializing server {self.name}: {e}")
            await self.cleanup()
            raise

    async def list_tools(self) -> list[Any]:
        """List available tools from the server.

        Returns:
            A list of available tools.

        Raises:
            RuntimeError: If the server is not initialized.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        tools_response = await self.session.list_tools()
        tools = []

        for item in tools_response:
            if isinstance(item, tuple) and item[0] == "tools":
                tools.extend(
                    Tool(tool.name, tool.description, tool.inputSchema)
                    for tool in item[1]
                )

        return tools

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        retries: int = 2,
        delay: float = 1.0,
    ) -> Any:
        """Execute a tool with retry mechanism.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.
            retries: Number of retry attempts.
            delay: Delay between retries in seconds.

        Returns:
            Tool execution result.

        Raises:
            RuntimeError: If server is not initialized.
            Exception: If tool execution fails after all retries.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        attempt = 0
        while attempt < retries:
            try:
                logging.info(f"Executing {tool_name}...")
                result = await self.session.call_tool(tool_name, arguments)

                return result

            except Exception as e:
                attempt += 1
                logging.warning(
                    f"Error executing tool: {e}. Attempt {attempt} of {retries}."
                )
                if attempt < retries:
                    logging.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logging.error("Max retries reached. Failing.")
                    raise

    async def cleanup(self) -> None:
        """Clean up server resources."""
        async with self._cleanup_lock:
            try:
                await self.exit_stack.aclose()
                self.session = None
                self.stdio_context = None
            except Exception as e:
                logging.error(f"Error during cleanup of server {self.name}: {e}")


class Tool:
    """Represents a tool with its properties and formatting."""

    def __init__(
        self, name: str, description: str, input_schema: dict[str, Any]
    ) -> None:
        self.name: str = name
        self.description: str = description
        self.input_schema: dict[str, Any] = input_schema

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert tool to OpenAI function schema format.

        Returns:
            A dictionary in OpenAI function schema format.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


class ChatSession:
    """Orchestrates the interaction between user, LLM, and tools using OpenAIMessageChain."""

    def __init__(
        self,
        servers: list[Server],
        api_key: str,
        model_name: str = "google/gemini-flash-1.5",
        base_url: str = "https://openrouter.ai/api/v1",
    ) -> None:
        self.servers: list[Server] = servers
        self.api_key: str = api_key
        self.model_name: str = model_name
        self.base_url: str = base_url
        self.chain: OpenAIMessageChain | None = None

    async def cleanup_servers(self) -> None:
        """Clean up all servers properly."""
        for server in reversed(self.servers):
            try:
                await server.cleanup()
            except Exception as e:
                logging.warning(f"Warning during final cleanup: {e}")

    async def create_tool_functions(self) -> tuple[list[dict], dict]:
        """Create tool functions and schemas for OpenAIMessageChain.

        Returns:
            Tuple of (tool_schemas, tool_mapping)
        """
        tool_schemas = []
        tool_mapping = {}

        for server in self.servers:
            tools = await server.list_tools()
            for tool in tools:
                # Create the schema
                tool_schemas.append(tool.to_openai_schema())

                # Create a proper async tool function that calls MCP directly
                def make_tool_function(srv, tool_name):
                    async def tool_function(**kwargs):
                        try:
                            result = await srv.execute_tool(tool_name, kwargs)

                            # Handle CallToolResult properly - preserve multimodal content
                            if hasattr(result, "content"):
                                if (
                                    isinstance(result.content, list)
                                    and len(result.content) > 0
                                ):
                                    # Return the full content list to preserve multimodal data
                                    return {
                                        "content": [
                                            {
                                                "type": getattr(item, "type", "text"),
                                                "text": getattr(item, "text", None),
                                                "data": getattr(item, "data", None),
                                                "mimeType": getattr(
                                                    item, "mimeType", None
                                                ),
                                            }
                                            for item in result.content
                                        ]
                                    }
                                else:
                                    return str(result.content)
                            else:
                                return str(result)
                        except Exception as e:
                            return f"Error executing tool {tool_name}: {str(e)}"

                    return tool_function

                tool_mapping[tool.name] = make_tool_function(server, tool.name)

        return tool_schemas, tool_mapping

    async def process_user_input(self, user_input: str) -> str:
        """Process user input using the async chain."""
        self.chain = await self.chain.user(user_input).generate_bot()

        if self.chain.last_response:
            return self.chain.last_response
        else:
            return "No response generated."

    async def start(self) -> None:
        """Main chat session handler using OpenAIMessageChain."""
        try:
            # Initialize servers
            for server in self.servers:
                try:
                    await server.initialize()
                except Exception as e:
                    logging.error(f"Failed to initialize server: {e}")
                    await self.cleanup_servers()
                    return

            # Create tool functions and schemas
            tool_schemas, tool_mapping = await self.create_tool_functions()

            # Initialize the chain
            self.chain = (
                OpenAIMessageChain(
                    model_name=self.model_name,
                    base_url=self.base_url,
                )
                .with_tools(tool_schemas, tool_mapping)
                .system(
                    "You are a helpful assistant with access to various tools. "
                    "Use the appropriate tool based on the user's question. "
                    "If no tool is needed, reply directly. "
                    "Provide natural, conversational responses based on tool results."
                    "When you use a tool, output only json with no prefixes."
                )
            )

            # Check if input is coming from a pipe or interactive terminal
            is_interactive = sys.stdin.isatty()

            if is_interactive:
                print("Chat session started. Type 'quit' or 'exit' to end.")

            if not is_interactive:
                # Handle piped input - read all input at once
                try:
                    user_input = sys.stdin.read().strip()
                    if user_input:
                        print(f"You: {user_input}")
                        response = await self.process_user_input(user_input)
                        print(f"Assistant: {response}")
                except Exception as e:
                    logging.error(f"Error processing piped input: {e}")
            else:
                # Interactive mode
                while True:
                    try:
                        user_input = input("You: ").strip()
                        if user_input.lower() in ["quit", "exit"]:
                            logging.info("\nExiting...")
                            break

                        if not user_input:
                            continue

                        # Use the new async-aware method
                        response = await self.process_user_input(user_input)
                        print(f"Assistant: {response}")

                    except KeyboardInterrupt:
                        logging.info("\nExiting...")
                        break
                    except Exception as e:
                        logging.error(f"Error during interaction: {e}")
                        continue

        finally:
            await self.cleanup_servers()


async def main() -> None:
    """Initialize and run the chat session."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="MCP Client with OpenAI Message Chain")
    parser.add_argument(
        "--model",
        default="google/gemini-flash-1.5",
        help="Model name to use (default: google/gemini-flash-1.5)",
    )
    parser.add_argument(
        "--base-url",
        default="https://openrouter.ai/api/v1",
        help="API base URL (default: https://openrouter.ai/api/v1)",
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
        server_config = { "mcpServers": {
        "minecraft-controller_stdio": {
            "command": "npx",
            "args": [
                "tsx",
                "/Users/ohadr/scrape_lm_copy/minecraft-web-client/minecraft-mcp-server.ts",
                "--transport",
                "stdio"
            ]
        },}}

    servers = [
        Server(name, srv_config)
        for name, srv_config in server_config["mcpServers"].items()
    ]

    chat_session = ChatSession(
        servers, config.llm_api_key, model_name=args.model, base_url=args.base_url
    )
    await chat_session.start()


# usage: echo "walk forwards in minecraft" | python simple_client.py
# usage: echo "walk forwards in minecraft" | python simple_client.py --model "google/gemma-3-12b" --base-url "http://localhost:1234/v1"
# usage: echo "what's the weather in seattle?" | python simple_client.py --model "google/gemma-3-12b" --base-url "http://localhost:1234/v1"
# usage: echo "what's the weather in tel aviv?" | python simple_client.py --model "google/gemma-3-12b" --base-url "http://localhost:1234/v1"
if __name__ == "__main__":
    asyncio.run(main())
