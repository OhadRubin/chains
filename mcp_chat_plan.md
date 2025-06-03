<!-- read tool_chat.py mcp_support.md and oai_msg_chain.py to understand this. -->

# MCP Chat Adaptation Plan

## Overview
This document outlines the plan to adapt `tool_chat.py` into an MCP (Model Context Protocol) client while preserving the use of the OpenAIMessageChain API from the chains library.
The primary goal is to achieve a **Minimum Viable Product (MVP - Barebones)** that connects to a single MCP server and utilizes **one predefined tool** from it.

## Current State Analysis

### tool_chat.py Current Architecture
- Uses `OpenAIMessageChain` from `chains.msg_chains.oai_msg_chain`
- Has a built-in tool: `search_gutenberg_books`
- Supports custom API endpoints (OpenRouter, etc.)
- Uses OpenAI tool calling format
- Interactive CLI chat interface

### Key Components to Preserve (for MVP - Barebones)
- OpenAIMessageChain API usage
- Support for custom API endpoints (OpenRouter, etc.)
- Interactive chat loop
- Tool calling integration (for one predefined MCP server tool)

## Adaptation Plan (MVP - Barebones Focus)

### Phase 1: Core MCP Infrastructure for a Single Server & Single Predefined Tool

#### 1.1 Add MCP Dependencies (Same as before)
```python
# New imports to add
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack
import asyncio # For async/sync bridge
import shutil # For finding npx path if needed
```

#### 1.2 Create `MCPClientManager` (Barebones: Manages one server, executes one known tool)
- **Responsibility**: Start/stop a single specified MCP server process and provide an async method to call a *specific, predefined tool* on that server.
- **No dynamic tool discovery from server for this MVP.**

```python
# Detailed MCPClientManager (Barebones MVP)
class MCPClientManager:
    def __init__(self, server_script_path: str, predefined_tool_name: str):
        self.server_script_path = server_script_path
        self.predefined_tool_name = predefined_tool_name
        self.exit_stack = AsyncExitStack()
        self.session: ClientSession | None = None
        self._server_process = None # To hold the subprocess object

    async def start_server(self):
        # Logic to determine command (python vs node)
        # Based on self.server_script_path extension
        command = []
        if self.server_script_path.endswith('.py'):
            command = [sys.executable, self.server_script_path]
        elif self.server_script_path.endswith('.js'):
            # Consider npx for local packages if server_script_path is not a direct executable
            # For simplicity, assume direct node execution first or a globally available script
            node_executable = shutil.which('node')
            if not node_executable:
                raise RuntimeError("Node.js executable not found.")
            command = [node_executable, self.server_script_path]
        else:
            raise ValueError("Unsupported server script type. Use .py or .js")

        server_params = StdioServerParameters(
            command=command[0],
            args=command[1:], # Pass script as an argument to python/node
            env=os.environ.copy() # Pass current environment
        )
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        read, write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()
        print(f"MCP Server connected for tool: {self.predefined_tool_name}")

    async def execute_predefined_tool(self, arguments: dict) -> Any:
        if not self.session:
            raise RuntimeError("MCP Server not started or session not initialized.")
        print(f"[MCPClientManager] Calling tool: {self.predefined_tool_name} with args: {arguments}")
        # Actual MCP tool call
        result = await self.session.call_tool(self.predefined_tool_name, arguments)
        print(f"[MCPClientManager] Result for {self.predefined_tool_name}: {result.content}")
        return result.content # Or the whole result object if needed

    async def stop_server(self):
        await self.exit_stack.aclose()
        if self._server_process and self._server_process.poll() is None:
            self._server_process.terminate()
            await self._server_process.wait()
        print("MCP Server stopped.")

# Global instance for simplicity in MVP
# In a larger app, this would be managed differently.
mcp_manager: MCPClientManager | None = None
```

#### 1.3 Server Configuration (Simplified)
- A single command-line argument: `--server-script path/to/server.py` (or `.js`).
- A hardcoded name for the one tool we expect that server to provide (e.g., `SOME_PREDEFINED_TOOL_NAME`).

### Phase 2: Hardcoded Tool Definition & Execution Bridge (No Dynamic Discovery)

#### 2.1 Define the Single Tool's Schema Manually
- Instead of dynamic discovery, we hardcode the `OPENAI_TOOLS_SCHEMA` for the *one specific tool* we intend to use from the MCP server.

```python
# Example: Hardcoded schema for a hypothetical MCP tool
PREDEFINED_MCP_TOOL_NAME = "mcp_example_tool" # Must match the tool name on the MCP server
OPENAI_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": PREDEFINED_MCP_TOOL_NAME, # This name is used by the LLM
            "description": "A predefined tool accessible via an MCP server. Describe its function here.",
            "parameters": {
                "type": "object",
                "properties": {
                    "param1": {"type": "string", "description": "First parameter"},
                    "param2": {"type": "integer", "description": "Second parameter"}
                },
                "required": ["param1"]
            }
        }
    }
]
```

#### 2.2 Tool Execution Bridge (Async/Sync Wrapper)
- The `TOOLS_MAPPING` will map the `PREDEFINED_MCP_TOOL_NAME` to a synchronous wrapper function.
- This wrapper will use `asyncio.run()` to call `mcp_manager.execute_predefined_tool()`.

```python
# Synchronous wrapper for the async MCP tool call
def execute_mcp_tool_sync_wrapper(**kwargs):
    if not mcp_manager:
        return {"error": "MCP Manager not initialized."}
    try:
        # This is a blocking call, suitable if OpenAIMessageChain is sync
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we are already in an event loop (e.g. Jupyter notebook)
            # This part needs careful handling depending on the environment.
            # For simple CLI, asyncio.run might be okay, but nested loops are problematic.
            # A more robust solution might involve a dedicated thread for the asyncio loop.
            # For MVP CLI, we might assume direct asyncio.run is fine or handle simply.
            # Let's try a simple approach first for CLI.
            # If this wrapper is called from an already running async context, this will fail.
            # However, OpenAIMessageChain.generate() is typically synchronous.
            future = asyncio.run_coroutine_threadsafe(mcp_manager.execute_predefined_tool(kwargs), loop)
            return future.result() # This blocks until the coroutine completes
        else:
            return asyncio.run(mcp_manager.execute_predefined_tool(kwargs))
    except Exception as e:
        print(f"Error in execute_mcp_tool_sync_wrapper: {e}")
        return {"error": str(e)}

TOOLS_MAPPING = {
    PREDEFINED_MCP_TOOL_NAME: execute_mcp_tool_sync_wrapper
}
```

### Phase 3: OpenAIMessageChain Integration (with the single hardcoded tool)

#### 3.1 Preserve Chain API (Same as before)

#### 3.2 Static Tool Registration (Single Hardcoded Tool)
- The `chain.with_tools()` will use the hardcoded `OPENAI_TOOLS_SCHEMA` and `TOOLS_MAPPING`.

```python
# Chain setup
chain = OpenAIMessageChain(model_name=model_name, base_url=api_endpoint)
chain = (
    chain
    .with_tools(OPENAI_TOOLS_SCHEMA, TOOLS_MAPPING) # Using predefined schema and mapping
    .system(
        "You are a helpful assistant. You have access to one specific tool: '" + PREDEFINED_MCP_TOOL_NAME + "'. "
        "Use it when appropriate. Its description and parameters are provided."
    )
)
```

#### 3.3 Async/Sync Bridge (Handled by `execute_mcp_tool_sync_wrapper`)

### Phase 4: CLI Interface Updates (Minimal)

#### 4.1 Simplified Argument Parsing
- Add `argparse` for handling `--server-script`.

```python
# In main()
import argparse
parser = argparse.ArgumentParser(description="MCP Tool Chat Client (Barebones MVP)")
parser.add_argument("api_endpoint", nargs='?', help="Optional API endpoint URL")
parser.add_argument("--server-script", required=True, help="Path to the MCP server script (.py or .js)")
# Potentially add --predefined-tool-name if we don't want to hardcode it in the script
args = parser.parse_args()

api_endpoint = args.api_endpoint
server_script_path = args.server_script
```

#### 4.2 Server Management (Single Server Lifecycle in `main`)
- Initialize `MCPClientManager` and start its server *before* the chat loop.
- Stop the server *after* the chat loop (e.g., in a `finally` block).

```python
# In main() - (this will be an async main now)
async def main_async():
    global mcp_manager # To modify the global instance
    # ... (arg parsing, API key setup) ...

    # Initialize and start MCP Manager
    # PREDEFINED_MCP_TOOL_NAME is globally defined or passed as arg
    mcp_manager = MCPClientManager(server_script_path, PREDEFINED_MCP_TOOL_NAME)
    try:
        await mcp_manager.start_server()

        # --- Create the Chain (as before, using global mcp_manager via wrapper) ---
        # ... chain setup ...

        print(f"Interactive Tool-Calling Chat CLI (Model: {chain.model_name})")
        # ... (chat loop, but it's synchronous) ...
        # The chat loop itself remains synchronous. The tool call wrapper handles async.
        while True:
            try:
                user_input = input("You: ").strip()
                if user_input.lower() in ["exit", "quit"]:
                    print("Goodbye!")
                    break
                if not user_input:
                    continue
                
                chain = chain.user(user_input).generate() # This is a synchronous call

                if chain.last_response:
                    print(f"Bot: {chain.last_response}")
                # ... (rest of the response handling) ...
            except KeyboardInterrupt:
                print("Goodbye!")
                break
            except Exception as e:
                print(f"Error during interaction: {e}", file=sys.stderr)
                # ... traceback ...
                continue

    except Exception as e:
        print(f"Error during MCP Manager setup or chat: {e}", file=sys.stderr)
    finally:
        if mcp_manager:
            await mcp_manager.stop_server()

def main():
    # ... (sync setup like API key, initial arg parsing if any part must be sync) ...
    # Run the async main function
    # This requires python 3.7+
    args = parser.parse_args() # Simplified, assuming parser is defined globally or passed
    # ... (rest of the setup for server_script_path, api_key_name, etc.) ...
    
    # Setup for main_async call
    global api_endpoint, model_name, server_script_path, api_key_name # Ensure these are accessible
    # ... initialize these variables from args / env ...
    api_key = os.getenv(api_key_name)
    if not api_key:
        print(f"Error: {api_key_name} environment variable not set.", file=sys.stderr)
        return 1

    asyncio.run(main_async())

if __name__ == "__main__":
    # Adjust sys.exit call if main() returns a status code
    sys.exit(main() or 0)
```

#### 4.3 Error Handling (Basic)
- Basic `try-except` around server start and tool calls.

## Implementation Details (MVP - Barebones)

### File Structure Changes
- Create `mcp_tool_chat_barebones.py` (or similar name).

### Key Classes & Functions (MVP - Barebones)

#### `MCPClientManager` (Barebones - see above)
- `__init__(self, server_script_path: str, predefined_tool_name: str)`
- `async def start_server(self)`
- `async def execute_predefined_tool(self, arguments: dict) -> Any`
- `async def stop_server(self)`

#### `execute_mcp_tool_sync_wrapper(**kwargs)` (see above)
- Synchronous wrapper calling the async `execute_predefined_tool`.

#### Modified `main()` and new `main_async()`
- `main()`: Handles initial synchronous setup and then calls `asyncio.run(main_async())`.
- `main_async()`: Handles async operations like starting/stopping the MCP server and contains the chat loop.

### Integration Points

1.  **Hardcoded Tool Schema**: `OPENAI_TOOLS_SCHEMA` is defined statically for the one tool.
2.  **Hardcoded Tool Mapping**: `TOOLS_MAPPING` directly maps the tool name to `execute_mcp_tool_sync_wrapper`.
3.  **MCP Manager Initialization**: `MCPClientManager` is instantiated in `main_async()` with the server script path and the predefined tool name.
4.  **Server Lifecycle**: `mcp_manager.start_server()` called before chat loop, `mcp_manager.stop_server()` in `finally` block.

## Migration Strategy (MVP - Barebones First)

- Create `mcp_tool_chat_barebones.py` from `tool_chat.py`.
- Implement the `MCPClientManager` (barebones version).
- Hardcode `OPENAI_TOOLS_SCHEMA` and `TOOLS_MAPPING` for one tool.
- Implement the `execute_mcp_tool_sync_wrapper`.
- Modify `main()` to be async- शीर्षस्थ (top-level async) or to use `asyncio.run()` for an async main part, managing the server lifecycle.
- Test thoroughly with one MCP server that provides the predefined tool.

## Testing Plan (MVP - Barebones Focus)
- Test with a single, simple MCP server (e.g., a Python server with one basic tool).
- Verify the `OpenAIMessageChain` successfully calls the MCP tool via the wrapper.
- Check basic error handling for server start and tool execution.

## Benefits of This Approach (MVP - Barebones)

1.  **Absolute Fastest Path**: Gets a single MCP tool working with minimal new code.
2.  **Focus on Core Bridge**: Solves the critical async/sync interaction for one tool.
3.  **Clear Validation**: Easy to verify if the fundamental MCP integration works.
4.  **Highly Incremental**: Future features (dynamic discovery, multi-tool, multi-server) can be added on this proven base.

## Potential Challenges (Relevant for MVP - Barebones)

1.  **Async/Sync Bridge (`asyncio.run` behavior)**: Ensuring the `asyncio.run()` or `asyncio.run_coroutine_threadsafe` approach in the wrapper works correctly without causing nested loop issues, especially if `tool_chat.py` might be run in environments that already have an event loop (like Jupyter). For a CLI, it's usually simpler.
2.  **Server Script Execution**: Ensuring `StdioServerParameters` correctly launches the Python/Node.js server script across different environments (paths, permissions, python vs python3, node availability).
3.  **Hardcoding Brittleness**: The hardcoded tool name and schema must exactly match what the MCP server expects. Any mismatch will lead to failures.

## Success Criteria (MVP - Barebones)

- [ ] Successfully connects to and starts a single specified MCP server script.
- [ ] `OpenAIMessageChain` can trigger a call to the **one predefined MCP tool**.
- [ ] The MCP tool executes on the server, and its result is returned to the `OpenAIMessageChain` flow.
- [ ] The LLM receives the tool's result and can formulate a response.
- [ ] Basic CLI interaction for starting, chatting, and exiting works.
- [ ] Server process is cleaned up on exit. 