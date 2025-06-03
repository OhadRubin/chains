# For Client Developers

> Get started building your own client that can integrate with all MCP servers.

In this tutorial, you'll learn how to build a LLM-powered chatbot client that connects to MCP servers. It helps to have gone through the [Server quickstart](/quickstart/server) that guides you through the basic of building your first server.

<Tabs>
  <Tab title="Python">
    [You can find the complete code for this tutorial here.](https://github.com/modelcontextprotocol/quickstart-resources/tree/main/mcp-client-python)

    ## System Requirements

    Before starting, ensure your system meets these requirements:

    * Mac or Windows computer
    * Latest Python version installed
    * Latest version of `uv` installed

    ## Setting Up Your Environment

    First, create a new Python project with `uv`:

    ```bash
    # Create project directory
    uv init mcp-client
    cd mcp-client

    # Create virtual environment
    uv venv

    # Activate virtual environment
    # On Windows:
    .venv\Scripts\activate
    # On Unix or MacOS:
    source .venv/bin/activate

    # Install required packages
    uv add mcp anthropic python-dotenv

    # Remove boilerplate files
    # On Windows:
    del main.py
    # On Unix or MacOS:
    rm main.py

    # Create our main file
    touch client.py
    ```

    ## Setting Up Your API Key

    You'll need an Anthropic API key from the [Anthropic Console](https://console.anthropic.com/settings/keys).

    Create a `.env` file to store it:

    ```bash
    # Create .env file
    touch .env
    ```

    Add your key to the `.env` file:

    ```bash
    ANTHROPIC_API_KEY=<your key here>
    ```

    Add `.env` to your `.gitignore`:

    ```bash
    echo ".env" >> .gitignore
    ```

    <Warning>
      Make sure you keep your `ANTHROPIC_API_KEY` secure!
    </Warning>

    ## Creating the Client

    ### Basic Client Structure

    First, let's set up our imports and create the basic client class:

    ```python
    import asyncio
    from typing import Optional
    from contextlib import AsyncExitStack

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    from anthropic import Anthropic
    from dotenv import load_dotenv

    load_dotenv()  # load environment variables from .env

    class MCPClient:
        def __init__(self):
            # Initialize session and client objects
            self.session: Optional[ClientSession] = None
            self.exit_stack = AsyncExitStack()
            self.anthropic = Anthropic()
        # methods will go here
    ```

    ### Server Connection Management

    Next, we'll implement the method to connect to an MCP server:

    ```python
    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])
    ```

    ### Query Processing Logic

    Now let's add the core functionality for processing queries and handling tool calls:

    ```python
    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        response = await self.session.list_tools()
        available_tools = [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]

        # Initial Claude API call
        response = self.anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=messages,
            tools=available_tools
        )

        # Process response and handle tool calls
        final_text = []

        assistant_message_content = []
        for content in response.content:
            if content.type == 'text':
                final_text.append(content.text)
                assistant_message_content.append(content)
            elif content.type == 'tool_use':
                tool_name = content.name
                tool_args = content.input

                # Execute tool call
                result = await self.session.call_tool(tool_name, tool_args)
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

                assistant_message_content.append(content)
                messages.append({
                    "role": "assistant",
                    "content": assistant_message_content
                })
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": content.id,
                            "content": result.content
                        }
                    ]
                })

                # Get next response from Claude
                response = self.anthropic.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    messages=messages,
                    tools=available_tools
                )

                final_text.append(response.content[0].text)

        return "\n".join(final_text)
    ```

    ### Interactive Chat Interface

    Now we'll add the chat loop and cleanup functionality:

    ```python
    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()
    ```

    ### Main Entry Point

    Finally, we'll add the main execution logic:

    ```python
    async def main():
        if len(sys.argv) < 2:
            print("Usage: python client.py <path_to_server_script>")
            sys.exit(1)

        client = MCPClient()
        try:
            await client.connect_to_server(sys.argv[1])
            await client.chat_loop()
        finally:
            await client.cleanup()

    if __name__ == "__main__":
        import sys
        asyncio.run(main())
    ```

    You can find the complete `client.py` file [here.](https://gist.github.com/zckly/f3f28ea731e096e53b39b47bf0a2d4b1)

    ## Key Components Explained

    ### 1. Client Initialization

    * The `MCPClient` class initializes with session management and API clients
    * Uses `AsyncExitStack` for proper resource management
    * Configures the Anthropic client for Claude interactions

    ### 2. Server Connection

    * Supports both Python and Node.js servers
    * Validates server script type
    * Sets up proper communication channels
    * Initializes the session and lists available tools

    ### 3. Query Processing

    * Maintains conversation context
    * Handles Claude's responses and tool calls
    * Manages the message flow between Claude and tools
    * Combines results into a coherent response

    ### 4. Interactive Interface

    * Provides a simple command-line interface
    * Handles user input and displays responses
    * Includes basic error handling
    * Allows graceful exit

    ### 5. Resource Management

    * Proper cleanup of resources
    * Error handling for connection issues
    * Graceful shutdown procedures

    ## Common Customization Points

    1. **Tool Handling**
       * Modify `process_query()` to handle specific tool types
       * Add custom error handling for tool calls
       * Implement tool-specific response formatting

    2. **Response Processing**
       * Customize how tool results are formatted
       * Add response filtering or transformation
       * Implement custom logging

    3. **User Interface**
       * Add a GUI or web interface
       * Implement rich console output
       * Add command history or auto-completion

    ## Running the Client

    To run your client with any MCP server:

    ```bash
    uv run client.py path/to/server.py # python server
    uv run client.py path/to/build/index.js # node server
    ```

    <Note>
      If you're continuing the weather tutorial from the server quickstart, your command might look something like this: `python client.py .../quickstart-resources/weather-server-python/weather.py`
    </Note>

    The client will:

    1. Connect to the specified server
    2. List available tools
    3. Start an interactive chat session where you can:
       * Enter queries
       * See tool executions
       * Get responses from Claude

    Here's an example of what it should look like if connected to the weather server from the server quickstart:

    <Frame>
      <img src="https://mintlify.s3.us-west-1.amazonaws.com/mcp/images/client-claude-cli-python.png" />
    </Frame>

    ## How It Works

    When you submit a query:

    1. The client gets the list of available tools from the server
    2. Your query is sent to Claude along with tool descriptions
    3. Claude decides which tools (if any) to use
    4. The client executes any requested tool calls through the server
    5. Results are sent back to Claude
    6. Claude provides a natural language response
    7. The response is displayed to you

    ## Best practices

    1. **Error Handling**
       * Always wrap tool calls in try-catch blocks
       * Provide meaningful error messages
       * Gracefully handle connection issues

    2. **Resource Management**
       * Use `AsyncExitStack` for proper cleanup
       * Close connections when done
       * Handle server disconnections

    3. **Security**
       * Store API keys securely in `.env`
       * Validate server responses
       * Be cautious with tool permissions

    ## Troubleshooting

    ### Server Path Issues

    * Double-check the path to your server script is correct
    * Use the absolute path if the relative path isn't working
    * For Windows users, make sure to use forward slashes (/) or escaped backslashes (\\) in the path
    * Verify the server file has the correct extension (.py for Python or .js for Node.js)

    Example of correct path usage:

    ```bash
    # Relative path
    uv run client.py ./server/weather.py

    # Absolute path
    uv run client.py /Users/username/projects/mcp-server/weather.py

    # Windows path (either format works)
    uv run client.py C:/projects/mcp-server/weather.py
    uv run client.py C:\\projects\\mcp-server\\weather.py
    ```

    ### Response Timing

    * The first response might take up to 30 seconds to return
    * This is normal and happens while:
      * The server initializes
      * Claude processes the query
      * Tools are being executed
    * Subsequent responses are typically faster
    * Don't interrupt the process during this initial waiting period

    ### Common Error Messages

    If you see:

    * `FileNotFoundError`: Check your server path
    * `Connection refused`: Ensure the server is running and the path is correct
    * `Tool execution failed`: Verify the tool's required environment variables are set
    * `Timeout error`: Consider increasing the timeout in your client configuration
  </Tab>

  <Tab title="Node">
    [You can find the complete code for this tutorial here.](https://github.com/modelcontextprotocol/quickstart-resources/tree/main/mcp-client-typescript)

    ## System Requirements

    Before starting, ensure your system meets these requirements:

    * Mac or Windows computer
    * Node.js 17 or higher installed
    * Latest version of `npm` installed
    * Anthropic API key (Claude)

    ## Setting Up Your Environment

    First, let's create and set up our project:

    <CodeGroup>
      ```bash MacOS/Linux
      # Create project directory
      mkdir mcp-client-typescript
      cd mcp-client-typescript

      # Initialize npm project
      npm init -y

      # Install dependencies
      npm install @anthropic-ai/sdk @modelcontextprotocol/sdk dotenv

      # Install dev dependencies
      npm install -D @types/node typescript

      # Create source file
      touch index.ts
      ```

      ```powershell Windows
      # Create project directory
      md mcp-client-typescript
      cd mcp-client-typescript

      # Initialize npm project
      npm init -y

      # Install dependencies
      npm install @anthropic-ai/sdk @modelcontextprotocol/sdk dotenv

      # Install dev dependencies
      npm install -D @types/node typescript

      # Create source file
      new-item index.ts
      ```
    </CodeGroup>

    Update your `package.json` to set `type: "module"` and a build script:

    ```json package.json
    {
      "type": "module",
      "scripts": {
        "build": "tsc && chmod 755 build/index.js"
      }
    }
    ```

    Create a `tsconfig.json` in the root of your project:

    ```json tsconfig.json
    {
      "compilerOptions": {
        "target": "ES2022",
        "module": "Node16",
        "moduleResolution": "Node16",
        "outDir": "./build",
        "rootDir": "./",
        "strict": true,
        "esModuleInterop": true,
        "skipLibCheck": true,
        "forceConsistentCasingInFileNames": true
      },
      "include": ["index.ts"],
      "exclude": ["node_modules"]
    }
    ```

    ## Setting Up Your API Key

    You'll need an Anthropic API key from the [Anthropic Console](https://console.anthropic.com/settings/keys).

    Create a `.env` file to store it:

    ```bash
    echo "ANTHROPIC_API_KEY=<your key here>" > .env
    ```

    Add `.env` to your `.gitignore`:

    ```bash
    echo ".env" >> .gitignore
    ```

    <Warning>
      Make sure you keep your `ANTHROPIC_API_KEY` secure!
    </Warning>

    ## Creating the Client

    ### Basic Client Structure

    First, let's set up our imports and create the basic client class in `index.ts`:

    ```typescript
    import { Anthropic } from "@anthropic-ai/sdk";
    import {
      MessageParam,
      Tool,
    } from "@anthropic-ai/sdk/resources/messages/messages.mjs";
    import { Client } from "@modelcontextprotocol/sdk/client/index.js";
    import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
    import readline from "readline/promises";
    import dotenv from "dotenv";

    dotenv.config();

    const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY;
    if (!ANTHROPIC_API_KEY) {
      throw new Error("ANTHROPIC_API_KEY is not set");
    }

    class MCPClient {
      private mcp: Client;
      private anthropic: Anthropic;
      private transport: StdioClientTransport | null = null;
      private tools: Tool[] = [];

      constructor() {
        this.anthropic = new Anthropic({
          apiKey: ANTHROPIC_API_KEY,
        });
        this.mcp = new Client({ name: "mcp-client-cli", version: "1.0.0" });
      }
      // methods will go here
    }
    ```

    ### Server Connection Management

    Next, we'll implement the method to connect to an MCP server:

    ```typescript
    async connectToServer(serverScriptPath: string) {
      try {
        const isJs = serverScriptPath.endsWith(".js");
        const isPy = serverScriptPath.endsWith(".py");
        if (!isJs && !isPy) {
          throw new Error("Server script must be a .js or .py file");
        }
        const command = isPy
          ? process.platform === "win32"
            ? "python"
            : "python3"
          : process.execPath;

        this.transport = new StdioClientTransport({
          command,
          args: [serverScriptPath],
        });
        this.mcp.connect(this.transport);

        const toolsResult = await this.mcp.listTools();
        this.tools = toolsResult.tools.map((tool) => {
          return {
            name: tool.name,
            description: tool.description,
            input_schema: tool.inputSchema,
          };
        });
        console.log(
          "Connected to server with tools:",
          this.tools.map(({ name }) => name)
        );
      } catch (e) {
        console.log("Failed to connect to MCP server: ", e);
        throw e;
      }
    }
    ```

    ### Query Processing Logic

    Now let's add the core functionality for processing queries and handling tool calls:

    ```typescript
    async processQuery(query: string) {
      const messages: MessageParam[] = [
        {
          role: "user",
          content: query,
        },
      ];

      const response = await this.anthropic.messages.create({
        model: "claude-3-5-sonnet-20241022",
        max_tokens: 1000,
        messages,
        tools: this.tools,
      });

      const finalText = [];
      const toolResults = [];

      for (const content of response.content) {
        if (content.type === "text") {
          finalText.push(content.text);
        } else if (content.type === "tool_use") {
          const toolName = content.name;
          const toolArgs = content.input as { [x: string]: unknown } | undefined;

          const result = await this.mcp.callTool({
            name: toolName,
            arguments: toolArgs,
          });
          toolResults.push(result);
          finalText.push(
            `[Calling tool ${toolName} with args ${JSON.stringify(toolArgs)}]`
          );

          messages.push({
            role: "user",
            content: result.content as string,
          });

          const response = await this.anthropic.messages.create({
            model: "claude-3-5-sonnet-20241022",
            max_tokens: 1000,
            messages,
          });

          finalText.push(
            response.content[0].type === "text" ? response.content[0].text : ""
          );
        }
      }

      return finalText.join("\n");
    }
    ```

    ### Interactive Chat Interface

    Now we'll add the chat loop and cleanup functionality:

    ```typescript
    async chatLoop() {
      const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
      });

      try {
        console.log("\nMCP Client Started!");
        console.log("Type your queries or 'quit' to exit.");

        while (true) {
          const message = await rl.question("\nQuery: ");
          if (message.toLowerCase() === "quit") {
            break;
          }
          const response = await this.processQuery(message);
          console.log("\n" + response);
        }
      } finally {
        rl.close();
      }
    }

    async cleanup() {
      await this.mcp.close();
    }
    ```

    ### Main Entry Point

    Finally, we'll add the main execution logic:

    ```typescript
    async function main() {
      if (process.argv.length < 3) {
        console.log("Usage: node index.ts <path_to_server_script>");
        return;
      }
      const mcpClient = new MCPClient();
      try {
        await mcpClient.connectToServer(process.argv[2]);
        await mcpClient.chatLoop();
      } finally {
        await mcpClient.cleanup();
        process.exit(0);
      }
    }

    main();
    ```

    ## Running the Client

    To run your client with any MCP server:

    ```bash
    # Build TypeScript
    npm run build

    # Run the client
    node build/index.js path/to/server.py # python server
    node build/index.js path/to/build/index.js # node server
    ```

    <Note>
      If you're continuing the weather tutorial from the server quickstart, your command might look something like this: `node build/index.js .../quickstart-resources/weather-server-typescript/build/index.js`
    </Note>

    **The client will:**

    1. Connect to the specified server
    2. List available tools
    3. Start an interactive chat session where you can:
       * Enter queries
       * See tool executions
       * Get responses from Claude

    ## How It Works

    When you submit a query:

    1. The client gets the list of available tools from the server
    2. Your query is sent to Claude along with tool descriptions
    3. Claude decides which tools (if any) to use
    4. The client executes any requested tool calls through the server
    5. Results are sent back to Claude
    6. Claude provides a natural language response
    7. The response is displayed to you

    ## Best practices

    1. **Error Handling**
       * Use TypeScript's type system for better error detection
       * Wrap tool calls in try-catch blocks
       * Provide meaningful error messages
       * Gracefully handle connection issues

    2. **Security**
       * Store API keys securely in `.env`
       * Validate server responses
       * Be cautious with tool permissions

    ## Troubleshooting

    ### Server Path Issues

    * Double-check the path to your server script is correct
    * Use the absolute path if the relative path isn't working
    * For Windows users, make sure to use forward slashes (/) or escaped backslashes (\\) in the path
    * Verify the server file has the correct extension (.js for Node.js or .py for Python)

    Example of correct path usage:

    ```bash
    # Relative path
    node build/index.js ./server/build/index.js

    # Absolute path
    node build/index.js /Users/username/projects/mcp-server/build/index.js

    # Windows path (either format works)
    node build/index.js C:/projects/mcp-server/build/index.js
    node build/index.js C:\\projects\\mcp-server\\build\\index.js
    ```

    ### Response Timing

    * The first response might take up to 30 seconds to return
    * This is normal and happens while:
      * The server initializes
      * Claude processes the query
      * Tools are being executed
    * Subsequent responses are typically faster
    * Don't interrupt the process during this initial waiting period

    ### Common Error Messages

    If you see:

    * `Error: Cannot find module`: Check your build folder and ensure TypeScript compilation succeeded
    * `Connection refused`: Ensure the server is running and the path is correct
    * `Tool execution failed`: Verify the tool's required environment variables are set
    * `ANTHROPIC_API_KEY is not set`: Check your .env file and environment variables
    * `TypeError`: Ensure you're using the correct types for tool arguments
  </Tab>

  <Tab title="Java">
    <Note>
      This is a quickstart demo based on Spring AI MCP auto-configuration and boot starters.
      To learn how to create sync and async MCP Clients manually, consult the [Java SDK Client](/sdk/java/mcp-client) documentation
    </Note>

    This example demonstrates how to build an interactive chatbot that combines Spring AI's Model Context Protocol (MCP) with the [Brave Search MCP Server](https://github.com/modelcontextprotocol/servers-archived/tree/main/src/brave-search). The application creates a conversational interface powered by Anthropic's Claude AI model that can perform internet searches through Brave Search, enabling natural language interactions with real-time web data.
    [You can find the complete code for this tutorial here.](https://github.com/spring-projects/spring-ai-examples/tree/main/model-context-protocol/web-search/brave-chatbot)

    ## System Requirements

    Before starting, ensure your system meets these requirements:

    * Java 17 or higher
    * Maven 3.6+
    * npx package manager
    * Anthropic API key (Claude)
    * Brave Search API key

    ## Setting Up Your Environment

    1. Install npx (Node Package eXecute):
       First, make sure to install [npm](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm)
       and then run:
       ```bash
       npm install -g npx
       ```

    2. Clone the repository:
       ```bash
       git clone https://github.com/spring-projects/spring-ai-examples.git
       cd model-context-protocol/brave-chatbot
       ```

    3. Set up your API keys:
       ```bash
       export ANTHROPIC_API_KEY='your-anthropic-api-key-here'
       export BRAVE_API_KEY='your-brave-api-key-here'
       ```

    4. Build the application:
       ```bash
       ./mvnw clean install
       ```

    5. Run the application using Maven:
       ```bash
       ./mvnw spring-boot:run
       ```

    <Warning>
      Make sure you keep your `ANTHROPIC_API_KEY` and `BRAVE_API_KEY` keys secure!
    </Warning>

    ## How it Works

    The application integrates Spring AI with the Brave Search MCP server through several components:

    ### MCP Client Configuration

    1. Required dependencies in pom.xml:

    ```xml
    <dependency>
        <groupId>org.springframework.ai</groupId>
        <artifactId>spring-ai-starter-mcp-client</artifactId>
    </dependency>
    <dependency>
        <groupId>org.springframework.ai</groupId>
        <artifactId>spring-ai-starter-model-anthropic</artifactId>
    </dependency>
    ```

    2. Application properties (application.yml):

    ```yml
    spring:
      ai:
        mcp:
          client:
            enabled: true
            name: brave-search-client
            version: 1.0.0
            type: SYNC
            request-timeout: 20s
            stdio:
              root-change-notification: true
              servers-configuration: classpath:/mcp-servers-config.json
            toolcallback:
              enabled: true
        anthropic:
          api-key: ${ANTHROPIC_API_KEY}
    ```

    This activates the `spring-ai-starter-mcp-client` to create one or more `McpClient`s based on the provided server configuration.
    The `spring.ai.mcp.client.toolcallback.enabled=true` property enables the tool callback mechanism, that automatically registers all MCP tool as spring ai tools.
    It is disabled by default.

    3. MCP Server Configuration (`mcp-servers-config.json`):

    ```json
    {
      "mcpServers": {
        "brave-search": {
          "command": "npx",
          "args": [
            "-y",
            "@modelcontextprotocol/server-brave-search"
          ],
          "env": {
            "BRAVE_API_KEY": "<PUT YOUR BRAVE API KEY>"
          }
        }
      }
    }
    ```

    ### Chat Implementation

    The chatbot is implemented using Spring AI's ChatClient with MCP tool integration:

    ```java
    var chatClient = chatClientBuilder
        .defaultSystem("You are useful assistant, expert in AI and Java.")
        .defaultToolCallbacks((Object[]) mcpToolAdapter.toolCallbacks())
        .defaultAdvisors(new MessageChatMemoryAdvisor(new InMemoryChatMemory()))
        .build();
    ```

    <Warning>
      Breaking change: From SpringAI 1.0.0-M8 onwards, use `.defaultToolCallbacks(...)` instead of `.defaultTool(...)` to register MCP tools.
    </Warning>

    Key features:

    * Uses Claude AI model for natural language understanding
    * Integrates Brave Search through MCP for real-time web search capabilities
    * Maintains conversation memory using InMemoryChatMemory
    * Runs as an interactive command-line application

    ### Build and run

    ```bash
    ./mvnw clean install
    java -jar ./target/ai-mcp-brave-chatbot-0.0.1-SNAPSHOT.jar
    ```

    or

    ```bash
    ./mvnw spring-boot:run
    ```

    The application will start an interactive chat session where you can ask questions. The chatbot will use Brave Search when it needs to find information from the internet to answer your queries.

    The chatbot can:

    * Answer questions using its built-in knowledge
    * Perform web searches when needed using Brave Search
    * Remember context from previous messages in the conversation
    * Combine information from multiple sources to provide comprehensive answers

    ### Advanced Configuration

    The MCP client supports additional configuration options:

    * Client customization through `McpSyncClientCustomizer` or `McpAsyncClientCustomizer`
    * Multiple clients with multiple transport types: `STDIO` and `SSE` (Server-Sent Events)
    * Integration with Spring AI's tool execution framework
    * Automatic client initialization and lifecycle management

    For WebFlux-based applications, you can use the WebFlux starter instead:

    ```xml
    <dependency>
        <groupId>org.springframework.ai</groupId>
        <artifactId>spring-ai-mcp-client-webflux-spring-boot-starter</artifactId>
    </dependency>
    ```

    This provides similar functionality but uses a WebFlux-based SSE transport implementation, recommended for production deployments.
  </Tab>

  <Tab title="Kotlin">
    [You can find the complete code for this tutorial here.](https://github.com/modelcontextprotocol/kotlin-sdk/tree/main/samples/kotlin-mcp-client)

    ## System Requirements

    Before starting, ensure your system meets these requirements:

    * Java 17 or higher
    * Anthropic API key (Claude)

    ## Setting up your environment

    First, let's install `java` and `gradle` if you haven't already.
    You can download `java` from [official Oracle JDK website](https://www.oracle.com/java/technologies/downloads/).
    Verify your `java` installation:

    ```bash
    java --version
    ```

    Now, let's create and set up your project:

    <CodeGroup>
      ```bash MacOS/Linux
      # Create a new directory for our project
      mkdir kotlin-mcp-client
      cd kotlin-mcp-client

      # Initialize a new kotlin project
      gradle init
      ```

      ```powershell Windows
      # Create a new directory for our project
      md kotlin-mcp-client
      cd kotlin-mcp-client
      # Initialize a new kotlin project
      gradle init
      ```
    </CodeGroup>

    After running `gradle init`, you will be presented with options for creating your project.
    Select **Application** as the project type, **Kotlin** as the programming language, and **Java 17** as the Java version.

    Alternatively, you can create a Kotlin application using the [IntelliJ IDEA project wizard](https://kotlinlang.org/docs/jvm-get-started.html).

    After creating the project, add the following dependencies:

    <CodeGroup>
      ```kotlin build.gradle.kts
      val mcpVersion = "0.4.0"
      val slf4jVersion = "2.0.9"
      val anthropicVersion = "0.8.0"

      dependencies {
          implementation("io.modelcontextprotocol:kotlin-sdk:$mcpVersion")
          implementation("org.slf4j:slf4j-nop:$slf4jVersion")
          implementation("com.anthropic:anthropic-java:$anthropicVersion")
      }
      ```

      ```groovy build.gradle
      def mcpVersion = '0.3.0'
      def slf4jVersion = '2.0.9'
      def anthropicVersion = '0.8.0'
      dependencies {
          implementation "io.modelcontextprotocol:kotlin-sdk:$mcpVersion"
          implementation "org.slf4j:slf4j-nop:$slf4jVersion"
          implementation "com.anthropic:anthropic-java:$anthropicVersion"
      }
      ```
    </CodeGroup>

    Also, add the following plugins to your build script:

    <CodeGroup>
      ```kotlin build.gradle.kts
      plugins {
          id("com.github.johnrengelman.shadow") version "8.1.1"
      }
      ```

      ```groovy build.gradle
      plugins {
          id 'com.github.johnrengelman.shadow' version '8.1.1'
      }
      ```
    </CodeGroup>

    ## Setting up your API key

    You'll need an Anthropic API key from the [Anthropic Console](https://console.anthropic.com/settings/keys).

    Set up your API key:

    ```bash
    export ANTHROPIC_API_KEY='your-anthropic-api-key-here'
    ```

    <Warning>
      Make sure your keep your `ANTHROPIC_API_KEY` secure!
    </Warning>

    ## Creating the Client

    ### Basic Client Structure

    First, let's create the basic client class:

    ```kotlin
    class MCPClient : AutoCloseable {
        private val anthropic = AnthropicOkHttpClient.fromEnv()
        private val mcp: Client = Client(clientInfo = Implementation(name = "mcp-client-cli", version = "1.0.0"))
        private lateinit var tools: List<ToolUnion>

        // methods will go here

        override fun close() {
            runBlocking {
                mcp.close()
                anthropic.close()
            }
        }
    ```

    ### Server connection management

    Next, we'll implement the method to connect to an MCP server:

    ```kotlin
    suspend fun connectToServer(serverScriptPath: String) {
        try {
            val command = buildList {
                when (serverScriptPath.substringAfterLast(".")) {
                    "js" -> add("node")
                    "py" -> add(if (System.getProperty("os.name").lowercase().contains("win")) "python" else "python3")
                    "jar" -> addAll(listOf("java", "-jar"))
                    else -> throw IllegalArgumentException("Server script must be a .js, .py or .jar file")
                }
                add(serverScriptPath)
            }

            val process = ProcessBuilder(command).start()
            val transport = StdioClientTransport(
                input = process.inputStream.asSource().buffered(),
                output = process.outputStream.asSink().buffered()
            )

            mcp.connect(transport)

            val toolsResult = mcp.listTools()
            tools = toolsResult?.tools?.map { tool ->
                ToolUnion.ofTool(
                    Tool.builder()
                        .name(tool.name)
                        .description(tool.description ?: "")
                        .inputSchema(
                            Tool.InputSchema.builder()
                                .type(JsonValue.from(tool.inputSchema.type))
                                .properties(tool.inputSchema.properties.toJsonValue())
                                .putAdditionalProperty("required", JsonValue.from(tool.inputSchema.required))
                                .build()
                        )
                        .build()
                )
            } ?: emptyList()
            println("Connected to server with tools: ${tools.joinToString(", ") { it.tool().get().name() }}")
        } catch (e: Exception) {
            println("Failed to connect to MCP server: $e")
            throw e
        }
    }
    ```

    Also create a helper function to convert from `JsonObject` to `JsonValue` for Anthropic:

    ```kotlin
    private fun JsonObject.toJsonValue(): JsonValue {
        val mapper = ObjectMapper()
        val node = mapper.readTree(this.toString())
        return JsonValue.fromJsonNode(node)
    }
    ```

    ### Query processing logic

    Now let's add the core functionality for processing queries and handling tool calls:

    ```kotlin
    private val messageParamsBuilder: MessageCreateParams.Builder = MessageCreateParams.builder()
        .model(Model.CLAUDE_3_5_SONNET_20241022)
        .maxTokens(1024)

    suspend fun processQuery(query: String): String {
        val messages = mutableListOf(
            MessageParam.builder()
                .role(MessageParam.Role.USER)
                .content(query)
                .build()
        )

        val response = anthropic.messages().create(
            messageParamsBuilder
                .messages(messages)
                .tools(tools)
                .build()
        )

        val finalText = mutableListOf<String>()
        response.content().forEach { content ->
            when {
                content.isText() -> finalText.add(content.text().getOrNull()?.text() ?: "")

                content.isToolUse() -> {
                    val toolName = content.toolUse().get().name()
                    val toolArgs =
                        content.toolUse().get()._input().convert(object : TypeReference<Map<String, JsonValue>>() {})

                    val result = mcp.callTool(
                        name = toolName,
                        arguments = toolArgs ?: emptyMap()
                    )
                    finalText.add("[Calling tool $toolName with args $toolArgs]")

                    messages.add(
                        MessageParam.builder()
                            .role(MessageParam.Role.USER)
                            .content(
                                """
                                    "type": "tool_result",
                                    "tool_name": $toolName,
                                    "result": ${result?.content?.joinToString("\n") { (it as TextContent).text ?: "" }}
                                """.trimIndent()
                            )
                            .build()
                    )

                    val aiResponse = anthropic.messages().create(
                        messageParamsBuilder
                            .messages(messages)
                            .build()
                    )

                    finalText.add(aiResponse.content().first().text().getOrNull()?.text() ?: "")
                }
            }
        }

        return finalText.joinToString("\n", prefix = "", postfix = "")
    }
    ```

    ### Interactive chat

    We'll add the chat loop:

    ```kotlin
    suspend fun chatLoop() {
        println("\nMCP Client Started!")
        println("Type your queries or 'quit' to exit.")

        while (true) {
            print("\nQuery: ")
            val message = readLine() ?: break
            if (message.lowercase() == "quit") break
            val response = processQuery(message)
            println("\n$response")
        }
    }
    ```

    ### Main entry point

    Finally, we'll add the main execution function:

    ```kotlin
    fun main(args: Array<String>) = runBlocking {
        if (args.isEmpty()) throw IllegalArgumentException("Usage: java -jar <your_path>/build/libs/kotlin-mcp-client-0.1.0-all.jar <path_to_server_script>")
        val serverPath = args.first()
        val client = MCPClient()
        client.use {
            client.connectToServer(serverPath)
            client.chatLoop()
        }
    }
    ```

    ## Running the client

    To run your client with any MCP server:

    ```bash
    ./gradlew build

    # Run the client
    java -jar build/libs/<your-jar-name>.jar path/to/server.jar # jvm server
    java -jar build/libs/<your-jar-name>.jar path/to/server.py # python server
    java -jar build/libs/<your-jar-name>.jar path/to/build/index.js # node server
    ```

    <Note>
      If you're continuing the weather tutorial from the server quickstart, your command might look something like this: `java -jar build/libs/kotlin-mcp-client-0.1.0-all.jar .../samples/weather-stdio-server/build/libs/weather-stdio-server-0.1.0-all.jar`
    </Note>

    **The client will:**

    1. Connect to the specified server
    2. List available tools
    3. Start an interactive chat session where you can:
       * Enter queries
       * See tool executions
       * Get responses from Claude

    ## How it works

    Here's a high-level workflow schema:

    ```mermaid
    ---
    config:
        theme: neutral
    ---
    sequenceDiagram
        actor User
        participant Client
        participant Claude
        participant MCP_Server as MCP Server
        participant Tools

        User->>Client: Send query
        Client<<->>MCP_Server: Get available tools
        Client->>Claude: Send query with tool descriptions
        Claude-->>Client: Decide tool execution
        Client->>MCP_Server: Request tool execution
        MCP_Server->>Tools: Execute chosen tools
        Tools-->>MCP_Server: Return results
        MCP_Server-->>Client: Send results
        Client->>Claude: Send tool results
        Claude-->>Client: Provide final response
        Client-->>User: Display response
    ```

    When you submit a query:

    1. The client gets the list of available tools from the server
    2. Your query is sent to Claude along with tool descriptions
    3. Claude decides which tools (if any) to use
    4. The client executes any requested tool calls through the server
    5. Results are sent back to Claude
    6. Claude provides a natural language response
    7. The response is displayed to you

    ## Best practices

    1. **Error Handling**
       * Leverage Kotlin's type system to model errors explicitly
       * Wrap external tool and API calls in `try-catch` blocks when exceptions are possible
       * Provide clear and meaningful error messages
       * Handle network timeouts and connection issues gracefully

    2. **Security**
       * Store API keys and secrets securely in `local.properties`, environment variables, or secret managers
       * Validate all external responses to avoid unexpected or unsafe data usage
       * Be cautious with permissions and trust boundaries when using tools

    ## Troubleshooting

    ### Server Path Issues

    * Double-check the path to your server script is correct
    * Use the absolute path if the relative path isn't working
    * For Windows users, make sure to use forward slashes (/) or escaped backslashes (\\) in the path
    * Make sure that the required runtime is installed (java for Java, npm for Node.js, or uv for Python)
    * Verify the server file has the correct extension (.jar for Java, .js for Node.js or .py for Python)

    Example of correct path usage:

    ```bash
    # Relative path
    java -jar build/libs/client.jar ./server/build/libs/server.jar

    # Absolute path
    java -jar build/libs/client.jar /Users/username/projects/mcp-server/build/libs/server.jar

    # Windows path (either format works)
    java -jar build/libs/client.jar C:/projects/mcp-server/build/libs/server.jar
    java -jar build/libs/client.jar C:\\projects\\mcp-server\\build\\libs\\server.jar
    ```

    ### Response Timing

    * The first response might take up to 30 seconds to return
    * This is normal and happens while:
      * The server initializes
      * Claude processes the query
      * Tools are being executed
    * Subsequent responses are typically faster
    * Don't interrupt the process during this initial waiting period

    ### Common Error Messages

    If you see:

    * `Connection refused`: Ensure the server is running and the path is correct
    * `Tool execution failed`: Verify the tool's required environment variables are set
    * `ANTHROPIC_API_KEY is not set`: Check your environment variables
  </Tab>

  <Tab title="C#">
    [You can find the complete code for this tutorial here.](https://github.com/modelcontextprotocol/csharp-sdk/tree/main/samples/QuickstartClient)

    ## System Requirements

    Before starting, ensure your system meets these requirements:

    * .NET 8.0 or higher
    * Anthropic API key (Claude)
    * Windows, Linux, or MacOS

    ## Setting up your environment

    First, create a new .NET project:

    ```bash
    dotnet new console -n QuickstartClient
    cd QuickstartClient
    ```

    Then, add the required dependencies to your project:

    ```bash
    dotnet add package ModelContextProtocol --prerelease
    dotnet add package Anthropic.SDK
    dotnet add package Microsoft.Extensions.Hosting
    ```

    ## Setting up your API key

    You'll need an Anthropic API key from the [Anthropic Console](https://console.anthropic.com/settings/keys).

    ```bash
    dotnet user-secrets init
    dotnet user-secrets set "ANTHROPIC_API_KEY" "<your key here>"
    ```

    ## Creating the Client

    ### Basic Client Structure

    First, let's setup the basic client class in the file `Program.cs`:

    ```csharp
    using Anthropic.SDK;
    using Microsoft.Extensions.AI;
    using Microsoft.Extensions.Configuration;
    using Microsoft.Extensions.Hosting;
    using ModelContextProtocol.Client;
    using ModelContextProtocol.Protocol.Transport;

    var builder = Host.CreateApplicationBuilder(args);

    builder.Configuration
        .AddEnvironmentVariables()
        .AddUserSecrets<Program>();
    ```

    This creates the beginnings of a .NET console application that can read the API key from user secrets.

    Next, we'll setup the MCP Client:

    ```csharp
    var (command, arguments) = GetCommandAndArguments(args);

    var clientTransport = new StdioClientTransport(new()
    {
        Name = "Demo Server",
        Command = command,
        Arguments = arguments,
    });

    await using var mcpClient = await McpClientFactory.CreateAsync(clientTransport);

    var tools = await mcpClient.ListToolsAsync();
    foreach (var tool in tools)
    {
        Console.WriteLine($"Connected to server with tools: {tool.Name}");
    }
    ```

    Add this function at the end of the `Program.cs` file:

    ```csharp
    static (string command, string[] arguments) GetCommandAndArguments(string[] args)
    {
        return args switch
        {
            [var script] when script.EndsWith(".py") => ("python", args),
            [var script] when script.EndsWith(".js") => ("node", args),
            [var script] when Directory.Exists(script) || (File.Exists(script) && script.EndsWith(".csproj")) => ("dotnet", ["run", "--project", script, "--no-build"]),
            _ => throw new NotSupportedException("An unsupported server script was provided. Supported scripts are .py, .js, or .csproj")
        };
    }
    ```

    This creates a MCP client that will connect to a server that is provided as a command line argument. It then lists the available tools from the connected server.

    ### Query processing logic

    Now let's add the core functionality for processing queries and handling tool calls:

    ```csharp
    using var anthropicClient = new AnthropicClient(new APIAuthentication(builder.Configuration["ANTHROPIC_API_KEY"]))
        .Messages
        .AsBuilder()
        .UseFunctionInvocation()
        .Build();

    var options = new ChatOptions
    {
        MaxOutputTokens = 1000,
        ModelId = "claude-3-5-sonnet-20241022",
        Tools = [.. tools]
    };

    Console.ForegroundColor = ConsoleColor.Green;
    Console.WriteLine("MCP Client Started!");
    Console.ResetColor();

    PromptForInput();
    while(Console.ReadLine() is string query && !"exit".Equals(query, StringComparison.OrdinalIgnoreCase))
    {
        if (string.IsNullOrWhiteSpace(query))
        {
            PromptForInput();
            continue;
        }

        await foreach (var message in anthropicClient.GetStreamingResponseAsync(query, options))
        {
            Console.Write(message);
        }
        Console.WriteLine();

        PromptForInput();
    }

    static void PromptForInput()
    {
        Console.WriteLine("Enter a command (or 'exit' to quit):");
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.Write("> ");
        Console.ResetColor();
    }
    ```

    ## Key Components Explained

    ### 1. Client Initialization

    * The client is initialized using `McpClientFactory.CreateAsync()`, which sets up the transport type and command to run the server.

    ### 2. Server Connection

    * Supports Python, Node.js, and .NET servers.
    * The server is started using the command specified in the arguments.
    * Configures to use stdio for communication with the server.
    * Initializes the session and available tools.

    ### 3. Query Processing

    * Leverages [Microsoft.Extensions.AI](https://learn.microsoft.com/dotnet/ai/ai-extensions) for the chat client.
    * Configures the `IChatClient` to use automatic tool (function) invocation.
    * The client reads user input and sends it to the server.
    * The server processes the query and returns a response.
    * The response is displayed to the user.

    ## Running the Client

    To run your client with any MCP server:

    ```bash
    dotnet run -- path/to/server.csproj # dotnet server
    dotnet run -- path/to/server.py # python server
    dotnet run -- path/to/server.js # node server
    ```

    <Note>
      If you're continuing the weather tutorial from the server quickstart, your command might look something like this: `dotnet run -- path/to/QuickstartWeatherServer`.
    </Note>

    The client will:

    1. Connect to the specified server
    2. List available tools
    3. Start an interactive chat session where you can:
       * Enter queries
       * See tool executions
       * Get responses from Claude
    4. Exit the session when done

    Here's an example of what it should look like it connected to a weather server quickstart:

    <Frame>
      <img src="https://mintlify.s3.us-west-1.amazonaws.com/mcp/images/quickstart-dotnet-client.png" />
    </Frame>
  </Tab>
</Tabs>

## Next steps

<CardGroup cols={2}>
  <Card title="Example servers" icon="grid" href="/examples">
    Check out our gallery of official MCP servers and implementations
  </Card>

  <Card title="Clients" icon="cubes" href="/clients">
    View the list of clients that support MCP integrations
  </Card>

  <Card title="Building MCP with LLMs" icon="comments" href="/tutorials/building-mcp-with-llms">
    Learn how to use LLMs like Claude to speed up your MCP development
  </Card>

  <Card title="Core architecture" icon="sitemap" href="/docs/concepts/architecture">
    Understand how MCP connects clients, servers, and LLMs
  </Card>
</CardGroup>



<mcp_client.py>

import asyncio
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server
        
        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")
            
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )
        
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        
        await self.session.initialize()
        
        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        response = await self.session.list_tools()
        available_tools = [{ 
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]

        # Initial Claude API call
        response = self.anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=messages,
            tools=available_tools
        )

        # Process response and handle tool calls
        tool_results = []
        final_text = []

        for content in response.content:
            if content.type == 'text':
                final_text.append(content.text)
            elif content.type == 'tool_use':
                tool_name = content.name
                tool_args = content.input
                
                # Execute tool call
                result = await self.session.call_tool(tool_name, tool_args)
                tool_results.append({"call": tool_name, "result": result})
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

                # Continue conversation with tool results
                if hasattr(content, 'text') and content.text:
                    messages.append({
                      "role": "assistant",
                      "content": content.text
                    })
                messages.append({
                    "role": "user", 
                    "content": result.content
                })

                # Get next response from Claude
                response = self.anthropic.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    messages=messages,
                )

                final_text.append(response.content[0].text)

        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() == 'quit':
                    break
                    
                response = await self.process_query(query)
                print("\n" + response)
                    
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)
        
    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())
    
</mcp_client.py>


└── examples
    └── clients
        └── simple-chatbot
            ├── .python-version
            ├── README.MD
            ├── mcp_simple_chatbot
                ├── .env.example
                ├── main.py
                ├── requirements.txt
                ├── servers_config.json
                └── test.db
            ├── pyproject.toml
            └── uv.lock


/examples/clients/simple-chatbot/.python-version:
--------------------------------------------------------------------------------
1 | 3.10
2 | 


--------------------------------------------------------------------------------
/examples/clients/simple-chatbot/README.MD:
--------------------------------------------------------------------------------
  1 | # MCP Simple Chatbot
  2 | 
  3 | This example demonstrates how to integrate the Model Context Protocol (MCP) into a simple CLI chatbot. The implementation showcases MCP's flexibility by supporting multiple tools through MCP servers and is compatible with any LLM provider that follows OpenAI API standards.
  4 | 
  5 | ## Requirements
  6 | 
  7 | - Python 3.10
  8 | - `python-dotenv`
  9 | - `requests`
 10 | - `mcp`
 11 | - `uvicorn`
 12 | 
 13 | ## Installation
 14 | 
 15 | 1. **Install the dependencies:**
 16 | 
 17 |    ```bash
 18 |    pip install -r requirements.txt
 19 |    ```
 20 | 
 21 | 2. **Set up environment variables:**
 22 | 
 23 |    Create a `.env` file in the root directory and add your API key:
 24 | 
 25 |    ```plaintext
 26 |    LLM_API_KEY=your_api_key_here
 27 |    ```
 28 |    **Note:** The current implementation is configured to use the Groq API endpoint (`https://api.groq.com/openai/v1/chat/completions`) with the `llama-3.2-90b-vision-preview` model. If you plan to use a different LLM provider, you'll need to modify the `LLMClient` class in `main.py` to use the appropriate endpoint URL and model parameters.
 29 | 
 30 | 3. **Configure servers:**
 31 | 
 32 |    The `servers_config.json` follows the same structure as Claude Desktop, allowing for easy integration of multiple servers. 
 33 |    Here's an example:
 34 | 
 35 |    ```json
 36 |    {
 37 |      "mcpServers": {
 38 |        "sqlite": {
 39 |          "command": "uvx",
 40 |          "args": ["mcp-server-sqlite", "--db-path", "./test.db"]
 41 |        },
 42 |        "puppeteer": {
 43 |          "command": "npx",
 44 |          "args": ["-y", "@modelcontextprotocol/server-puppeteer"]
 45 |        }
 46 |      }
 47 |    }
 48 |    ```
 49 |    Environment variables are supported as well. Pass them as you would with the Claude Desktop App.
 50 | 
 51 |    Example:
 52 |    ```json
 53 |    {
 54 |      "mcpServers": {
 55 |        "server_name": {
 56 |          "command": "uvx",
 57 |          "args": ["mcp-server-name", "--additional-args"],
 58 |          "env": {
 59 |            "API_KEY": "your_api_key_here"
 60 |          }
 61 |        }
 62 |      }
 63 |    }
 64 |    ```
 65 | 
 66 | ## Usage
 67 | 
 68 | 1. **Run the client:**
 69 | 
 70 |    ```bash
 71 |    python main.py
 72 |    ```
 73 | 
 74 | 2. **Interact with the assistant:**
 75 |    
 76 |    The assistant will automatically detect available tools and can respond to queries based on the tools provided by the configured servers.
 77 | 
 78 | 3. **Exit the session:**
 79 | 
 80 |    Type `quit` or `exit` to end the session.
 81 | 
 82 | ## Architecture
 83 | 
 84 | - **Tool Discovery**: Tools are automatically discovered from configured servers.
 85 | - **System Prompt**: Tools are dynamically included in the system prompt, allowing the LLM to understand available capabilities.
 86 | - **Server Integration**: Supports any MCP-compatible server, tested with various server implementations including Uvicorn and Node.js.
 87 | 
 88 | ### Class Structure
 89 | - **Configuration**: Manages environment variables and server configurations
 90 | - **Server**: Handles MCP server initialization, tool discovery, and execution
 91 | - **Tool**: Represents individual tools with their properties and formatting
 92 | - **LLMClient**: Manages communication with the LLM provider
 93 | - **ChatSession**: Orchestrates the interaction between user, LLM, and tools
 94 | 
 95 | ### Logic Flow
 96 | 
 97 | 1. **Tool Integration**:
 98 |    - Tools are dynamically discovered from MCP servers
 99 |    - Tool descriptions are automatically included in system prompt
100 |    - Tool execution is handled through standardized MCP protocol
101 | 
102 | 2. **Runtime Flow**:
103 |    - User input is received
104 |    - Input is sent to LLM with context of available tools
105 |    - LLM response is parsed:
106 |      - If it's a tool call → execute tool and return result
107 |      - If it's a direct response → return to user
108 |    - Tool results are sent back to LLM for interpretation
109 |    - Final response is presented to user
110 | 
111 | 
112 | 


--------------------------------------------------------------------------------
/examples/clients/simple-chatbot/mcp_simple_chatbot/.env.example:
--------------------------------------------------------------------------------
1 | LLM_API_KEY=gsk_1234567890
2 | 


--------------------------------------------------------------------------------
/examples/clients/simple-chatbot/mcp_simple_chatbot/main.py:
--------------------------------------------------------------------------------
  1 | import asyncio
  2 | import json
  3 | import logging
  4 | import os
  5 | import shutil
  6 | from contextlib import AsyncExitStack
  7 | from typing import Any
  8 | 
  9 | import httpx
 10 | from dotenv import load_dotenv
 11 | from mcp import ClientSession, StdioServerParameters
 12 | from mcp.client.stdio import stdio_client
 13 | 
 14 | # Configure logging
 15 | logging.basicConfig(
 16 |     level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
 17 | )
 18 | 
 19 | 
 20 | class Configuration:
 21 |     """Manages configuration and environment variables for the MCP client."""
 22 | 
 23 |     def __init__(self) -> None:
 24 |         """Initialize configuration with environment variables."""
 25 |         self.load_env()
 26 |         self.api_key = os.getenv("LLM_API_KEY")
 27 | 
 28 |     @staticmethod
 29 |     def load_env() -> None:
 30 |         """Load environment variables from .env file."""
 31 |         load_dotenv()
 32 | 
 33 |     @staticmethod
 34 |     def load_config(file_path: str) -> dict[str, Any]:
 35 |         """Load server configuration from JSON file.
 36 | 
 37 |         Args:
 38 |             file_path: Path to the JSON configuration file.
 39 | 
 40 |         Returns:
 41 |             Dict containing server configuration.
 42 | 
 43 |         Raises:
 44 |             FileNotFoundError: If configuration file doesn't exist.
 45 |             JSONDecodeError: If configuration file is invalid JSON.
 46 |         """
 47 |         with open(file_path, "r") as f:
 48 |             return json.load(f)
 49 | 
 50 |     @property
 51 |     def llm_api_key(self) -> str:
 52 |         """Get the LLM API key.
 53 | 
 54 |         Returns:
 55 |             The API key as a string.
 56 | 
 57 |         Raises:
 58 |             ValueError: If the API key is not found in environment variables.
 59 |         """
 60 |         if not self.api_key:
 61 |             raise ValueError("LLM_API_KEY not found in environment variables")
 62 |         return self.api_key
 63 | 
 64 | 
 65 | class Server:
 66 |     """Manages MCP server connections and tool execution."""
 67 | 
 68 |     def __init__(self, name: str, config: dict[str, Any]) -> None:
 69 |         self.name: str = name
 70 |         self.config: dict[str, Any] = config
 71 |         self.stdio_context: Any | None = None
 72 |         self.session: ClientSession | None = None
 73 |         self._cleanup_lock: asyncio.Lock = asyncio.Lock()
 74 |         self.exit_stack: AsyncExitStack = AsyncExitStack()
 75 | 
 76 |     async def initialize(self) -> None:
 77 |         """Initialize the server connection."""
 78 |         command = (
 79 |             shutil.which("npx")
 80 |             if self.config["command"] == "npx"
 81 |             else self.config["command"]
 82 |         )
 83 |         if command is None:
 84 |             raise ValueError("The command must be a valid string and cannot be None.")
 85 | 
 86 |         server_params = StdioServerParameters(
 87 |             command=command,
 88 |             args=self.config["args"],
 89 |             env={**os.environ, **self.config["env"]}
 90 |             if self.config.get("env")
 91 |             else None,
 92 |         )
 93 |         try:
 94 |             stdio_transport = await self.exit_stack.enter_async_context(
 95 |                 stdio_client(server_params)
 96 |             )
 97 |             read, write = stdio_transport
 98 |             session = await self.exit_stack.enter_async_context(
 99 |                 ClientSession(read, write)
100 |             )
101 |             await session.initialize()
102 |             self.session = session
103 |         except Exception as e:
104 |             logging.error(f"Error initializing server {self.name}: {e}")
105 |             await self.cleanup()
106 |             raise
107 | 
108 |     async def list_tools(self) -> list[Any]:
109 |         """List available tools from the server.
110 | 
111 |         Returns:
112 |             A list of available tools.
113 | 
114 |         Raises:
115 |             RuntimeError: If the server is not initialized.
116 |         """
117 |         if not self.session:
118 |             raise RuntimeError(f"Server {self.name} not initialized")
119 | 
120 |         tools_response = await self.session.list_tools()
121 |         tools = []
122 | 
123 |         for item in tools_response:
124 |             if isinstance(item, tuple) and item[0] == "tools":
125 |                 tools.extend(
126 |                     Tool(tool.name, tool.description, tool.inputSchema)
127 |                     for tool in item[1]
128 |                 )
129 | 
130 |         return tools
131 | 
132 |     async def execute_tool(
133 |         self,
134 |         tool_name: str,
135 |         arguments: dict[str, Any],
136 |         retries: int = 2,
137 |         delay: float = 1.0,
138 |     ) -> Any:
139 |         """Execute a tool with retry mechanism.
140 | 
141 |         Args:
142 |             tool_name: Name of the tool to execute.
143 |             arguments: Tool arguments.
144 |             retries: Number of retry attempts.
145 |             delay: Delay between retries in seconds.
146 | 
147 |         Returns:
148 |             Tool execution result.
149 | 
150 |         Raises:
151 |             RuntimeError: If server is not initialized.
152 |             Exception: If tool execution fails after all retries.
153 |         """
154 |         if not self.session:
155 |             raise RuntimeError(f"Server {self.name} not initialized")
156 | 
157 |         attempt = 0
158 |         while attempt < retries:
159 |             try:
160 |                 logging.info(f"Executing {tool_name}...")
161 |                 result = await self.session.call_tool(tool_name, arguments)
162 | 
163 |                 return result
164 | 
165 |             except Exception as e:
166 |                 attempt += 1
167 |                 logging.warning(
168 |                     f"Error executing tool: {e}. Attempt {attempt} of {retries}."
169 |                 )
170 |                 if attempt < retries:
171 |                     logging.info(f"Retrying in {delay} seconds...")
172 |                     await asyncio.sleep(delay)
173 |                 else:
174 |                     logging.error("Max retries reached. Failing.")
175 |                     raise
176 | 
177 |     async def cleanup(self) -> None:
178 |         """Clean up server resources."""
179 |         async with self._cleanup_lock:
180 |             try:
181 |                 await self.exit_stack.aclose()
182 |                 self.session = None
183 |                 self.stdio_context = None
184 |             except Exception as e:
185 |                 logging.error(f"Error during cleanup of server {self.name}: {e}")
186 | 
187 | 
188 | class Tool:
189 |     """Represents a tool with its properties and formatting."""
190 | 
191 |     def __init__(
192 |         self, name: str, description: str, input_schema: dict[str, Any]
193 |     ) -> None:
194 |         self.name: str = name
195 |         self.description: str = description
196 |         self.input_schema: dict[str, Any] = input_schema
197 | 
198 |     def format_for_llm(self) -> str:
199 |         """Format tool information for LLM.
200 | 
201 |         Returns:
202 |             A formatted string describing the tool.
203 |         """
204 |         args_desc = []
205 |         if "properties" in self.input_schema:
206 |             for param_name, param_info in self.input_schema["properties"].items():
207 |                 arg_desc = (
208 |                     f"- {param_name}: {param_info.get('description', 'No description')}"
209 |                 )
210 |                 if param_name in self.input_schema.get("required", []):
211 |                     arg_desc += " (required)"
212 |                 args_desc.append(arg_desc)
213 | 
214 |         return f"""
215 | Tool: {self.name}
216 | Description: {self.description}
217 | Arguments:
218 | {chr(10).join(args_desc)}
219 | """
220 | 
221 | 
222 | class LLMClient:
223 |     """Manages communication with the LLM provider."""
224 | 
225 |     def __init__(self, api_key: str) -> None:
226 |         self.api_key: str = api_key
227 | 
228 |     def get_response(self, messages: list[dict[str, str]]) -> str:
229 |         """Get a response from the LLM.
230 | 
231 |         Args:
232 |             messages: A list of message dictionaries.
233 | 
234 |         Returns:
235 |             The LLM's response as a string.
236 | 
237 |         Raises:
238 |             httpx.RequestError: If the request to the LLM fails.
239 |         """
240 |         url = "https://api.groq.com/openai/v1/chat/completions"
241 | 
242 |         headers = {
243 |             "Content-Type": "application/json",
244 |             "Authorization": f"Bearer {self.api_key}",
245 |         }
246 |         payload = {
247 |             "messages": messages,
248 |             "model": "meta-llama/llama-4-scout-17b-16e-instruct",
249 |             "temperature": 0.7,
250 |             "max_tokens": 4096,
251 |             "top_p": 1,
252 |             "stream": False,
253 |             "stop": None,
254 |         }
255 | 
256 |         try:
257 |             with httpx.Client() as client:
258 |                 response = client.post(url, headers=headers, json=payload)
259 |                 response.raise_for_status
260 |                 data = response.json()
261 |                 return data["choices"][0]["message"]["content"]
262 | 
263 |         except httpx.RequestError as e:
264 |             error_message = f"Error getting LLM response: {str(e)}"
265 |             logging.error(error_message)
266 | 
267 |             if isinstance(e, httpx.HTTPStatusError):
268 |                 status_code = e.response.status_code
269 |                 logging.error(f"Status code: {status_code}")
270 |                 logging.error(f"Response details: {e.response.text}")
271 | 
272 |             return (
273 |                 f"I encountered an error: {error_message}. "
274 |                 "Please try again or rephrase your request."
275 |             )
276 | 
277 | 
278 | class ChatSession:
279 |     """Orchestrates the interaction between user, LLM, and tools."""
280 | 
281 |     def __init__(self, servers: list[Server], llm_client: LLMClient) -> None:
282 |         self.servers: list[Server] = servers
283 |         self.llm_client: LLMClient = llm_client
284 | 
285 |     async def cleanup_servers(self) -> None:
286 |         """Clean up all servers properly."""
287 |         for server in reversed(self.servers):
288 |             try:
289 |                 await server.cleanup()
290 |             except Exception as e:
291 |                 logging.warning(f"Warning during final cleanup: {e}")
292 | 
293 |     async def process_llm_response(self, llm_response: str) -> str:
294 |         """Process the LLM response and execute tools if needed.
295 | 
296 |         Args:
297 |             llm_response: The response from the LLM.
298 | 
299 |         Returns:
300 |             The result of tool execution or the original response.
301 |         """
302 |         import json
303 | 
304 |         try:
305 |             tool_call = json.loads(llm_response)
306 |             if "tool" in tool_call and "arguments" in tool_call:
307 |                 logging.info(f"Executing tool: {tool_call['tool']}")
308 |                 logging.info(f"With arguments: {tool_call['arguments']}")
309 | 
310 |                 for server in self.servers:
311 |                     tools = await server.list_tools()
312 |                     if any(tool.name == tool_call["tool"] for tool in tools):
313 |                         try:
314 |                             result = await server.execute_tool(
315 |                                 tool_call["tool"], tool_call["arguments"]
316 |                             )
317 | 
318 |                             if isinstance(result, dict) and "progress" in result:
319 |                                 progress = result["progress"]
320 |                                 total = result["total"]
321 |                                 percentage = (progress / total) * 100
322 |                                 logging.info(
323 |                                     f"Progress: {progress}/{total} ({percentage:.1f}%)"
324 |                                 )
325 | 
326 |                             return f"Tool execution result: {result}"
327 |                         except Exception as e:
328 |                             error_msg = f"Error executing tool: {str(e)}"
329 |                             logging.error(error_msg)
330 |                             return error_msg
331 | 
332 |                 return f"No server found with tool: {tool_call['tool']}"
333 |             return llm_response
334 |         except json.JSONDecodeError:
335 |             return llm_response
336 | 
337 |     async def start(self) -> None:
338 |         """Main chat session handler."""
339 |         try:
340 |             for server in self.servers:
341 |                 try:
342 |                     await server.initialize()
343 |                 except Exception as e:
344 |                     logging.error(f"Failed to initialize server: {e}")
345 |                     await self.cleanup_servers()
346 |                     return
347 | 
348 |             all_tools = []
349 |             for server in self.servers:
350 |                 tools = await server.list_tools()
351 |                 all_tools.extend(tools)
352 | 
353 |             tools_description = "\n".join([tool.format_for_llm() for tool in all_tools])
354 | 
355 |             system_message = (
356 |                 "You are a helpful assistant with access to these tools:\n\n"
357 |                 f"{tools_description}\n"
358 |                 "Choose the appropriate tool based on the user's question. "
359 |                 "If no tool is needed, reply directly.\n\n"
360 |                 "IMPORTANT: When you need to use a tool, you must ONLY respond with "
361 |                 "the exact JSON object format below, nothing else:\n"
362 |                 "{\n"
363 |                 '    "tool": "tool-name",\n'
364 |                 '    "arguments": {\n'
365 |                 '        "argument-name": "value"\n'
366 |                 "    }\n"
367 |                 "}\n\n"
368 |                 "After receiving a tool's response:\n"
369 |                 "1. Transform the raw data into a natural, conversational response\n"
370 |                 "2. Keep responses concise but informative\n"
371 |                 "3. Focus on the most relevant information\n"
372 |                 "4. Use appropriate context from the user's question\n"
373 |                 "5. Avoid simply repeating the raw data\n\n"
374 |                 "Please use only the tools that are explicitly defined above."
375 |             )
376 | 
377 |             messages = [{"role": "system", "content": system_message}]
378 | 
379 |             while True:
380 |                 try:
381 |                     user_input = input("You: ").strip().lower()
382 |                     if user_input in ["quit", "exit"]:
383 |                         logging.info("\nExiting...")
384 |                         break
385 | 
386 |                     messages.append({"role": "user", "content": user_input})
387 | 
388 |                     llm_response = self.llm_client.get_response(messages)
389 |                     logging.info("\nAssistant: %s", llm_response)
390 | 
391 |                     result = await self.process_llm_response(llm_response)
392 | 
393 |                     if result != llm_response:
394 |                         messages.append({"role": "assistant", "content": llm_response})
395 |                         messages.append({"role": "system", "content": result})
396 | 
397 |                         final_response = self.llm_client.get_response(messages)
398 |                         logging.info("\nFinal response: %s", final_response)
399 |                         messages.append(
400 |                             {"role": "assistant", "content": final_response}
401 |                         )
402 |                     else:
403 |                         messages.append({"role": "assistant", "content": llm_response})
404 | 
405 |                 except KeyboardInterrupt:
406 |                     logging.info("\nExiting...")
407 |                     break
408 | 
409 |         finally:
410 |             await self.cleanup_servers()
411 | 
412 | 
413 | async def main() -> None:
414 |     """Initialize and run the chat session."""
415 |     config = Configuration()
416 |     server_config = config.load_config("servers_config.json")
417 |     servers = [
418 |         Server(name, srv_config)
419 |         for name, srv_config in server_config["mcpServers"].items()
420 |     ]
421 |     llm_client = LLMClient(config.llm_api_key)
422 |     chat_session = ChatSession(servers, llm_client)
423 |     await chat_session.start()
424 | 
425 | 
426 | if __name__ == "__main__":
427 |     asyncio.run(main())
428 | 


--------------------------------------------------------------------------------
/examples/clients/simple-chatbot/mcp_simple_chatbot/requirements.txt:
--------------------------------------------------------------------------------
1 | python-dotenv>=1.0.0
2 | requests>=2.31.0
3 | mcp>=1.0.0
4 | uvicorn>=0.32.1


--------------------------------------------------------------------------------
/examples/clients/simple-chatbot/mcp_simple_chatbot/servers_config.json:
--------------------------------------------------------------------------------
 1 | {
 2 |   "mcpServers": {
 3 |     "sqlite": {
 4 |       "command": "uvx",
 5 |       "args": ["mcp-server-sqlite", "--db-path", "./test.db"]
 6 |     },
 7 |     "puppeteer": {
 8 |       "command": "npx",
 9 |       "args": ["-y", "@modelcontextprotocol/server-puppeteer"]
10 |     }
11 |   }
12 | }


--------------------------------------------------------------------------------
/examples/clients/simple-chatbot/mcp_simple_chatbot/test.db:
--------------------------------------------------------------------------------
https://raw.githubusercontent.com/modelcontextprotocol/python-sdk/main/examples/clients/simple-chatbot/mcp_simple_chatbot/test.db


--------------------------------------------------------------------------------
/examples/clients/simple-chatbot/pyproject.toml:
--------------------------------------------------------------------------------
 1 | [project]
 2 | name = "mcp-simple-chatbot"
 3 | version = "0.1.0"
 4 | description = "A simple CLI chatbot using the Model Context Protocol (MCP)"
 5 | readme = "README.md"
 6 | requires-python = ">=3.10"
 7 | authors = [{ name = "Edoardo Cilia" }]
 8 | keywords = ["mcp", "llm", "chatbot", "cli"]
 9 | license = { text = "MIT" }
10 | classifiers = [
11 |     "Development Status :: 4 - Beta",
12 |     "Intended Audience :: Developers",
13 |     "License :: OSI Approved :: MIT License",
14 |     "Programming Language :: Python :: 3",
15 |     "Programming Language :: Python :: 3.10",
16 | ]
17 | dependencies = [
18 |     "python-dotenv>=1.0.0",
19 |     "requests>=2.31.0",
20 |     "mcp>=1.0.0",
21 |     "uvicorn>=0.32.1"
22 | ]
23 | 
24 | [project.script
25 | mcp-simple-chatbot = "mcp_simple_chatbot.client:main"
26 | 
27 | [build-system]
28 | requires = ["hatchling"]
29 | build-backend = "hatchling.build"
30 | 
31 | [tool.hatch.build.targets.wheel]
32 | packages = ["mcp_simple_chatbot"]
33 | 
34 | [tool.pyright]
35 | include = ["mcp_simple_chatbot"]
36 | venvPath = "."
37 | venv = ".venv"
38 | 
39 | [tool.ruff.lint]
40 | select = ["E", "F", "I"]
41 | ignore = []
42 | 
43 | [tool.ruff]
44 | line-length = 88
45 | target-version = "py310"
46 | 
47 | [tool.uv]
48 | dev-dependencies = ["pyright>=1.1.379", "pytest>=8.3.3", "ruff>=0.6.9"]
49 | 

