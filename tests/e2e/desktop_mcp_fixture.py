"""A real stdio MCP peer; tests own its subprocess through McpManager."""

from mcp.server.mcpserver import MCPServer

server = MCPServer("desktop-fixture")


@server.tool()
def fixture_echo(value: str) -> str:
    """Return a value so connection and tool discovery have observable evidence."""
    return value


if __name__ == "__main__":
    server.run(transport="stdio")
