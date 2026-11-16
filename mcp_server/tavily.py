"""Connect to Tavily's hosted MCP search tool."""

import asyncio
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient


class TavilySearchUnavailable(RuntimeError):
    """The hosted search service could not serve a research request."""


class TavilySearchLimit(TavilySearchUnavailable):
    """Tavily rejected a search because its quota was reached."""


async def load_search_tool(api_key: str | None = None) -> BaseTool:
    key = (api_key or "").strip()
    headers = {"Authorization": f"Bearer {key}"} if key else {"X-Tavily-Access-Mode": "keyless"}
    client = MultiServerMCPClient({
        "tavily": {
            "transport": "streamable_http",
            "url": "https://mcp.tavily.com/mcp/",
            "headers": headers,
        },
    })
    try:
        async with asyncio.timeout(12):
            tools = await client.get_tools(server_name="tavily")
    except Exception as error:
        raise TavilySearchUnavailable("Could not connect to Tavily MCP.") from error
    for tool in tools:
        if tool.name in {"tavily_search", "tavily-search"}:
            return tool
    raise TavilySearchUnavailable("Tavily MCP did not advertise its search tool.")
