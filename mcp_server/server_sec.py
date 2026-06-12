"""LedgerIQ — SEC EDGAR MCP Server.

Exposes the SEC EDGAR fetcher as Model Context Protocol (MCP) tools, so an
agent (Claude, or any MCP client) can retrieve real filings on demand as part
of an agentic RAG workflow.

This demonstrates the MCP-server pattern: the retrieval capability is a
*tool the model calls*, not hard-wired into the app. The same fetcher core
(scripts/edgar_fetcher.py) powers both this server and the offline corpus build.

Tools exposed:
  - list_filings(ticker, forms, limit)      -> recent filings metadata
  - fetch_filing(url, max_chars)            -> cleaned filing text
  - find_sections(url)                      -> locate standard 10-K/10-Q sections

Run (stdio transport):
    python mcp_server/server.py

Register in an MCP client (e.g. Claude Desktop) by pointing it at this command.
A claude_desktop_config.json snippet is in mcp_server/README.md.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# allow importing the shared fetcher core
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import edgar_fetcher as ef  # noqa: E402

from mcp.server import Server  # noqa: E402
from mcp.server.stdio import stdio_server  # noqa: E402
from mcp.types import Tool, TextContent  # noqa: E402

app = Server("ledgeriq-edgar")


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="list_filings",
            description="List a public company's recent SEC filings (10-K, 10-Q, 8-K) "
                        "from EDGAR. Returns metadata incl. the direct document URL.",
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker, e.g. AAPL"},
                    "forms": {"type": "array", "items": {"type": "string"},
                              "description": "Form types to include, e.g. [\"10-K\",\"10-Q\"]"},
                    "limit": {"type": "integer", "description": "Max filings to return"},
                },
                "required": ["ticker"],
            },
        ),
        Tool(
            name="fetch_filing",
            description="Fetch a filing document by its SEC.gov URL and return cleaned, "
                        "readable text (HTML stripped). Use a URL from list_filings.",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Direct SEC.gov document URL"},
                    "max_chars": {"type": "integer", "description": "Truncate to N chars"},
                },
                "required": ["url"],
            },
        ),
        Tool(
            name="find_sections",
            description="Fetch a filing and locate standard sections (Risk Factors, MD&A, "
                        "Liquidity, Revenue, Controls, Legal) with their character offsets.",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Direct SEC.gov document URL"},
                },
                "required": ["url"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "list_filings":
        forms = tuple(arguments.get("forms") or ["10-K", "10-Q"])
        limit = int(arguments.get("limit", 4))
        filings = ef.recent_filings(arguments["ticker"], forms=forms, limit=limit)
        payload = [ef.asdict(f) for f in filings]
        return [TextContent(type="text", text=json.dumps(payload, indent=2))]

    if name == "fetch_filing":
        max_chars = int(arguments.get("max_chars", 400_000))
        text = ef.fetch_document_text(arguments["url"], max_chars=max_chars)
        return [TextContent(type="text", text=text)]

    if name == "find_sections":
        text = ef.fetch_document_text(arguments["url"])
        low = text.lower()
        found = {}
        for label, pat in ef.SECTION_PATTERNS.items():
            m = re.search(pat, low)
            found[label] = m.start() if m else None
        return [TextContent(type="text", text=json.dumps(found, indent=2))]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
