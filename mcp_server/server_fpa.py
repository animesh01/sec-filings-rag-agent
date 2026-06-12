"""LedgerIQ FP&A MCP Server.

Exposes the FP&A document source as Model Context Protocol (MCP) tools, so an
agent (Claude, or any MCP client) can retrieve finance planning documents on
demand as part of an agentic FP&A workflow.

This demonstrates the MCP-server pattern: retrieval is a *tool the model calls*,
not hard-wired into the app. The same source module (scripts/fpa_source.py)
powers both this server and the offline corpus build.

Tools exposed:
  - list_documents(doc_type)        -> available FP&A docs (filterable by type)
  - get_document(url)               -> full text of a document
  - find_variances(threshold_pct)   -> documents reporting budget variances over a threshold

Run (stdio transport):
    python mcp_server/server.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
import fpa_source as src  # noqa: E402

from mcp.server import Server  # noqa: E402
from mcp.server.stdio import stdio_server  # noqa: E402
from mcp.types import Tool, TextContent  # noqa: E402

app = Server("ledgeriq-fpa")


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="list_documents",
            description="List available FP&A documents (variance commentary, forecast "
                        "assumptions, close notes, policies), optionally filtered by type.",
            inputSchema={
                "type": "object",
                "properties": {
                    "doc_type": {"type": "string",
                                 "description": "Optional filter, e.g. 'Policy' or 'Variance'"},
                },
            },
        ),
        Tool(
            name="get_document",
            description="Return the full text of an FP&A document by its url/id.",
            inputSchema={
                "type": "object",
                "properties": {"url": {"type": "string", "description": "Document url/id"}},
                "required": ["url"],
            },
        ),
        Tool(
            name="find_variances",
            description="Return FP&A documents that report budget variances at or above a "
                        "percentage threshold (e.g. the 10% commentary threshold).",
            inputSchema={
                "type": "object",
                "properties": {"threshold_pct": {"type": "number",
                               "description": "Variance threshold in percent, e.g. 10"}},
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    docs = src.load_sample_docs()

    if name == "list_documents":
        dtype = (arguments.get("doc_type") or "").lower()
        out = [{"section": d["section"], "form": d["form"], "company": d["company"],
                "period": d["filing_date"], "url": d["url"]}
               for d in docs if dtype in d["form"].lower()]
        return [TextContent(type="text", text=json.dumps(out, indent=2))]

    if name == "get_document":
        url = arguments["url"]
        for d in docs:
            if d["url"] == url:
                return [TextContent(type="text", text=d["text"])]
        return [TextContent(type="text", text=f"No document found for {url}")]

    if name == "find_variances":
        thr = float(arguments.get("threshold_pct", 10))
        out = []
        for d in docs:
            for m in re.finditer(r"\((\d+(?:\.\d+)?)%\)", d["text"]):
                if float(m.group(1)) >= thr:
                    out.append({"section": d["section"], "company": d["company"],
                                "url": d["url"], "snippet": d["text"][:160] + "..."})
                    break
        return [TextContent(type="text", text=json.dumps(out, indent=2))]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
