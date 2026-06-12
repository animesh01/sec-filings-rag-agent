# LedgerIQ MCP Servers

LedgerIQ exposes **both** of its retrieval sources as [Model Context Protocol](https://modelcontextprotocol.io)
servers, so any MCP-capable agent (Claude Desktop, or any MCP client) can ground itself in finance documents
as a *tool it calls* — the agentic-RAG pattern.

There are two servers, one per source:

| Server | Source | Tools |
| --- | --- | --- |
| `server_sec.py` | SEC EDGAR (live) | `list_filings`, `fetch_filing`, `find_sections` |
| `server_fpa.py` | FP&A documents | `list_documents`, `get_document`, `find_variances` |

The SEC server shares its fetcher core with the offline corpus build (`scripts/edgar_fetcher.py`), so the
same code path powers both this server and `build_corpus.py --sec --live`. The FP&A server serves the bundled
synthetic corpus (`scripts/fpa_source.py`).

## Run

```bash
# SEC EDGAR tools
python mcp_server/server_sec.py

# FP&A document tools
python mcp_server/server_fpa.py
```

## Register (Claude Desktop example)

```json
{
  "mcpServers": {
    "ledgeriq-edgar": { "command": "python", "args": ["/absolute/path/to/mcp_server/server_sec.py"] },
    "ledgeriq-fpa":   { "command": "python", "args": ["/absolute/path/to/mcp_server/server_fpa.py"] }
  }
}
```

Requires the `mcp` Python package (`pip install mcp`) in the environment that runs the server.
