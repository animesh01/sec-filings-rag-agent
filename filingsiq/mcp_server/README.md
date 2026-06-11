# FilingsIQ EDGAR MCP Server

Exposes the SEC EDGAR filing-fetcher as **Model Context Protocol (MCP)** tools, so an
agent (Claude Desktop, or any MCP client) can retrieve real filings on demand — the
agentic-RAG pattern where retrieval is a *tool the model calls*.

## Tools

| Tool | Purpose |
|---|---|
| `list_filings(ticker, forms, limit)` | Recent 10-K / 10-Q / 8-K filings with direct document URLs |
| `fetch_filing(url, max_chars)` | Fetch a filing by URL, return cleaned text |
| `find_sections(url)` | Locate Risk Factors, MD&A, Liquidity, Revenue, Controls, Legal |

## Run

```bash
pip install mcp
python mcp_server/server.py        # stdio transport
```

## Register (Claude Desktop)

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "filingsiq-edgar": {
      "command": "python",
      "args": ["/absolute/path/to/mcp_server/server.py"]
    }
  }
}
```

The server uses the official SEC EDGAR API (no key; ~10 req/s limit respected). Set a
descriptive `USER_AGENT` in `scripts/edgar_fetcher.py` with your contact email, as SEC requests.
