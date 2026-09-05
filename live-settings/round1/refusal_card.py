"""U3: a refused web_fetch must not render as a completed fetch."""
import asyncio, os, sys, tempfile, pathlib
sys.path.insert(0, "/tmp/lop-live-settings")
root = pathlib.Path(tempfile.mkdtemp(prefix="ev-refusal-"))
os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = str(root)

from local_operator.config import ConfigManager
from local_operator.web_fetch import service, tool
from local_operator.web_search import tool as search_tool
from local_operator.tui.widgets.tool_card import _fetch_result_output

async def main():
    service.set_fetch_enabled(ConfigManager(root), False)
    preview, details, is_error = await tool.run_fetch("https://example.com", tool_name="web_fetch")
    print("web_fetch refusal:")
    print(f"  text    = {preview}")
    print(f"  details = {details!r}")
    print(f"  is_error= {is_error}")
    rows = _fetch_result_output(details if details else None)
    print(f"  CARD ROWS the user sees above the sentence: {rows}")
    print(f"  -> contains a 'Fetched:' claim? {any('Fetched' in r for r in rows)}")

    from local_operator.web_search.service import set_search_enabled
    set_search_enabled(ConfigManager(root), False)
    r = await search_tool.execute_web_search("t1", {"query": "x"}, None, None, None)
    print("\nweb_search refusal (the shape web_fetch was brought into line with):")
    print(f"  text    = {r.text}")
    print(f"  details = {r.details!r}")

asyncio.run(main())
