import asyncio, sys
sys.path.insert(0, "/tmp/lop-live-settings")
from local_operator.harness.types import ToolContext
from local_operator.tools.builtin import execute_read
from local_operator.web_search.tool import execute_web_search


async def m():
    r = await execute_read("t", {"path": "https://example.com"}, None, None, ToolContext(cwd="/tmp/iso-live/ws"))
    print("read <url>   :", "error" if r.is_error else "ok   ", r.text[:100].replace("\n", " "))
    s = await execute_web_search("t", {"query": "x", "provider": "duckduckgo"}, None, None, None)
    print("web_search   :", "error" if s.is_error else "ok   ", s.text[:100].replace("\n", " "))


asyncio.run(m())
