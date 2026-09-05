"""Offline A/B probe of bash waiting/capture and web connection ownership.

Run with each checkout's OWN interpreter and --repo pointing at that checkout.
The script may live in either tree; it prints the imported source path so a
shared editable install cannot silently invalidate the comparison. Wall times
are observations, never pass/fail ceilings. TCP connection count and identical
response assertions establish the network optimization independently of timing.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import inspect
import json
import os
import shlex
import statistics
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Any

parser = argparse.ArgumentParser()
parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument("--output", type=Path)
args = parser.parse_args()
sys.path.insert(0, str(args.repo.resolve()))

from local_operator.harness.types import AbortSignal, ToolContext  # noqa: E402
from local_operator.tools import builtin  # noqa: E402
from local_operator.web_fetch.models import WebFetchSettings  # noqa: E402
from local_operator.web_fetch.service import WebFetchService  # noqa: E402


async def measure() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="lo-tool-io-bench-") as directory:
        os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = directory
        context = ToolContext(cwd=directory, session_id="tool-io-bench")
        short_ms = []
        for index in range(12):
            started = time.perf_counter()
            result = await builtin.execute_bash(
                str(index), {"command": "printf answer"}, AbortSignal(), None, context
            )
            short_ms.append((time.perf_counter() - started) * 1000)
            assert not result.is_error and "answer" in result.text

        memory_runs = []
        for mib in (1, 32):
            gc.collect()
            tracemalloc.start()
            program = (
                f'import sys; sys.stdout.write("HEAD\\n" + "x" * {mib * 1024 * 1024}'
                ' + "\\nFINAL")'
            )
            command = f"{shlex.quote(sys.executable)} -c {shlex.quote(program)}"
            started = time.perf_counter()
            result = await builtin.execute_bash(
                "capture", {"command": command}, AbortSignal(), None, context
            )
            elapsed = time.perf_counter() - started
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            assert not result.is_error and "HEAD" in result.text and "FINAL" in result.text
            memory_runs.append(
                {"produced_mib": mib, "peak_python_mib": peak / 1024**2, "wall_s": elapsed}
            )

        connections = 0
        requests = 0
        writers: set[asyncio.StreamWriter] = set()

        async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
            nonlocal connections, requests
            connections += 1
            writers.add(writer)
            try:
                while True:
                    await reader.readuntil(b"\r\n\r\n")
                    requests += 1
                    writer.write(
                        b"HTTP/1.1 200 OK\r\nContent-Length: 6\r\n"
                        b"Content-Type: text/plain\r\n\r\nanswer"
                    )
                    await writer.drain()
            except (asyncio.IncompleteReadError, ConnectionError):
                pass
            finally:
                writers.discard(writer)
                writer.close()
                await writer.wait_closed()

        server = await asyncio.start_server(serve, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        owner = None
        if "io" in inspect.signature(WebFetchService).parameters:
            from local_operator.web_search.io import WebReadIO

            owner = WebReadIO()
        started = time.perf_counter()
        try:
            for index in range(12):
                kwargs: dict[str, Any] = {"io": owner} if owner is not None else {}
                fetcher = WebFetchService(
                    WebFetchSettings(allow_private=True, enrich=False), **kwargs
                )
                result = await fetcher.fetch(f"http://127.0.0.1:{port}/{index}")
                assert result.content.strip() == "answer"
        finally:
            if owner is not None:
                await owner.aclose()
            for writer in list(writers):
                writer.close()
            server.close()
            await server.wait_closed()
        web_ms = (time.perf_counter() - started) * 1000

        return {
            "source": builtin.__file__,
            "python": sys.executable,
            "short_bash_ms": short_ms,
            "short_bash_median_ms": statistics.median(short_ms),
            "capture": memory_runs,
            "web": {"requests": requests, "tcp_connections": connections, "wall_ms": web_ms},
        }


result = asyncio.run(measure())
text = json.dumps(result, indent=2)
if args.output:
    args.output.write_text(text + "\n")
print(text)
