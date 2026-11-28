"""Static server for the built benchmark pages that also keeps suite results.

    python3 suite-server.py <port> <dist-dir> <results-dir>

`suite.html` POSTs its gathered JSON lines to `/save?name=<name>`; they land in
`<results-dir>/<name>.jsonl`. Plain `python -m http.server` cannot accept the
POST, and copying numbers by hand out of a browser read is how a benchmark table
drifts from its raw output. Loopback only; the name is reduced to [A-Za-z0-9._-].
"""
import http.server
import pathlib
import re
import sys
import urllib.parse

port, root, out = int(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=root, **kwargs)

    def do_POST(self):
        query = urllib.parse.urlparse(self.path)
        name = re.sub(r"[^A-Za-z0-9._-]", "_", urllib.parse.parse_qs(query.query).get("name", ["suite"])[0])
        body = self.rfile.read(int(self.headers.get("content-length", 0)))
        (out / f"{name}.jsonl").write_bytes(body)
        self.send_response(204)
        self.end_headers()


http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
