"""Collect public release metadata only, not distributions or credentials.

An npm integrity or PyPI artifact hash pins an available distribution, not its
transitive installation closure or the deleted historical installation.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
SOURCES = {
    'claude-code': 'https://registry.npmjs.org/@anthropic-ai%2fclaude-code/2.1.274',
    'codex-best': 'https://registry.npmjs.org/@openai%2fcodex/0.147.0',
    'omp': 'https://registry.npmjs.org/@oh-my-pi%2fpi-coding-agent/18.11.2',
    'dsh': 'https://registry.npmjs.org/@deepseek-ai%2fdsh/0.2.0-rc.2',
    'harbor': 'https://pypi.org/pypi/harbor/0.25.0/json',
    'opencode': 'https://api.github.com/repos/anomalyco/opencode/releases/tags/v2.0.27',
    'claude-linux-x64': 'https://registry.npmjs.org/@anthropic-ai%2fclaude-code-linux-x64/2.1.274',
    'codex-linux-x64': 'https://registry.npmjs.org/@openai%2fcodex/0.147.0-linux-x64',
    'httpx-proposed': 'https://pypi.org/pypi/httpx/json',
    'starlette-proposed': 'https://pypi.org/pypi/starlette/json',
    'uvicorn-proposed': 'https://pypi.org/pypi/uvicorn/json',
    'cryptography-proposed': 'https://pypi.org/pypi/cryptography/json',
    'hermes-pyproject': 'https://raw.githubusercontent.com/NousResearch/hermes-agent/f0d94882e89ab8d4098b5bb2f166adfb9ee0a0ab/pyproject.toml',
    'hermes-uv-lock': 'https://raw.githubusercontent.com/NousResearch/hermes-agent/f0d94882e89ab8d4098b5bb2f166adfb9ee0a0ab/uv.lock',
    'lop-pyproject': 'https://raw.githubusercontent.com/damianvtran/local-operator/500883ef56ca504325ffd80c5b90bf2435d00e3f/pyproject.toml',
    'lop-uv-lock': 'https://raw.githubusercontent.com/damianvtran/local-operator/500883ef56ca504325ffd80c5b90bf2435d00e3f/uv.lock',
    'opencode-no-v': 'https://api.github.com/repos/anomalyco/opencode/releases/tags/2.0.27',
    'opencode-npm': 'https://registry.npmjs.org/opencode-ai/2.0.27',
    'sanitize-file-history': 'https://api.github.com/repos/harbor-framework/terminal-bench-2-1/commits?path=tasks/sanitize-git-repo/tests/test_outputs.py&per_page=5',
    'task-license': 'https://raw.githubusercontent.com/harbor-framework/terminal-bench-2-1/7131e4375048a0e408a8fb404b5f499d726b695b/LICENSE',
    'opencode-npm-correct': 'https://registry.npmjs.org/@opencode%2fcli/2.0.27',
    'opencode-linux-x64': 'https://registry.npmjs.org/@opencode%2fcli-linux-x64/2.0.27',
    'opencode-cli-package': 'https://raw.githubusercontent.com/anomalyco/opencode/1f4d07a4a519185955ae8ab2ea3607805509e386/packages/cli/package.json',
    'opencode-install': 'https://raw.githubusercontent.com/anomalyco/opencode/1f4d07a4a519185955ae8ab2ea3607805509e386/install',
}


def metadata_sources():
    sources = dict(SOURCES)
    bindings = HERE / 'image-bindings.json'
    if bindings.exists():
        for row in json.loads(bindings.read_text())['observations']:
            sources['image-' + row['task']] = row['source_metadata_url']
    closure = HERE / 'source-closure-findings.json'
    if closure.exists():
        # Preserve exact source/metadata URLs for replay without invoking any
        # package manager or downloading the platform binaries they describe.
        sources.update(json.loads(closure.read_text())['collection_sources'])
    return sources


def main():
    sources = metadata_sources()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', nargs='+', choices=sorted(sources), help='Collect only selected sources; preserve earlier receipts')
    args = parser.parse_args()
    rows = []
    for name in args.only or sources:
        url = sources[name]
        row = {'name': name, 'url': url, 'recorded_at': datetime.datetime.now(datetime.UTC).isoformat()}
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read(4_000_001)
            if len(data) > 4_000_000:
                raise ValueError('metadata exceeds collection bound')
            digest = hashlib.sha256(data).hexdigest()
            try:
                json.loads(data)
                suffix = '.json'
            except ValueError:
                suffix = '.toml' if name.endswith('pyproject') else '.lock' if name.endswith('uv-lock') else '.txt'
            dest = HERE / 'metadata' / (name + '-' + digest + suffix)
            dest.parent.mkdir(exist_ok=True)
            if not dest.exists():
                with dest.open('xb') as out:
                    out.write(data)
            row.update(status='available', sha256=digest, file=str(dest.relative_to(HERE)))
        except (OSError, urllib.error.URLError, ValueError) as error:
            row.update(status='unavailable', error=str(error), sha256=None)
        rows.append(row)
    dest = HERE / ('metadata-receipt-' + datetime.datetime.now(datetime.UTC).strftime('%Y%m%dT%H%M%S%fZ') + '.jsonl')
    with dest.open('x') as out:
        for row in rows:
            out.write(json.dumps(row, sort_keys=True) + '\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
