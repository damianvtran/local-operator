"""Desktop import routes: sources, sessions, preview, job polling and cancel."""

from __future__ import annotations

import time
from types import SimpleNamespace

from fastapi.testclient import TestClient

from local_operator.server.app import app


def _client(monkeypatch, config):
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "token")
    app.state.config_manager = SimpleNamespace(config_dir=config)
    app.state.harness_import_service = None
    import local_operator.server.routes.desktop_imports as route

    route._JOBS.clear()
    return TestClient(app, headers={"Host": "127.0.0.1:1111"})


def test_import_routes_end_to_end(omp_store, monkeypatch):
    home, config = omp_store
    with _client(monkeypatch, config) as client:
        headers = {"Authorization": "Bearer token"}
        sources = client.get("/v1/desktop/imports/sources", headers=headers)
        assert sources.status_code == 200
        assert any(row["source"] == "omp" for row in sources.json()["result"]["sources"])

        sessions = client.get("/v1/desktop/imports/sessions?source=omp", headers=headers)
        assert sessions.status_code == 200
        rows = sessions.json()["result"]["sessions"]
        assert [row["source_id"] for row in rows] == ["omp-synthetic"]

        preview = client.post(
            "/v1/desktop/imports/preview",
            json={"sources": ["omp"], "session_ids": {"omp": ["omp-synthetic"]}},
            headers=headers,
        )
        assert preview.status_code == 200
        assert preview.json()["result"]["total_sessions"] == 1

        started = client.post(
            "/v1/desktop/imports/jobs",
            json={"sources": ["omp"], "session_ids": {"omp": ["omp-synthetic"]}},
            headers=headers,
        )
        assert started.status_code == 200
        job_id = started.json()["result"]["job_id"]
        payload = {}
        deadline = time.time() + 10
        while time.time() < deadline:
            polled = client.get(f"/v1/desktop/imports/jobs/{job_id}", headers=headers)
            assert polled.status_code == 200
            payload = polled.json()["result"]
            if payload["state"] in ("done", "failed", "cancelled"):
                break
            time.sleep(0.02)
        assert payload["state"] == "done"
        assert payload["result"]["sessions"][0]["outcome"] == "imported"

        settled = client.delete(f"/v1/desktop/imports/jobs/{job_id}", headers=headers)
        assert settled.status_code == 200
        assert settled.json()["result"]["job_id"] == job_id
