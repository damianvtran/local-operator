"""Probe the installed legacy SDK's JWT path, without pytest or real secrets.

Run with the isolated benchmark interpreter, e.g. ``python -I -B <this file>``.
The fixed synthetic input makes old/new installations directly comparable. No
network request, credential-store lookup, or token output is performed.
"""

from __future__ import annotations

import hashlib
import json
from importlib import import_module
from importlib.metadata import version
from unittest.mock import patch

EXPECTED_DIGEST = "b8a059da3a7ff0ab0a4267332a61e69a5e98cce8e16d96adbbfa9e37503f1f43"


def probe() -> dict[str, str | bool]:
    import jwt

    # This SDK belongs to the paid OSWorld extra, not the normal development
    # or CI dependency set. Resolve it only when this explicit probe runs.
    generate_token = import_module("zhipuai.core._jwt_token").generate_token

    secret = "synthetic-" + "x" * 48
    generate_token.cache_clear()
    try:
        with patch("zhipuai.core._jwt_token.time.time", return_value=1800000000.0):
            token = generate_token("synthetic-id." + secret)
        digest = hashlib.sha256(token.encode()).hexdigest()
        if digest != EXPECTED_DIGEST:
            raise RuntimeError("SDK signing bytes differ from the retained control")
        claims = jwt.decode(token, secret, algorithms=["HS256"], options={"verify_exp": False})
        if claims != {
            "api_key": "synthetic-id",
            "exp": 1800000210000,
            "timestamp": 1800000000000,
        }:
            raise RuntimeError("SDK signing claims differ from the synthetic input")
        if jwt.get_unverified_header(token) != {
            "alg": "HS256",
            "sign_type": "SIGN",
            "typ": "JWT",
        }:
            raise RuntimeError("SDK signing header differs from the control")
        return {
            "pyjwt": version("PyJWT"),
            "zhipuai": version("zhipuai"),
            "token_sha256": digest,
            "claims_verified": True,
            "header_verified": True,
        }
    finally:
        generate_token.cache_clear()


if __name__ == "__main__":
    print(json.dumps(probe(), sort_keys=True))
