"""Token-efficient, headless-safe web fetching for the built-in ``web_fetch``
tool and the ``read <url>`` sugar.

The package mirrors :mod:`local_operator.web_search`: ``models`` holds the
config/result contracts, ``render`` turns a fetched body into readable text,
``service`` owns the SSRF policy plus the fetch/render/spill/cache
orchestration, ``tool`` is the model-facing surface, and ``cli`` is the
``lop fetch`` configuration experience. One shared engine (``service.fetch``)
backs both doorways so the tool and the sugar cannot drift into two pipelines.
"""
