"""Turn supplements ("Highlights"): the frozen C0 contract and the pure document assembler.

Design authority: ``docs/design/turn-supplements.md`` (§2.4 row, §2.6 document, §2.7 wire,
§6 lane decomposition). This package is deliberately EMPTY at import: every lane codes
against :mod:`local_operator.supplements.contract` (vocabulary, row/event/message shapes,
capability strings) and :mod:`local_operator.supplements.document` (the one assembler),
and neither pulls anything heavier than the standard library, so importing the package
from ``harness/types.py`` or a route module costs nothing at server start.
"""
