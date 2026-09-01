"""Episode execution for the evaluation stack.

Importing this package is deliberately inert. The evaluation subsystem is
never loaded by an ordinary session, and ``tests/unit/evaluation`` asserts
that startup imports pull in nothing under ``local_operator.evaluation``, so
this module must not re-export the runner or its contracts.
"""
