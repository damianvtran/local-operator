"""Keep remote-compute tests package-qualified during pytest prepend imports.

Without this marker, test_api.py collides with evaluation/adapters/test_api.py
when the two files are collected together, including in the same CI shard.
"""
