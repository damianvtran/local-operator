"""Dense float32 vector matrix with exhaustive cosine search, stdlib only.

Why not an ANN library
----------------------
The skill index holds one vector per discovered skill — tens of rows, low
hundreds at the extreme — and every query must score *all* of them, because
the selector filters by an absolute similarity threshold rather than taking a
fixed top-k. Exhaustive inner product over L2-normalized vectors is therefore
not an approximation of what we want; it is exactly what we want, and it is
one line of arithmetic per row.

An approximate-nearest-neighbour library would give the identical answer at
this scale (a flat inner-product index *is* a brute-force scan) while adding
two compiled wheels and tens of megabytes to every install. On Windows that is
the difference between a fast wheel install and a toolchain hunt.

:func:`math.sumprod` (CPython 3.12+) does the dot product in C over
:class:`array.array` buffers, so the scan runs at a few microseconds per row —
measurably faster end to end than the native alternatives once their import
cost is counted.

Persistence
-----------
:meth:`VectorMatrix.serialize` / :func:`deserialize` implement a small
self-describing container: a magic string, the shape, the caller's content
hash, and a zlib-compressed little-endian float32 payload. Embedding the hash
*inside* the blob is what lets the index prove that a vector file and its
sidecar metadata describe each other rather than two interleaved writes.
"""

from __future__ import annotations

import math
import struct
import sys
import zlib
from array import array
from collections.abc import Sequence

__all__ = ["FloatArray", "VectorMatrix", "deserialize"]

#: One row of the matrix. ``array("f")`` holds C floats, so the Python-level
#: item type is ``float``; the alias keeps the annotations readable.
FloatArray = array[float]

#: Container magic. The trailing digit is the format version: a future change
#: bumps it, old files then fail the magic check and are treated as a cache
#: miss (they are only ever a cache, so there is nothing to migrate).
_MAGIC = b"LOVEC001"

#: Header after the magic: row count, dimension, content-hash byte length.
_HEADER = struct.Struct("<III")

#: ``array("f")`` is the platform's native single-precision layout. The file
#: format is fixed little-endian so a cache directory stays readable if it is
#: ever shared across architectures.
_NATIVE_IS_LITTLE = sys.byteorder == "little"


class VectorMatrix:
    """A ``rows x dim`` block of float32 vectors, one per indexed item.

    Rows are stored individually rather than as one flat buffer because every
    operation here is row-at-a-time, and a list of :class:`array.array` lets
    :func:`math.sumprod` consume each row directly with no slicing copy.

    Instances are treated as immutable once built; nothing mutates a row.
    """

    __slots__ = ("_rows", "_dim")

    def __init__(self, rows: list[FloatArray], dim: int) -> None:
        self._rows = rows
        self._dim = dim

    @classmethod
    def from_vectors(cls, vectors: Sequence[Sequence[float]]) -> VectorMatrix:
        """Build from a sequence of equal-length float sequences.

        Raises:
            ValueError: If ``vectors`` is ragged. A backend returning rows of
                differing width is a bug in the backend, and silently padding
                would corrupt every later similarity score.
        """
        if not vectors:
            return cls([], 0)
        dim = len(vectors[0])
        rows: list[FloatArray] = []
        for index, vector in enumerate(vectors):
            if len(vector) != dim:
                raise ValueError(
                    f"Ragged embedding output: row 0 has width {dim}, row {index} has "
                    f"width {len(vector)}"
                )
            rows.append(array("f", vector))
        return cls(rows, dim)

    @classmethod
    def zeros(cls, count: int, dim: int) -> VectorMatrix:
        """A ``count x dim`` all-zero matrix.

        Used as the placeholder when a backend returns unusable output: every
        similarity then scores 0, so selection degrades to "nothing matches"
        instead of raising mid-turn.
        """
        return cls([array("f", bytes(4 * dim)) for _ in range(count)], dim)

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def dim(self) -> int:
        """Row width. Zero for an empty matrix built from no vectors."""
        return self._dim

    def scores(self, query: Sequence[float]) -> list[float]:
        """Inner product of ``query`` against every row, in row order.

        Rows and queries arrive L2-normalized from the embedding backends, so
        the inner product *is* the cosine similarity. Normalization is not
        re-applied here: doing so would mask a backend that forgot to
        normalize, and the selector's absolute threshold depends on the
        backend honouring that contract.

        Raises:
            ValueError: If ``query`` width does not match :attr:`dim`.
        """
        if len(query) != self._dim:
            raise ValueError(f"Query vector dim {len(query)} != index dim {self._dim}")
        # array("f") avoids per-element float boxing inside sumprod and lets
        # the same buffer be reused for every row.
        vector = array("f", query)
        return [math.sumprod(row, vector) for row in self._rows]

    def serialize(self, content_hash: str) -> bytes:
        """Pack the matrix and ``content_hash`` into one self-describing blob."""
        payload = array("f")
        for row in self._rows:
            payload.extend(row)
        if not _NATIVE_IS_LITTLE:
            payload.byteswap()
        digest = content_hash.encode("utf-8")
        return b"".join(
            (
                _MAGIC,
                _HEADER.pack(len(self._rows), self._dim, len(digest)),
                digest,
                zlib.compress(payload.tobytes(), 6),
            )
        )


def deserialize(blob: bytes) -> tuple[VectorMatrix, str]:
    """Unpack a blob written by :meth:`VectorMatrix.serialize`.

    Returns:
        The matrix and the content hash that was packed with it.

    Raises:
        ValueError: If the blob is truncated, has the wrong magic, or its
            payload length disagrees with the declared shape.
        zlib.error: If the compressed payload is itself corrupt — a bit-flipped
            cache file reaches the decompressor before any length check can
            catch it, and zlib.error is NOT a ValueError subclass.

    Callers treat any failure as a cache miss and rebuild (see
    ``skills/index.py``, which catches broadly for this reason), so both are
    equivalent in practice. Both are documented because a future caller
    trusting only the ValueError contract would crash on a corrupt file.
    """
    header_end = len(_MAGIC) + _HEADER.size
    if len(blob) < header_end or not blob.startswith(_MAGIC):
        raise ValueError("Not a vector cache blob")
    rows, dim, hash_len = _HEADER.unpack_from(blob, len(_MAGIC))
    digest_end = header_end + hash_len
    if len(blob) < digest_end:
        raise ValueError("Truncated vector cache header")
    content_hash = blob[header_end:digest_end].decode("utf-8")

    payload = array("f")
    payload.frombytes(zlib.decompress(blob[digest_end:]))
    if not _NATIVE_IS_LITTLE:
        payload.byteswap()
    if len(payload) != rows * dim:
        raise ValueError(f"Vector cache payload holds {len(payload)} floats, expected {rows * dim}")

    matrix_rows = [payload[index * dim : (index + 1) * dim] for index in range(rows)]
    return VectorMatrix(matrix_rows, dim), content_hash
