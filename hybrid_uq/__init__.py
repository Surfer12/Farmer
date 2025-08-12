"""Hybrid AI-Physics UQ minimal stubs.

This package exposes PyTorch-based components in `core.py`. To avoid
import-time dependencies when running lightweight utilities (e.g.,
`python -m hybrid_uq.example_numeric_check`), we do not import `core`
from here. Import from `hybrid_uq.core` explicitly when needed.
"""

__all__ = []