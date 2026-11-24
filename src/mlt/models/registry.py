"""Registry of supported model families, keyed by mlx-lm `model_type`."""

from __future__ import annotations

import importlib

from .families.base import FamilyAdapter

_BY_TYPE: dict[str, FamilyAdapter] = {}
_FAMILY_MODULES = ("llama", "qwen")
_loaded = False


def register(adapter: FamilyAdapter) -> FamilyAdapter:
    for t in adapter.model_types:
        existing = _BY_TYPE.get(t)
        if existing is not None and existing.name != adapter.name:
            raise ValueError(f"model_type {t!r} already registered by {existing.name!r}")
        _BY_TYPE[t] = adapter
    return adapter


def _ensure_loaded() -> None:
    global _loaded
    if not _loaded:
        _loaded = True
        for mod in _FAMILY_MODULES:
            importlib.import_module(f"{__package__}.families.{mod}")


def get_family(model_type: str) -> FamilyAdapter:
    _ensure_loaded()
    try:
        return _BY_TYPE[model_type]
    except KeyError:
        raise ValueError(
            f"model_type {model_type!r} has no mlt family adapter. Supported: "
            f"{sorted(_BY_TYPE)}. To add one, create src/mlt/models/families/<name>.py "
            "registering a FamilyAdapter and list it in registry._FAMILY_MODULES."
        ) from None


def supported_model_types() -> list[str]:
    _ensure_loaded()
    return sorted(_BY_TYPE)
