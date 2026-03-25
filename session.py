# session.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar
import numpy as np


@dataclass
class ModelEntry:
    key: str
    name: str | None = None
    facies: Any = None
    grid: Any = None
    reservoir_facies: set = field(default_factory=set)

    view_mode: str | None = None
    study: str | None = None
    path: str | None = None

    extra: dict[str, Any] = field(default_factory=dict)

    _KNOWN_FIELDS: ClassVar[set[str]] = {
        "key", "name", "facies", "grid", "reservoir_facies",
        "view_mode", "study", "path", "extra"
    }

    @classmethod
    def from_legacy_dict(cls, key: str, data: dict[str, Any]) -> "ModelEntry":
        data = dict(data or {})

        known = {}
        extra = {}

        for k, v in data.items():
            if k in cls._KNOWN_FIELDS:
                known[k] = v
            else:
                extra[k] = v

        rf = known.get("reservoir_facies", set()) or set()
        if not isinstance(rf, set):
            try:
                rf = set(rf)
            except Exception:
                rf = {rf}

        return cls(
            key=str(key),
            name=known.get("name"),
            facies=known.get("facies"),
            grid=known.get("grid"),
            reservoir_facies=rf,
            view_mode=known.get("view_mode"),
            study=known.get("study"),
            path=known.get("path"),
            extra=extra,
        )

    def to_legacy_dict(self) -> dict[str, Any]:
        out = {
            "name": self.name,
            "facies": self.facies,
            "grid": self.grid,
            "reservoir_facies": self.reservoir_facies,
        }

        if self.view_mode is not None:
            out["view_mode"] = self.view_mode
        if self.study is not None:
            out["study"] = self.study
        if self.path is not None:
            out["path"] = self.path

        out.update(self.extra)
        return out

    def __getitem__(self, item):
        if item in self._KNOWN_FIELDS:
            return getattr(self, item)
        return self.extra[item]

    def __setitem__(self, item, value):
        if item in self._KNOWN_FIELDS:
            if item == "reservoir_facies" and not isinstance(value, set):
                try:
                    value = set(value)
                except Exception:
                    value = {value}
            setattr(self, item, value)
        else:
            self.extra[item] = value

    def get(self, item, default=None):
        try:
            return self[item]
        except Exception:
            return default

    def __contains__(self, item):
        return (item in self._KNOWN_FIELDS) or (item in self.extra)

    def keys(self):
        return self.to_legacy_dict().keys()

    def items(self):
        return self.to_legacy_dict().items()

    def values(self):
        return self.to_legacy_dict().values()


class ModelStore(dict):

    def __setitem__(self, key, value):
        key = str(key)

        if isinstance(value, ModelEntry):
            entry = value
            if entry.key != key:
                entry.key = key
        elif isinstance(value, dict):
            entry = ModelEntry.from_legacy_dict(key, value)
        else:
            raise TypeError(
                f"ModelStore aceita apenas dict ou ModelEntry. Recebido: {type(value)}"
            )

        super().__setitem__(key, entry)

    def update(self, other=None, **kwargs):
        if other:
            if hasattr(other, "items"):
                iterable = other.items()
            else:
                iterable = other
            for k, v in iterable:
                self[k] = v
        for k, v in kwargs.items():
            self[k] = v

    def as_legacy_dict(self) -> dict[str, dict]:
        return {k: v.to_legacy_dict() for k, v in self.items()}


@dataclass
class AppSession:
    models: ModelStore = field(default_factory=ModelStore)
    wells: dict[str, Any] = field(default_factory=dict)

    facies_reference: list = field(default_factory=list)
    facies_colors_dict: dict = field(default_factory=dict)
    facies_colors: dict = field(default_factory=dict)
    markers_db: dict = field(default_factory=dict)

    color_reference_path: str | None = None
    markers_path: str | None = None

    facies_grouping_map: dict = field(default_factory=dict)
    use_facies_grouping: bool = False
    _fg_src: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    _fg_dst: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))

    state_reservoir_raw: set = field(default_factory=set)
    state_reservoir_grouped: set = field(default_factory=set)

    cached_metrics: dict = field(default_factory=dict)
    compare_states: dict = field(default_factory=dict)

    base_model_key: str | None = None
    active_model_key: str | None = None

    def add_model(self, key: str, model: dict | ModelEntry):
        self.models[key] = model

    def get_model(self, key: str, default=None):
        return self.models.get(str(key), default)

    def remove_model(self, key: str):
        key = str(key)
        self.models.pop(key, None)
        self.cached_metrics.pop(key, None)
        self.compare_states.pop(key, None)

        if self.base_model_key == key:
            self.base_model_key = None
        if self.active_model_key == key:
            self.active_model_key = None