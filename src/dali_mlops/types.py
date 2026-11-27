"""Portable SDK artifact types."""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import re
from typing import Any, BinaryIO


class SDKError(RuntimeError):
    """A sanitized operational failure."""


def safe_filename(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}", name):
        raise ValueError("Use a safe filename containing letters, digits, dots, underscores or hyphens.")
    return name


@dataclass(frozen=True)
class DatasetFile:
    filename: str
    media_type: str | None
    dataset_id: str
    asset_id: str
    storage_uri: str
    agreement_id: str
    transfer_id: str
    size_bytes: int

    def open(self, mode: str = "rb") -> BinaryIO:
        """Open the staged object using the current worker's stack connection."""
        if mode != "rb":
            raise ValueError("DatasetFile supports only rb mode.")
        from .storage import open_staged

        return open_staged(self.storage_uri, self.size_bytes)


@dataclass(frozen=True)
class ExportedModel:
    filename: str
    media_type: str
    data: bytes

    def __post_init__(self) -> None:
        safe_filename(self.filename)
        if not re.fullmatch(r"[\w.+-]+/[\w.+-]+", self.media_type):
            raise ValueError("Provide a MIME media type such as application/onnx.")
        if not isinstance(self.data, bytes) or not self.data:
            raise ValueError("Provide a nonempty exported model as bytes.")

    @classmethod
    def from_file(cls, path: str | Path, media_type: str = "application/onnx") -> "ExportedModel":
        path = Path(path)
        return cls(path.name, media_type, path.read_bytes())

    @classmethod
    def from_bytes(cls, data: bytes, filename: str = "model.onnx", media_type: str = "application/onnx") -> "ExportedModel":
        return cls(filename, media_type, data)

    def open(self) -> BytesIO:
        return BytesIO(self.data)


@dataclass(frozen=True)
class Checkpoint:
    """Training state a later run can resume from: bytes in a format the author chooses."""

    filename: str
    media_type: str = "application/octet-stream"
    data: bytes = b""

    def __post_init__(self) -> None:
        safe_filename(self.filename)
        if not re.fullmatch(r"[\w.+-]+/[\w.+-]+", self.media_type):
            raise ValueError("Provide a MIME media type such as application/octet-stream.")
        if not isinstance(self.data, bytes) or not self.data:
            raise ValueError("Provide a nonempty checkpoint as bytes.")

    @classmethod
    def from_file(cls, path: str | Path, media_type: str = "application/octet-stream") -> "Checkpoint":
        path = Path(path)
        return cls(path.name, media_type, path.read_bytes())

    @classmethod
    def from_bytes(
        cls, data: bytes, filename: str = "checkpoint.bin", media_type: str = "application/octet-stream",
    ) -> "Checkpoint":
        return cls(filename, media_type, data)

    def open(self) -> BytesIO:
        return BytesIO(self.data)


@dataclass(frozen=True)
class TrainingOutput:
    """What a pipeline body returns: the exported model, final metrics and optional per-epoch metrics."""

    model: Any
    metrics: Any
    metric_history: Any = None
    checkpoint: Any = None
