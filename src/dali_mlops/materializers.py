"""ZenML materializers that use the active artifact store."""

from dataclasses import asdict
import json

from zenml.materializers.base_materializer import BaseMaterializer

from .types import Checkpoint, DatasetFile, ExportedModel


class DatasetFileMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (DatasetFile,)

    def save(self, data: DatasetFile) -> None:
        with self.artifact_store.open(f"{self.uri}/dataset.json", "w") as stream:
            json.dump(asdict(data), stream)

    def load(self, data_type: type[DatasetFile]) -> DatasetFile:
        with self.artifact_store.open(f"{self.uri}/dataset.json", "r") as stream:
            return data_type(**json.load(stream))


class ExportedModelMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (ExportedModel,)

    def save(self, data: ExportedModel) -> None:
        # The payload uses a fixed artifact path so any valid model filename is supported.
        with self.artifact_store.open(f"{self.uri}/payload.bin", "wb") as stream:
            stream.write(data.data)
        with self.artifact_store.open(f"{self.uri}/export.json", "w") as stream:
            json.dump({"filename": data.filename, "media_type": data.media_type}, stream)

    def load(self, data_type: type[ExportedModel]) -> ExportedModel:
        with self.artifact_store.open(f"{self.uri}/export.json", "r") as stream:
            metadata = json.load(stream)
        with self.artifact_store.open(f"{self.uri}/payload.bin", "rb") as stream:
            return data_type(data=stream.read(), **metadata)


class CheckpointMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (Checkpoint,)

    def save(self, data: Checkpoint) -> None:
        with self.artifact_store.open(f"{self.uri}/payload.bin", "wb") as stream:
            stream.write(data.data)
        with self.artifact_store.open(f"{self.uri}/checkpoint.json", "w") as stream:
            json.dump({"filename": data.filename, "media_type": data.media_type}, stream)

    def load(self, data_type: type[Checkpoint]) -> Checkpoint:
        with self.artifact_store.open(f"{self.uri}/checkpoint.json", "r") as stream:
            metadata = json.load(stream)
        with self.artifact_store.open(f"{self.uri}/payload.bin", "rb") as stream:
            return data_type(data=stream.read(), **metadata)
