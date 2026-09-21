"""Exceptions raised by the graphicalization pipeline."""


class ConfigurationError(ValueError):
    """Raised when an ontology or prompt YAML file is invalid."""


class StageOutputError(ValueError):
    """Raised when a model returns an invalid structured stage response."""

    def __init__(self, stage: str, message: str, *, document_id: str | None = None,
                 chunk_id: str | None = None) -> None:
        location = ".".join(value for value in (document_id, chunk_id) if value)
        suffix = f" ({location})" if location else ""
        super().__init__(f"Invalid {stage} output{suffix}: {message}")
        self.stage = stage
        self.document_id = document_id
        self.chunk_id = chunk_id
