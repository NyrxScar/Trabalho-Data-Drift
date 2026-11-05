"""Domain-specific exceptions shared across the pipeline modules."""


class DataValidationError(ValueError):
    """Raised when a dataset does not satisfy the expected contract."""


class ModelArtifactError(RuntimeError):
    """Raised when the persisted model is missing, invalid or out of lineage."""
