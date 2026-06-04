"""Project-specific exceptions."""


class MasScopeError(Exception):
    """Base exception for the mas_scope package."""


class DatasetValidationError(MasScopeError):
    """Raised when a raw dataset example cannot be normalized."""


class ConfigurationError(MasScopeError):
    """Raised for invalid experiment configuration."""


class EnvironmentDependencyError(MasScopeError):
    """Raised when an optional environment dependency is unavailable."""
