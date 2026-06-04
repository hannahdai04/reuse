"""Project registry for MAS backbones, dataset builders, and memory providers."""

from __future__ import annotations


class MasRegistry:
    """Registry for in-house MAS backbones."""

    def __init__(self) -> None:
        self._mas_classes: dict[str, type] = {}
        self._dataset_builder_classes: dict[str, type] = {}
        self._memory_provider_classes: dict[str, type] = {}
        self._defaults_loaded = False
        self._dataset_defaults_loaded = False
        self._memory_defaults_loaded = False

    def register_mas(self, name: str):
        def decorator(cls):
            self._mas_classes[name] = cls
            return cls

        return decorator

    def get_mas(self, name: str):
        self._ensure_default_mas_registered()
        if name not in self._mas_classes:
            known = ", ".join(sorted(self._mas_classes)) or "<empty>"
            raise KeyError(f"Unknown MAS '{name}'. Known MAS backbones: {known}")
        return self._mas_classes[name]

    def names(self) -> list[str]:
        self._ensure_default_mas_registered()
        return sorted(self._mas_classes)

    def register_dataset_builder(self, name: str):
        def decorator(cls):
            self._dataset_builder_classes[name] = cls
            return cls

        return decorator

    def get_dataset_builder(self, name: str):
        self._ensure_default_dataset_builders_registered()
        if name not in self._dataset_builder_classes:
            known = ", ".join(sorted(self._dataset_builder_classes)) or "<empty>"
            raise KeyError(f"Unknown dataset builder '{name}'. Known dataset builders: {known}")
        return self._dataset_builder_classes[name]

    def dataset_builder_names(self) -> list[str]:
        self._ensure_default_dataset_builders_registered()
        return sorted(self._dataset_builder_classes)

    def register_memory_provider(self, name: str):
        def decorator(cls):
            self._memory_provider_classes[name] = cls
            return cls

        return decorator

    def get_memory_provider(self, name: str | None):
        provider_name = name or "null"
        self._ensure_default_memory_providers_registered()
        if provider_name not in self._memory_provider_classes:
            known = ", ".join(sorted(self._memory_provider_classes)) or "<empty>"
            raise KeyError(f"Unknown memory provider '{provider_name}'. Known memory providers: {known}")
        return self._memory_provider_classes[provider_name]

    def memory_provider_names(self) -> list[str]:
        self._ensure_default_memory_providers_registered()
        return sorted(self._memory_provider_classes)

    def _ensure_default_mas_registered(self) -> None:
        if self._defaults_loaded:
            return
        self._defaults_loaded = True
        import mas_scope.mas.autogen_style  # noqa: F401
        import mas_scope.mas.camel_style  # noqa: F401
        import mas_scope.mas.dylan_style  # noqa: F401
        import mas_scope.mas.macnet_style  # noqa: F401

    def _ensure_default_dataset_builders_registered(self) -> None:
        if self._dataset_defaults_loaded:
            return
        self._dataset_defaults_loaded = True
        import mas_scope.datasets.builders  # noqa: F401

    def _ensure_default_memory_providers_registered(self) -> None:
        if self._memory_defaults_loaded:
            return
        self._memory_defaults_loaded = True
        import mas_scope.memory.llm_scope_memory  # noqa: F401
        import mas_scope.memory.null_memory  # noqa: F401


registry = MasRegistry()
