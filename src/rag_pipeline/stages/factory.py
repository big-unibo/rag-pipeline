"""Common factory for automatically discovered stage classes."""

import importlib
import inspect
import pkgutil
from typing import Any


def find_class(class_name: str) -> type:
    """Find a concrete stage class by its class name."""
    import rag_pipeline.stages as stages

    matches = []
    for module_info in pkgutil.walk_packages(stages.__path__, stages.__name__ + "."):
        if module_info.ispkg:
            continue
        module = importlib.import_module(module_info.name)
        cls = getattr(module, class_name, None)
        if inspect.isclass(cls) and cls.__module__ == module.__name__:
            matches.append(cls)

    if not matches:
        raise ValueError(f"Stage class not found: {class_name}")
    if len(matches) > 1:
        locations = ", ".join(f"{cls.__module__}.{cls.__name__}" for cls in matches)
        raise ValueError(f"Stage class name is ambiguous: {class_name} ({locations})")
    return matches[0]


def build_stage(config: dict[str, Any], default_class: str, *args: Any, **kwargs: Any) -> Any:
    """Instantiate the configured class discovered under ``rag_pipeline.stages``.

    ``default_class`` keeps a stage usable when a manifest omits the strategy,
    while an explicit ``strategy`` value lets experiments select any compatible
    implementation without editing a central class list.
    """
    class_name = config.get("strategy", default_class)
    if not isinstance(class_name, str) or not class_name:
        raise ValueError(f"Stage class must be a class name, got: {class_name!r}")
    return find_class(class_name)(config, *args, **kwargs)
