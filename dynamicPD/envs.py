import os
from typing import Any, Callable

environment_variables: dict[str, Callable[[], Any]] = {
    "DYNAMICPD_ENABLED": lambda: os.getenv("DYNAMICPD_ENABLED", "0") == "1",
    "DYNAMICPD_SKIP_VERSION_CHECK": lambda: os.getenv("DYNAMICPD_SKIP_VERSION_CHECK", "0") == "1",
}

def __getattr__(name: str) -> Any:
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"Environment variable '{name}' not found.")
