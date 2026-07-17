import sys
from types import MethodType, ModuleType
from typing import Type, Union

from vllm.logger import logger

Patchable = Union[Type, ModuleType]

#copy from https://github.com/snowflakedb/ArcticInference.git

class dynamicPDPatch:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "_plugin_patch_target"):
            raise TypeError(
                "Subclasses of dynamicPDPatch must be defined as dynamicPDPatch[Target]"
            )

    @classmethod
    def __class_getitem__(cls, target: Patchable) -> Type:
        if not isinstance(target, Patchable):
            raise TypeError(
                f"dynamicPDPatch can only target a class or module, not {type(target)}"
            )
        return type(
            f"{cls.__name__}[{target.__name__}]",
            (cls,),
            {"_plugin_patch_target": target},
        )

    @classmethod
    def apply_patch(cls):
        if cls is dynamicPDPatch or not issubclass(cls, dynamicPDPatch):
            raise TypeError("apply_patch() must be called on a dynamicPDPatch subclass")

        target = cls._plugin_patch_target

        if "_plugin_patches" not in target.__dict__:
            target._plugin_patches = {}

        for name, attr in cls.__dict__.items():
            if name in (
                "_plugin_patch_target",
                "__dict__",
                "__weakref__",
                "__module__",
                "__doc__",
                "__parameters__",
            ):
                continue

            if name in target._plugin_patches:
                patch = target._plugin_patches[name]
                raise ValueError(
                    f"{target.__name__}.{name} is already patched by {patch.__name__}"
                )

            target._plugin_patches[name] = cls

            if isinstance(attr, MethodType):
                attr = MethodType(attr.__func__, target)

            setattr(target, name, attr)
            # logger.info("Patched %s.%s", target.__name__, name)
