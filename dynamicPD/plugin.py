import sys
import vllm

import dynamicPD.envs as envs
from dynamicPD.utils import get_compatible_vllm_version


def dynamicPD_plugin():
    if not envs.DYNAMICPD_ENABLED: #export DYNAMICPD_ENABLED=1 to enable the plugin
        print(
            "Your plugin is disabled. Set DYNAMICPD_ENABLED=1 to enable.",
            file=sys.stderr,
        )
        return

    if not envs.DYNAMICPD_SKIP_VERSION_CHECK:
        compatible_version = get_compatible_vllm_version()
        if vllm.__version__ != compatible_version:
            raise RuntimeError(
                f"Your plugin requires vllm=={compatible_version}, "
                f"but found {vllm.__version__}"
            )


    from .patches import apply_dynamicPD_patches
    apply_dynamicPD_patches()
