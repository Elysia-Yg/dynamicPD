import re
from importlib.metadata import requires


def get_compatible_vllm_version():
    reqs = requires("dynamicPD")
    for req in reqs:
        match = re.match(r'vllm==(.*); extra == "vllm"', req)
        if match is not None:
            return match.group(1)
    return None
