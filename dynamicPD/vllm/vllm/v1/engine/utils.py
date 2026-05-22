import os

from vllm.platforms import current_platform

def get_device_indices(device_control_env_var: str, local_dp_rank: int,
                       world_size: int):
    """
    Returns a comma-separated string of device indices for the specified
    data parallel rank.

    For example, if world_size=2 and local_dp_rank=1, and there are 4 devices,
    this will select devices 2 and 3 for local_dp_rank=1.
    """
    try:
        offset = int(os.getenv("START_DEVICE_ID", 0))
        value = ",".join(
            str(current_platform.device_id_to_physical_device_id(i+offset))
            for i in range(local_dp_rank * world_size, (local_dp_rank + 1) *
                           world_size))
    except IndexError as e:
        raise Exception(f"Error setting {device_control_env_var}: "
                        f"local range: [{local_dp_rank * world_size}, "
                        f"{(local_dp_rank + 1) * world_size}) "
                        "base value: "
                        f"\"{os.getenv(device_control_env_var)}\"") from e
    return value