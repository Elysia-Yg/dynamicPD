from typing import Optional, Any

from vllm.lora.request import LoRARequest
from vllm.logger import logger
from vllm.tokenizers import get_tokenizer

# benchmark使用，与性能优化无关

def get_lora_tokenizer(lora_request: LoRARequest, *args,
                       **kwargs) -> Optional[Any]:
    if lora_request is None:
        return None
    try:
        tokenizer = get_tokenizer(lora_request.lora_path, *args, **kwargs)
    except Exception as e:
        # No tokenizer was found in the LoRA folder,
        # use base model tokenizer
        logger.warning(
            "No tokenizer found in %s, using base model tokenizer instead. "
            "(Exception: %s)", lora_request.lora_path, e)
        tokenizer = None
    return tokenizer