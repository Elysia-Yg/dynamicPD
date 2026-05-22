from vllm.entrypoints.openai.protocol import OpenAIBaseModel

class UpdateRequest(OpenAIBaseModel):
    use_migrate: int
    use_split: int
    busy_threshold: int
    slo: int
    split_trd: int