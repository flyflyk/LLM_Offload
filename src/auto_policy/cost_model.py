import dataclasses

from FlexLLMGen.flexllmgen.opt_config import get_opt_config, OptConfig

@dataclasses.dataclass
class ModelInfo:
    name: str
    config: OptConfig

def get_model_info(model_name: str) -> ModelInfo:
    config = get_opt_config(model_name)
    return ModelInfo(name=model_name, config=config)
