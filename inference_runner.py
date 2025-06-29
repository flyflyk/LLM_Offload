
import torch
import time
import logging
from typing import List, Tuple, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator

logger = logging.getLogger(__name__)

class InferenceRunner:
    def __init__(self, model_name: str, p_type: torch.dtype = torch.float16, use_accelerate: bool = False, offload_dir: str = None):
        self.model_name = model_name
        self.p_type = p_type
        self.use_accelerate = use_accelerate
        self.accelerator = None
        
        if self.use_accelerate:
            self.accelerator = Accelerator()
            self.device = self.accelerator.device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading model '{model_name}'...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        model_kwargs = {"torch_dtype": self.p_type}
        if use_accelerate and offload_dir:
             # When using accelerate with offloading, device_map is handled by accelerate
            pass
        elif not use_accelerate:
             model_kwargs["device_map"] = "auto"


        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.use_accelerate:
            self.model = self.accelerator.prepare(self.model)
        
        logger.info(f"Model '{model_name}' loaded on device: {self.device}")


    def run_inference(self, prompts: List[str], max_new_tokens: int = 50) -> List[str]:
        if not prompts or not all(prompts):
            raise ValueError("Prompt list cannot be empty or contain empty prompts.")

        logger.info(f"Running inference for batch of {len(prompts)} prompts.")

        inputs = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            generation_output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=True,
                temperature=1.0,
                top_k=50,
            )
        
        input_token_len = inputs.input_ids.shape[1]
        generated_tokens = generation_output[:, input_token_len:]
        generated_texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        return generated_texts
