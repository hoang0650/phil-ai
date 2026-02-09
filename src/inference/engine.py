import os
from abc import ABC, abstractmethod

class InferenceEngine(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int = 200, temperature: float = 0.7) -> str:
        pass

class TransformersEngine(InferenceEngine):
    def __init__(self, model_path: str):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16
        )

    def generate(self, prompt: str, max_new_tokens: int = 200, temperature: float = 0.7) -> str:
        import torch
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            do_sample=True, 
            temperature=temperature
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

class VLLMEngine(InferenceEngine):
    def __init__(self, model_path: str, gpu_memory_utilization: float = 0.9):
        from vllm import LLM, SamplingParams
        self.llm = LLM(model=model_path, gpu_memory_utilization=gpu_memory_utilization)
        self.sampling_params = SamplingParams(temperature=0.7, max_tokens=200)

    def generate(self, prompt: str, max_new_tokens: int = 200, temperature: float = 0.7) -> str:
        from vllm import SamplingParams
        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_new_tokens)
        outputs = self.llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text

class TGIEngine(InferenceEngine):
    def __init__(self, base_url: str = "http://localhost:8080"):
        from text_generation import Client
        self.client = Client(base_url)

    def generate(self, prompt: str, max_new_tokens: int = 200, temperature: float = 0.7) -> str:
        response = self.client.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
        return response.generated_text

class LlamaCppEngine(InferenceEngine):
    def __init__(self, model_path: str, n_ctx: int = 2048):
        from llama_cpp import Llama
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=-1)

    def generate(self, prompt: str, max_new_tokens: int = 200, temperature: float = 0.7) -> str:
        output = self.llm(
            prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            stop=["<|eot_id|>", "User:"],
            echo=False
        )
        return output["choices"][0]["text"]

def get_engine(engine_type: str, **kwargs) -> InferenceEngine:
    if engine_type == "transformers":
        return TransformersEngine(kwargs.get("model_path"))
    elif engine_type == "vllm":
        return VLLMEngine(kwargs.get("model_path"))
    elif engine_type == "tgi":
        return TGIEngine(kwargs.get("base_url", "http://localhost:8080"))
    elif engine_type == "llama.cpp":
        return LlamaCppEngine(kwargs.get("model_path"))
    else:
        raise ValueError(f"Unsupported engine type: {engine_type}")