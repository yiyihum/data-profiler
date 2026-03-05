import json
import logging
import re
from typing import Optional

from config import Config

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, config: Config):
        self.config = config
        self._backend = None
        self._init_backend()

    def _init_backend(self):
        """Try OpenAI API first, fall back to local transformers."""
        try:
            from openai import OpenAI
            client = OpenAI(
                base_url=self.config.vllm_endpoint,
                api_key="EMPTY",
                timeout=120,
            )
            # Quick connectivity check
            client.models.list()
            self._backend = "openai"
            self._client = client
            self._model = self.config.model_name
            logger.info(f"Using OpenAI-compatible API at {self.config.vllm_endpoint}")
        except Exception as e:
            logger.warning(f"OpenAI API not available ({e}), falling back to local transformers")
            self._init_transformers()

    def _init_transformers(self):
        """Initialize local HuggingFace transformers pipeline."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = self.config.model_name
        logger.info(f"Loading {model_name} with transformers...")

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model_hf = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        self._backend = "transformers"
        logger.info(f"Model loaded on {self._model_hf.device}")

    def chat(self, prompt: str, system: str = "", temperature: float = 0.7,
             max_tokens: int = 4096) -> str:
        if self._backend == "openai":
            return self._chat_openai(prompt, system, temperature, max_tokens)
        return self._chat_transformers(prompt, system, temperature, max_tokens)

    def _chat_openai(self, prompt, system, temperature, max_tokens):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

    def _chat_transformers(self, prompt, system, temperature, max_tokens):
        import torch

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model_hf.device)

        with torch.no_grad():
            outputs = self._model_hf.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 0.01),
                do_sample=temperature > 0,
                top_p=0.95,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def chat_json(self, prompt: str, system: str = "", temperature: float = 0.3,
                  max_tokens: int = 4096) -> Optional[dict | list]:
        """Chat and parse JSON from response."""
        raw = self.chat(prompt, system=system, temperature=temperature,
                        max_tokens=max_tokens)
        return self._extract_json(raw)

    @staticmethod
    def _extract_json(text: str):
        """Extract JSON from LLM response, handling markdown code blocks."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass
        for pattern in [r"\[.*\]", r"\{.*\}"]:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    continue
        logger.warning(f"Failed to parse JSON from LLM response: {text[:200]}...")
        return None
