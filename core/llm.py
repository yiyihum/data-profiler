"""LLM Client - Unified interface for LLM interactions with structured output."""

import json
import os
from typing import Any, Dict, List, Optional, Type
from dataclasses import dataclass
import re

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


@dataclass
class LLMConfig:
    """Configuration for LLM client."""
    provider: str = "openai"  # "openai" or "anthropic"
    model: str = "gpt-4o"
    api_key: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 4096

    def __post_init__(self):
        if self.api_key is None:
            if self.provider == "openai":
                self.api_key = os.getenv("OPENAI_API_KEY")
            elif self.provider == "anthropic":
                self.api_key = os.getenv("ANTHROPIC_API_KEY")


class LLMClient:
    """
    Unified LLM client with support for structured JSON output.

    Uses function calling / structured outputs to enforce JSON schemas
    and avoid parsing errors.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        """Initialize LLM client with configuration."""
        self.config = config or LLMConfig()
        self._client = None
        self._setup_client()

    def _setup_client(self) -> None:
        """Setup the appropriate client based on provider."""
        if self.config.provider == "openai":
            if not HAS_OPENAI:
                raise ImportError("openai package not installed. Run: pip install openai")
            self._client = OpenAI(api_key=self.config.api_key)

        elif self.config.provider == "anthropic":
            if not HAS_ANTHROPIC:
                raise ImportError("anthropic package not installed. Run: pip install anthropic")
            self._client = anthropic.Anthropic(api_key=self.config.api_key)
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: Optional[Dict[str, Any]] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate a response from the LLM.

        Args:
            system_prompt: System-level instructions.
            user_prompt: User query/context.
            json_schema: Optional JSON schema to enforce structured output.
            max_tokens: Override default max_tokens for this call.

        Returns:
            Dictionary with:
            - content: The generated text or parsed JSON
            - raw: Raw response text
            - success: Whether generation succeeded
            - error: Error message if failed
        """
        original_max_tokens = self.config.max_tokens
        if max_tokens is not None:
            self.config.max_tokens = max_tokens
        try:
            if self.config.provider == "openai":
                return self._generate_openai(system_prompt, user_prompt, json_schema)
            else:
                return self._generate_anthropic(system_prompt, user_prompt, json_schema)
        except Exception as e:
            return {
                "content": None,
                "raw": None,
                "success": False,
                "error": str(e)
            }
        finally:
            self.config.max_tokens = original_max_tokens

    def _generate_openai(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate using OpenAI API."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        kwargs = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
        }

        # GPT-5 family uses max_completion_tokens instead of max_tokens.
        model_name = (self.config.model or "").lower()
        if model_name.startswith("gpt-5"):
            kwargs["max_completion_tokens"] = self.config.max_tokens
        else:
            kwargs["max_tokens"] = self.config.max_tokens

        # Use structured output if schema provided
        if json_schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "strict": True,
                    "schema": json_schema
                }
            }

        response = self._client.chat.completions.create(**kwargs)
        raw_content = response.choices[0].message.content

        # Parse JSON if schema was provided
        if json_schema:
            try:
                content = json.loads(raw_content)
            except json.JSONDecodeError as e:
                return {
                    "content": None,
                    "raw": raw_content,
                    "success": False,
                    "error": f"JSON parse error: {e}"
                }
        else:
            content = raw_content

        return {
            "content": content,
            "raw": raw_content,
            "success": True,
            "error": None
        }

    def _generate_anthropic(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate using Anthropic API."""
        # For Anthropic, we embed JSON instructions in the system prompt
        if json_schema:
            schema_str = json.dumps(json_schema, indent=2)
            system_prompt = f"""{system_prompt}

IMPORTANT: You must respond with valid JSON that matches this exact schema:
```json
{schema_str}
```

Respond ONLY with the JSON object, no other text."""

        response = self._client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )

        raw_content = response.content[0].text

        # Parse JSON if schema was provided
        if json_schema:
            try:
                # Try to extract JSON from the response
                content = self._extract_json(raw_content)
            except (json.JSONDecodeError, ValueError) as e:
                return {
                    "content": None,
                    "raw": raw_content,
                    "success": False,
                    "error": f"JSON parse error: {e}"
                }
        else:
            content = raw_content

        return {
            "content": content,
            "raw": raw_content,
            "success": True,
            "error": None
        }

    def _extract_json(self, text: str) -> Any:
        """Extract JSON from text that might contain markdown code blocks."""
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON in code blocks
        patterns = [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
            r'\{[\s\S]*\}'
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    json_str = match.group(1) if '```' in pattern else match.group(0)
                    return json.loads(json_str)
                except (json.JSONDecodeError, IndexError):
                    continue

        raise ValueError("Could not extract valid JSON from response")

    def generate_code(
        self,
        task_description: str,
        context: str,
        available_vars: List[str]
    ) -> Dict[str, Any]:
        """
        Generate Python code for a specific task.

        Args:
            task_description: What the code should do.
            context: Current state context (stats, insights, etc.).
            available_vars: Variables available in the sandbox namespace.

        Returns:
            Dictionary with generated code and metadata.
        """
        system_prompt = """You are an expert Python data scientist. Generate clean, efficient Python code.

Rules:
1. Use pandas (pd), numpy (np), scipy, sklearn as needed
2. The DataFrame is already loaded as 'df'
3. Print ALL important results so they can be captured in stdout
4. Do NOT modify original data files on disk
5. Handle edge cases gracefully (check column existence, handle NaN, etc.)
6. Code should be self-contained and executable
7. Use ONLY actual column names from the DataFrame — never invent or rename columns
8. Do NOT generate any plots or visualizations"""

        user_prompt = f"""Task: {task_description}

Available variables: {', '.join(available_vars)}

Context:
{context}

Generate Python code to accomplish this task. Return ONLY the code, no explanations."""

        result = self.generate(system_prompt, user_prompt)

        if result["success"]:
            # Clean up code (remove markdown code blocks if present)
            code = result["content"]
            if isinstance(code, str):
                code = re.sub(r'^```python\s*', '', code)
                code = re.sub(r'^```\s*', '', code)
                code = re.sub(r'\s*```$', '', code)
                result["content"] = code.strip()

        return result
