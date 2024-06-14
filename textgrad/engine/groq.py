try:
    from groq import Groq
except ImportError:
    raise ImportError(
        "Please install the Groq package by running `pip install groq`, and add 'GROQ_API_KEY' to your environment variables."
    )

import os
import platformdirs
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from .base import EngineLM, CachedEngine


class ChatGroq(EngineLM, CachedEngine):
    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."

    def __init__(
        self, model_string="llama3-8b-8192", system_prompt=DEFAULT_SYSTEM_PROMPT
    ):
        """
        :param model_string: The model to use with the Groq API.
        :param system_prompt: The system prompt to use as the default.
        """
        root = platformdirs.user_cache_dir("textgrad")
        cache_path = os.path.join(root, f"cache_groq_{model_string}.db")
        super().__init__(cache_path=cache_path)

        self.system_prompt = system_prompt
        if os.getenv("GROQ_API_KEY") is None:
            raise ValueError(
                "Please set the GROQ_API_KEY environment variable to use Groq models."
            )

        self.client = Groq(
            api_key=os.getenv("GROQ_API_KEY"),
        )
        self.model_string = model_string

    def generate(
        self, prompt, system_prompt=None, temperature=0.5, max_tokens=1024, top_p=1
    ):
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt

        cache_or_none = self._check_cache(sys_prompt_arg + prompt)
        if cache_or_none is not None:
            return cache_or_none

        response = self.client.chat.completions.create(
            model=self.model_string,
            messages=[
                {"role": "system", "content": sys_prompt_arg},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=None,
        )

        response_content = response.choices[0].message.content
        self._save_cache(sys_prompt_arg + prompt, response_content)
        return response_content

    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(5))
    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)
