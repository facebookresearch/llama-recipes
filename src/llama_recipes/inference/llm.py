# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import logging

import time
from abc import ABC, abstractmethod
from typing import Callable

import openai
from langchain_together import Together

from typing_extensions import override


NUM_LLM_RETRIES = 10

MAX_TOKENS = 1000

LOG: logging.Logger = logging.getLogger(__name__)


class LLM(ABC):
    def __init__(self, model: str, api_key: str | None = None) -> None:
        if model not in self.valid_models():
            LOG.warning(
                f"{model} is not in the valid model list for {type(self).__name__}. Valid models are: {', '.join(self.valid_models())}."
            )
        self.model: str = model
        self.api_key: str | None = api_key

    @abstractmethod
    def query(self, prompt: str) -> str:
        """
        Abstract method to query an LLM with a given prompt and return the response.

        Args:
            prompt (str): The prompt to send to the LLM.

        Returns:
            str: The response from the LLM.
        """
        pass

    def query_with_system_prompt(self, system_prompt: str, prompt: str) -> str:
        """
        Abstract method to query an LLM with a given prompt and system prompt and return the response.

        Args:
            system prompt (str): The system prompt to send to the LLM.
            prompt (str): The prompt to send to the LLM.

        Returns:
            str: The response from the LLM.
        """
        return self.query(system_prompt + "\n" + prompt)

    def _query_with_retries(
        self,
        func: Callable[..., str],
        *args: str,
        retries: int = NUM_LLM_RETRIES,
        backoff_factor: float = 0.5,
    ) -> str:
        last_exception = None
        for retry in range(retries):
            try:
                return func(*args)
            except Exception as exception:
                last_exception = exception
                sleep_time = backoff_factor * (2**retry)
                time.sleep(sleep_time)
                LOG.debug(
                    f"LLM Query failed with error: {exception}. Sleeping for {sleep_time} seconds..."
                )
        raise RuntimeError(
            f"Unable to query LLM after {retries} retries: {last_exception}"
        )

    def query_with_retries(self, prompt: str) -> str:
        return self._query_with_retries(self.query, prompt)

    def query_with_system_prompt_with_retries(
        self, system_prompt: str, prompt: str
    ) -> str:
        return self._query_with_retries(
            self.query_with_system_prompt, system_prompt, prompt
        )

    def valid_models(self) -> list[str]:
        """List of valid model parameters, e.g. 'gpt-3.5-turbo' for GPT"""
        return []

class ANYSCALE(LLM):
    """Accessing ANYSCALE"""

    def __init__(self, model: str, api_key: str) -> None:
        super().__init__(model, api_key)
        self.client = openai.OpenAI(base_url="https://api.endpoints.anyscale.com/v1", api_key=api_key)  # noqa

    @override
    def query(self, prompt: str) -> str:
        # Best-level effort to suppress openai log-spew.
        # Likely not work well in multi-threaded environment.
        level = logging.getLogger().level
        logging.getLogger().setLevel(logging.WARNING)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_tokens=MAX_TOKENS,
        )
        logging.getLogger().setLevel(level)
        return response.choices[0].message.content

    @override
    def valid_models(self) -> list[str]:
        return [
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-2-13b-chat-hf",
            "meta-llama/Llama-2-70b-chat-hf",
            "codellama/CodeLlama-34b-Instruct-hf",
            "mistralai/Mistral-7B-Instruct-v0.1",
            "HuggingFaceH4/zephyr-7b-beta",
        ]


class TOGETHER(LLM):
    """Accessing TOGETHER"""

    @override
    def query(self, prompt: str) -> str:
        llm = Together(
            model=self.model,
            temperature=0.75,
            top_p=1,
            max_tokens=MAX_TOKENS,
            together_api_key=self.api_key,
        )
        response = llm(prompt)
        return "".join(response)

    @override
    def valid_models(self) -> list[str]:
        return [
            "mistralai/Mistral-7B-v0.1",
            "lmsys/vicuna-7b-v1.5",
            "togethercomputer/CodeLlama-7b",
            "togethercomputer/CodeLlama-7b-Python",
            "togethercomputer/CodeLlama-7b-Instruct",
            "togethercomputer/CodeLlama-13b",
            "togethercomputer/CodeLlama-13b-Python",
            "togethercomputer/CodeLlama-13b-Instruct",
            "togethercomputer/falcon-40b",
            "togethercomputer/llama-2-7b",
            "togethercomputer/llama-2-7b-chat",
            "togethercomputer/llama-2-13b",
            "togethercomputer/llama-2-13b-chat",
            "togethercomputer/llama-2-70b",
            "togethercomputer/llama-2-70b-chat",
        ]
