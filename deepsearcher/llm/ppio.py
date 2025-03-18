import os
from typing import Dict, List

from deepsearcher.llm.base import BaseLLM, ChatResponse


class PPIO(BaseLLM):
    """
    PPIO language model implementation.

    This class provides an interface to interact with language models
    hosted on the PPIO infrastructure platform.

    Attributes:
        model (str): The model identifier to use on PPIO platform.
        client: The OpenAI-compatible client instance for PPIO API.
    """

    def __init__(self, model: str = "deepseek/deepseek-r1-turbo", **kwargs):
        """
        Initialize a PPIO language model client.

        Args:
            model (str, optional): The model identifier to use. Defaults to "deepseek/deepseek-r1-turbo".
            **kwargs: Additional keyword arguments to pass to the OpenAI client.
                - api_key: PPIO API key. If not provided, uses PPIO_API_KEY environment variable.
                - base_url: PPIO API base URL. If not provided, defaults to "https://api.ppinfra.com/v3/openai".
        """
        from openai import OpenAI as OpenAI_

        self.model = model
        if "api_key" in kwargs:
            api_key = kwargs.pop("api_key")
        else:
            api_key = os.getenv("PPIO_API_KEY")
        if "base_url" in kwargs:
            base_url = kwargs.pop("base_url")
        else:
            base_url = "https://api.ppinfra.com/v3/openai"
        self.client = OpenAI_(api_key=api_key, base_url=base_url, **kwargs)

    def chat(self, messages: List[Dict]) -> ChatResponse:
        """
        Send a chat message to the PPIO model and get a response.

        Args:
            messages (List[Dict]): A list of message dictionaries, typically in the format
                                  [{"role": "system", "content": "..."},
                                   {"role": "user", "content": "..."}]

        Returns:
            ChatResponse: An object containing the model's response and token usage information.
        """
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return ChatResponse(
            content=completion.choices[0].message.content,
            total_tokens=completion.usage.total_tokens,
        )
