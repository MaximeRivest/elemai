"""LLM client abstraction for multiple providers."""

import json
import os
from typing import Any, Dict, List, Optional
from .config import Config, resolve_model


class LLMClient:
    """Abstract base for LLM clients."""

    def __init__(self, config: Config):
        self.config = config

    def complete(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Send messages to LLM and get response.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional parameters

        Returns:
            Response text
        """
        raise NotImplementedError


class AnthropicClient(LLMClient):
    """Client for Anthropic's Claude API."""

    def __init__(self, config: Config):
        super().__init__(config)
        try:
            from anthropic import Anthropic
            api_key = config.api_key or os.environ.get('ANTHROPIC_API_KEY')
            self.client = Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")

    def complete(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Send messages to Claude."""
        # Separate system message from conversation
        system = None
        conversation = []

        for msg in messages:
            if msg['role'] == 'system':
                system = msg['content']
            else:
                conversation.append(msg)

        # Prepare parameters
        params = {
            'model': resolve_model(kwargs.get('model', self.config.model)),
            'messages': conversation,
            'temperature': kwargs.get('temperature', self.config.temperature),
            'max_tokens': kwargs.get('max_tokens', self.config.max_tokens or 4096),
        }

        if system:
            params['system'] = system

        # Call API
        response = self.client.messages.create(**params)

        # Extract text content
        text = ''
        for block in response.content:
            if hasattr(block, 'text'):
                text += block.text

        return {
            'text': text,
            'raw_response': response,
            'usage': {
                'input_tokens': response.usage.input_tokens,
                'output_tokens': response.usage.output_tokens,
            }
        }


class OpenAIClient(LLMClient):
    """Client for OpenAI's API."""

    def __init__(self, config: Config):
        super().__init__(config)
        try:
            from openai import OpenAI
            api_key = config.api_key or os.environ.get('OPENAI_API_KEY')
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

    def complete(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Send messages to OpenAI."""
        params = {
            'model': resolve_model(kwargs.get('model', self.config.model)),
            'messages': messages,
            'temperature': kwargs.get('temperature', self.config.temperature),
        }

        if self.config.max_tokens or kwargs.get('max_tokens'):
            params['max_tokens'] = kwargs.get('max_tokens', self.config.max_tokens)

        response = self.client.chat.completions.create(**params)

        return {
            'text': response.choices[0].message.content,
            'raw_response': response,
            'usage': {
                'input_tokens': response.usage.prompt_tokens,
                'output_tokens': response.usage.completion_tokens,
            }
        }


def get_client(config: Optional[Config] = None) -> LLMClient:
    """
    Get an LLM client based on config.

    Args:
        config: Configuration to use, or None for global config

    Returns:
        LLM client instance
    """
    if config is None:
        from .config import get_config
        config = get_config()

    if config.provider == 'anthropic':
        return AnthropicClient(config)
    elif config.provider == 'openai':
        return OpenAIClient(config)
    else:
        raise ValueError(f"Unknown provider: {config.provider}")
