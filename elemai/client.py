"""LLM client using litellm for unified provider access.

This module provides a simple interface to interact with various LLM providers
through litellm, supporting 100+ providers including Anthropic, OpenAI, Google, etc.

Example:
    >>> from elemai.config import Config
    >>> from elemai.client import LLMClient
    >>> config = Config(model="claude-sonnet-4-5-20250929")
    >>> client = LLMClient(config)
    >>> # client.complete([{"role": "user", "content": "Hello"}])
"""

import os
from typing import Any, Dict, List, Optional
from .config import Config, resolve_model


class LLMClient:
    """LLM client using litellm for unified access to multiple providers.

    This class provides a simple interface to send messages to LLM providers
    and receive responses. It handles model resolution, API key management,
    and response formatting.

    Attributes:
        config: Configuration object containing model settings
        litellm: Reference to litellm module

    Example:
        >>> from elemai.config import Config
        >>> config = Config(model="claude-sonnet-4-5-20250929", temperature=0.7)
        >>> client = LLMClient(config)
        >>> messages = [{"role": "user", "content": "Say hello"}]
        >>> # response = client.complete(messages)
        >>> # response['text'] would contain the LLM's response
    """

    def __init__(self, config: Config):
        """Initialize the LLM client with configuration.

        Args:
            config: Configuration object with model, temperature, and other settings

        Raises:
            ImportError: If litellm package is not installed

        Example:
            >>> from elemai.config import Config
            >>> config = Config(model="claude-sonnet-4-5-20250929")
            >>> client = LLMClient(config)
            >>> client.config.model
            'claude-sonnet-4-5-20250929'
        """
        self.config = config

        try:
            import litellm
            self.litellm = litellm

            # Suppress litellm logging by default
            litellm.suppress_debug_info = True

        except ImportError:
            raise ImportError(
                "litellm package not installed. Run: pip install litellm\n"
                "litellm provides unified access to 100+ LLM providers."
            )

    def complete(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Send messages to LLM and get response.

        This method sends a list of messages to the configured LLM provider
        and returns the response along with usage statistics.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            **kwargs: Additional parameters like model, temperature, max_tokens

        Returns:
            dict: Dictionary containing:
                - text (str): The LLM's response text
                - raw_response: The raw litellm response object
                - usage (dict): Token usage statistics

        Raises:
            RuntimeError: If API key is missing or invalid

        Example:
            >>> from elemai.config import Config
            >>> config = Config(model="claude-sonnet-4-5-20250929")
            >>> client = LLMClient(config)
            >>> messages = [{"role": "user", "content": "What is 2+2?"}]
            >>> # response = client.complete(messages)
            >>> # response['text'] would contain the answer
            >>> # response['usage'] would contain token counts
        """
        # Prepare parameters
        model = resolve_model(kwargs.get('model', self.config.model))

        params = {
            'model': model,
            'messages': messages,
            'temperature': kwargs.get('temperature', self.config.temperature),
        }

        # Add max_tokens if specified
        if self.config.max_tokens or kwargs.get('max_tokens'):
            params['max_tokens'] = kwargs.get('max_tokens', self.config.max_tokens)

        # Add API key if specified in config
        if self.config.api_key:
            params['api_key'] = self.config.api_key

        # Add any extra config parameters
        params.update(self.config.extra)

        # Call litellm
        try:
            response = self.litellm.completion(**params)
        except Exception as e:
            # Provide helpful error messages
            error_msg = str(e)

            if 'API_KEY' in error_msg.upper():
                raise RuntimeError(
                    f"API key not found. Set the appropriate environment variable:\n"
                    f"  - Anthropic: export ANTHROPIC_API_KEY='your-key'\n"
                    f"  - OpenAI: export OPENAI_API_KEY='your-key'\n"
                    f"  - Or pass api_key in config\n"
                    f"Original error: {error_msg}"
                )
            raise

        # Extract response
        text = response.choices[0].message.content or ""

        # Extract usage info
        usage = {}
        if hasattr(response, 'usage') and response.usage:
            usage = {
                'input_tokens': getattr(response.usage, 'prompt_tokens', 0),
                'output_tokens': getattr(response.usage, 'completion_tokens', 0),
                'total_tokens': getattr(response.usage, 'total_tokens', 0),
            }

        return {
            'text': text,
            'raw_response': response,
            'usage': usage,
        }


def get_client(config: Optional[Config] = None) -> LLMClient:
    """Get an LLM client instance.

    This is a convenience function to create an LLMClient. If no config is
    provided, it uses the global configuration.

    Args:
        config: Configuration to use, or None to use global config

    Returns:
        LLMClient: Configured LLM client instance

    Example:
        >>> from elemai.client import get_client
        >>> client = get_client()
        >>> client.config.model
        'claude-sonnet-4-5-20250929'

        >>> from elemai.config import Config
        >>> custom_config = Config(model="claude-sonnet-4-5-20250929", temperature=0.5)
        >>> client = get_client(custom_config)
        >>> client.config.temperature
        0.5
    """
    if config is None:
        from .config import get_config
        config = get_config()

    return LLMClient(config)
