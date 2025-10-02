"""LLM client using litellm for unified provider access."""

import os
from typing import Any, Dict, List, Optional
from .config import Config, resolve_model


class LLMClient:
    """LLM client using litellm for unified access to multiple providers."""

    def __init__(self, config: Config):
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
        """
        Send messages to LLM and get response.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional parameters (model, temperature, etc.)

        Returns:
            Dict with 'text', 'raw_response', and 'usage'
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
    """
    Get an LLM client.

    Args:
        config: Configuration to use, or None for global config

    Returns:
        LLM client instance
    """
    if config is None:
        from .config import get_config
        config = get_config()

    return LLMClient(config)
