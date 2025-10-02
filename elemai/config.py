"""Configuration system for elemai.

This module provides global configuration management for the elemai library,
including model selection, temperature settings, and API keys.

Example:
    >>> from elemai import set_config, configure
    >>> # Set global configuration
    >>> set_config(model="opus", temperature=0.5)
    >>>
    >>> # Temporary override
    >>> with configure(model="haiku"):
    ...     pass  # Uses haiku model here
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from contextlib import contextmanager


@dataclass
class Config:
    """Configuration dataclass for elemai settings.

    Attributes:
        model: The LLM model to use (default: claude-sonnet-4.5)
        temperature: Sampling temperature 0-1 (default: 0.7)
        max_tokens: Maximum tokens to generate (default: None/provider default)
        api_key: API key for the provider (default: None/uses env vars)
        default_template: Default message template to use (default: None)
        extra: Additional provider-specific parameters (default: {})

    Example:
        >>> config = Config(model="gpt4o", temperature=0.3)
        >>> config.model
        'gpt4o'
        >>> config.temperature
        0.3
    """

    model: str = "claude-sonnet-4-5-20250929"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    api_key: Optional[str] = None
    default_template: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def copy(self):
        """Create an independent copy of this configuration.

        Returns:
            Config: A new Config instance with the same values

        Example:
            >>> config1 = Config(model="sonnet")
            >>> config2 = config1.copy()
            >>> config2.model = "opus"
            >>> config1.model  # Original unchanged
            'sonnet'
        """
        return Config(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=self.api_key,
            default_template=self.default_template,
            extra=self.extra.copy()
        )

    def merge(self, **kwargs):
        """Create a new config with updated values.

        Args:
            **kwargs: Configuration parameters to update

        Returns:
            Config: A new Config instance with merged values

        Example:
            >>> config = Config(model="sonnet", temperature=0.7)
            >>> new_config = config.merge(temperature=0.3, max_tokens=1000)
            >>> new_config.temperature
            0.3
            >>> new_config.model  # Unchanged values preserved
            'sonnet'
        """
        new_config = self.copy()
        for key, value in kwargs.items():
            if value is not None:
                setattr(new_config, key, value)
        return new_config


# Global default configuration
_global_config = Config()


def get_config() -> Config:
    """Get the current global configuration.

    Returns:
        Config: The current global configuration instance

    Example:
        >>> from elemai.config import Config
        >>> # Create a fresh config to show default
        >>> fresh_config = Config()
        >>> fresh_config.model
        'claude-sonnet-4-5-20250929'
    """
    return _global_config


def set_config(**kwargs):
    """Update the global configuration.

    This modifies the global config that will be used by all AI functions
    unless overridden locally.

    Args:
        **kwargs: Configuration parameters to update (model, temperature, etc.)

    Example:
        >>> from elemai import set_config, get_config
        >>> set_config(model="opus", temperature=0.5)
        >>> get_config().model
        'opus'
        >>> get_config().temperature
        0.5
    """
    global _global_config
    for key, value in kwargs.items():
        if hasattr(_global_config, key):
            setattr(_global_config, key, value)


@contextmanager
def configure(**kwargs):
    """Temporarily override configuration within a context.

    The configuration changes are automatically reverted when exiting
    the context manager.

    Args:
        **kwargs: Configuration parameters to temporarily override

    Yields:
        Config: The temporarily modified configuration

    Example:
        >>> from elemai import configure, get_config
        >>> original = get_config().model
        >>> with configure(model="haiku", temperature=0):
        ...     assert get_config().model == "haiku"
        ...     assert get_config().temperature == 0
        >>> get_config().model == original  # Restored
        True
    """
    global _global_config
    old_config = _global_config
    _global_config = _global_config.merge(**kwargs)
    try:
        yield _global_config
    finally:
        _global_config = old_config


# Model aliases for convenience
# Based on latest models as of 2025
MODEL_ALIASES = {
    # Claude 4.5 and 4.x (latest)
    'sonnet': 'claude-sonnet-4-5-20250929',  # Latest Sonnet 4.5
    'sonnet-4': 'claude-sonnet-4-20250514',   # Sonnet 4
    'opus': 'claude-opus-4-1-20250805',       # Latest Opus 4.1
    'opus-4': 'claude-opus-4-20250514',       # Opus 4
    'haiku': 'claude-3-5-haiku-20241022',     # Latest Haiku 3.5 (no 4.x yet)

    # OpenAI GPT models
    'gpt4o': 'gpt-4o',
    'gpt4': 'gpt-4-turbo',
    'gpt35': 'gpt-3.5-turbo',
    'gpt4o-mini': 'gpt-4o-mini',

    # Google Gemini models
    'gemini-pro': 'gemini-2.5-pro',
    'gemini-flash': 'gemini-2.5-flash',
    'gemini-flash-lite': 'gemini-2.5-flash-lite',
}


def resolve_model(model: str) -> str:
    """Resolve a model alias to its full name.

    If the model string is an alias (like 'sonnet'), returns the full
    model name. Otherwise, returns the input unchanged.

    Args:
        model: Model name or alias

    Returns:
        str: Full model name

    Example:
        >>> resolve_model("sonnet")
        'claude-sonnet-4-5-20250929'
        >>> resolve_model("opus")
        'claude-opus-4-1-20250805'
        >>> resolve_model("haiku")
        'claude-3-5-haiku-20241022'
        >>> resolve_model("gpt4o")
        'gpt-4o'
        >>> resolve_model("my-custom-model")  # Unknown alias passes through
        'my-custom-model'
    """
    return MODEL_ALIASES.get(model, model)
