"""Configuration system for elemai."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from contextlib import contextmanager


@dataclass
class Config:
    """Global configuration for elemai."""

    model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    api_key: Optional[str] = None
    default_template: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def copy(self):
        """Create a copy of this config."""
        return Config(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=self.api_key,
            default_template=self.default_template,
            extra=self.extra.copy()
        )

    def merge(self, **kwargs):
        """Create a new config with updated values."""
        new_config = self.copy()
        for key, value in kwargs.items():
            if value is not None:
                setattr(new_config, key, value)
        return new_config


# Global default configuration
_global_config = Config()


def get_config() -> Config:
    """Get the current global configuration."""
    return _global_config


def set_config(**kwargs):
    """Update the global configuration."""
    global _global_config
    for key, value in kwargs.items():
        if hasattr(_global_config, key):
            setattr(_global_config, key, value)


@contextmanager
def configure(**kwargs):
    """
    Temporarily override configuration.

    Example:
        with configure(model="opus", temperature=0):
            result = task(input)
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
    # Claude 4 models (latest)
    'sonnet': 'claude-sonnet-4-20250514',
    'opus': 'claude-opus-4-20250514',
    'haiku': 'claude-3-5-haiku-20241022',

    # Claude 3.7
    'sonnet-3.7': 'claude-3-7-sonnet-20250219',

    # Claude 3.5 (previous generation)
    'sonnet-3.5': 'claude-3-5-sonnet-20241022',

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
    """Resolve a model alias to its full name."""
    return MODEL_ALIASES.get(model, model)
