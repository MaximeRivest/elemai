"""Configuration system for elemai."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from contextlib import contextmanager


@dataclass
class Config:
    """Global configuration for elemai."""

    model: str = "claude-3-5-sonnet-20241022"
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
# litellm uses model names like: claude-3-5-sonnet-20241022, gpt-4, etc.
MODEL_ALIASES = {
    'sonnet': 'claude-3-5-sonnet-20241022',
    'opus': 'claude-3-opus-20240229',
    'haiku': 'claude-3-haiku-20240307',
    'gpt4': 'gpt-4-turbo',
    'gpt4o': 'gpt-4o',
    'gpt35': 'gpt-3.5-turbo',
}


def resolve_model(model: str) -> str:
    """Resolve a model alias to its full name."""
    return MODEL_ALIASES.get(model, model)
