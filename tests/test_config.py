"""Tests for configuration system."""

import pytest
from elemai.config import Config, get_config, set_config, configure, resolve_model


def test_default_config():
    """Test default configuration"""
    config = Config()

    assert config.model == "claude-3-5-sonnet-20241022"
    assert config.temperature == 0.7


def test_config_copy():
    """Test config copying"""
    config1 = Config(model="opus", temperature=0.5)
    config2 = config1.copy()

    assert config2.model == "opus"
    assert config2.temperature == 0.5

    # Should be independent
    config2.model = "sonnet"
    assert config1.model == "opus"


def test_config_merge():
    """Test config merging"""
    config1 = Config(model="opus", temperature=0.5)
    config2 = config1.merge(temperature=0.8, max_tokens=1000)

    assert config2.model == "opus"  # Unchanged
    assert config2.temperature == 0.8  # Changed
    assert config2.max_tokens == 1000  # New


def test_set_global_config():
    """Test setting global configuration"""
    original_model = get_config().model

    set_config(model="test-model")

    assert get_config().model == "test-model"

    # Restore
    set_config(model=original_model)


def test_configure_context_manager():
    """Test temporary configuration override"""
    original_config = get_config()
    original_model = original_config.model

    with configure(model="temp-model", temperature=0.1):
        assert get_config().model == "temp-model"
        assert get_config().temperature == 0.1

    # Should be restored
    assert get_config().model == original_model


def test_model_aliases():
    """Test model alias resolution"""
    assert resolve_model("sonnet") == "claude-3-5-sonnet-20241022"
    assert resolve_model("opus") == "claude-3-opus-20240229"
    assert resolve_model("haiku") == "claude-3-haiku-20240307"
    assert resolve_model("gpt4o") == "gpt-4o"

    # Unknown models pass through
    assert resolve_model("custom-model") == "custom-model"


def test_config_extra_fields():
    """Test extra configuration fields"""
    config = Config(extra={"custom": "value"})

    assert config.extra["custom"] == "value"
