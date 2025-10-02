"""Tests for _ai sentinel and introspection."""

import pytest
from elemai.sentinel import _ai, AISentinel, FunctionIntrospector


def test_ai_sentinel_basic():
    """Test basic _ai sentinel"""
    sentinel = _ai

    assert isinstance(sentinel, AISentinel)
    assert sentinel.description is None


def test_ai_sentinel_with_description():
    """Test _ai with description"""
    sentinel = _ai["Think step by step"]

    assert isinstance(sentinel, AISentinel)
    assert sentinel.description == "Think step by step"


def test_function_introspection_inputs():
    """Test extracting input fields from function"""

    def test_func(text: str, count: int = 5):
        pass

    inspector = FunctionIntrospector(test_func)
    inputs = inspector.get_input_fields()

    assert len(inputs) == 2
    assert inputs[0]['name'] == 'text'
    assert inputs[0]['type'] == str
    assert inputs[1]['name'] == 'count'
    assert inputs[1]['default'] == 5


def test_function_introspection_outputs():
    """Test extracting output fields from function body"""

    def test_func(text: str) -> str:
        thinking: str = _ai["Think about it"]
        return _ai

    inspector = FunctionIntrospector(test_func)
    outputs = inspector.get_output_fields()

    # Should find 'thinking' and 'result'
    names = [o['name'] for o in outputs]
    assert 'thinking' in names


def test_function_introspection_docstring():
    """Test extracting instruction from docstring"""

    def test_func(text: str) -> str:
        """This is the instruction"""
        return _ai

    inspector = FunctionIntrospector(test_func)
    instruction = inspector.get_instruction()

    assert instruction == "This is the instruction"


def test_function_introspection_full():
    """Test full introspection"""

    def test_func(text: str, context: str) -> str:
        """Analyze the text"""
        thinking: str = _ai["Reason through this"]
        return _ai

    inspector = FunctionIntrospector(test_func)
    metadata = inspector.introspect()

    assert metadata['fn_name'] == 'test_func'
    assert metadata['instruction'] == 'Analyze the text'
    assert len(metadata['input_fields']) == 2
    assert metadata['return_type'] == str


def test_introspection_no_source():
    """Test introspection when source is not available"""
    # Built-in functions don't have source
    inspector = FunctionIntrospector(len)

    # Should handle gracefully
    outputs = inspector.get_output_fields()
    assert isinstance(outputs, list)
