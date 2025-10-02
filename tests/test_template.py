"""Tests for template system."""

import pytest
from elemai.template import MessageTemplate, TemplateFunctions, templates, Field


def test_simple_template_rendering():
    """Test basic template rendering"""
    template = MessageTemplate([
        {"role": "system", "content": "You are {role}"},
        {"role": "user", "content": "{text}"}
    ])

    messages = template.render(role="helpful", text="Hello")

    assert len(messages) == 2
    assert messages[0]["content"] == "You are helpful"
    assert messages[1]["content"] == "Hello"


def test_inputs_function():
    """Test inputs() template function"""
    template = MessageTemplate([
        {"role": "user", "content": "{inputs()}"}
    ])

    messages = template.render(inputs={"text": "hello", "count": 5})

    assert "text: hello" in messages[0]["content"]
    assert "count: 5" in messages[0]["content"]


def test_inputs_function_with_style():
    """Test inputs() with style parameter"""
    template = MessageTemplate([
        {"role": "user", "content": "{inputs(style='list')}"}
    ])

    messages = template.render(inputs={"text": "hello", "count": 5})

    assert "- text" in messages[0]["content"]
    assert "- count" in messages[0]["content"]


def test_outputs_function():
    """Test outputs() template function"""
    template = MessageTemplate([
        {"role": "system", "content": "{outputs(style='list')}"}
    ])

    output_fields = [
        Field("thinking", str, "Your reasoning"),
        Field("answer", str, "The final answer")
    ]

    messages = template.render(output_fields=output_fields)

    assert "thinking" in messages[0]["content"]
    assert "answer" in messages[0]["content"]


def test_nested_access():
    """Test nested variable access like {inputs.text}"""
    template = MessageTemplate([
        {"role": "user", "content": "Text: {inputs.text}"}
    ])

    messages = template.render(inputs={"text": "hello", "other": "world"})

    assert messages[0]["content"] == "Text: hello"


def test_builtin_templates():
    """Test built-in template presets"""
    # Simple template
    template = MessageTemplate(templates.simple)
    messages = template.render(
        instruction="Do something",
        inputs={"text": "input"}
    )

    assert any("Do something" in m["content"] for m in messages)


def test_template_functions_registry():
    """Test custom template function registration"""
    funcs = TemplateFunctions()

    def custom_func():
        return "custom output"

    funcs.register("custom", custom_func)

    result = funcs.call("custom")
    assert result == "custom output"


def test_schema_rendering():
    """Test schema rendering for types"""
    funcs = TemplateFunctions()

    schema_str = funcs._type_to_schema(str)
    assert schema_str == "string"

    schema_int = funcs._type_to_schema(int)
    assert schema_int == "integer"


def test_function_call_parsing():
    """Test parsing of function calls in templates"""
    template = MessageTemplate([
        {"role": "user", "content": "{inputs(style='yaml')}"}
    ])

    # Should parse and execute the function call
    messages = template.render(inputs={"key": "value"})

    # YAML format should include the key
    assert "key" in messages[0]["content"]


def test_empty_context():
    """Test template with missing context variables"""
    template = MessageTemplate([
        {"role": "user", "content": "Text: {text}"}
    ])

    # Should handle missing variables gracefully
    try:
        messages = template.render()
        # If it doesn't raise, check it returns something
        assert len(messages) > 0
    except KeyError:
        # Expected if strict mode
        pass
