"""Tests for Result object and all= parameter."""

import pytest
from elemai.task import Result


def test_result_single_field():
    """Test Result with single field."""
    r = Result(result="positive")

    assert r.result == "positive"
    assert hasattr(r, '_fields')
    assert r._fields == {'result': 'positive'}


def test_result_multiple_fields():
    """Test Result with multiple fields."""
    r = Result(
        thinking="The text is positive",
        sentiment="positive",
        result="positive"
    )

    assert r.thinking == "The text is positive"
    assert r.sentiment == "positive"
    assert r.result == "positive"
    assert len(r._fields) == 3


def test_result_repr_single_field():
    """Test Result repr with single field."""
    r = Result(result="positive")

    assert repr(r) == "Result(result='positive')"


def test_result_repr_multiple_fields():
    """Test Result repr with multiple fields."""
    r = Result(thinking="analysis", result="positive")

    # Should show all fields
    assert "thinking" in repr(r)
    assert "result" in repr(r)
    assert "Result(" in repr(r)


def test_result_str_single_field():
    """Test Result str with single field."""
    r = Result(result="positive")

    # Single field should just show the value
    assert str(r) == "positive"


def test_result_str_multiple_fields():
    """Test Result str with multiple fields."""
    r = Result(thinking="analysis", result="positive")

    # Multiple fields should show all on separate lines
    s = str(r)
    assert "thinking: analysis" in s
    assert "result: positive" in s
    assert "\n" in s


def test_result_markdown_single_field():
    """Test Result markdown representation with single field."""
    r = Result(result="positive")

    md = r._repr_markdown_()
    assert "**Result:**" in md
    assert "positive" in md


def test_result_markdown_multiple_fields():
    """Test Result markdown representation with multiple fields."""
    r = Result(
        thinking="The text is positive",
        sentiment="positive",
        result="positive"
    )

    md = r._repr_markdown_()
    assert "### AI Function Result" in md
    assert "**thinking:**" in md
    assert "**sentiment:**" in md
    assert "**result:**" in md


def test_result_attribute_access():
    """Test that fields are accessible as attributes."""
    r = Result(field1="value1", field2="value2", result="final")

    assert r.field1 == "value1"
    assert r.field2 == "value2"
    assert r.result == "final"


def test_result_with_complex_types():
    """Test Result with complex field types."""
    r = Result(
        items=["apple", "orange", "banana"],
        count=3,
        summary={"total": 3, "types": ["fruit"]},
        result="processed"
    )

    assert r.items == ["apple", "orange", "banana"]
    assert r.count == 3
    assert r.summary == {"total": 3, "types": ["fruit"]}
    assert r.result == "processed"


def test_result_empty():
    """Test Result with no fields."""
    r = Result()

    assert r._fields == {}
    assert repr(r) == "Result()"


def test_result_field_ordering():
    """Test that field order is preserved in representations."""
    r = Result(first="a", second="b", third="c")

    # Check that all fields appear in repr
    r_str = repr(r)
    assert "first='a'" in r_str
    assert "second='b'" in r_str
    assert "third='c'" in r_str
