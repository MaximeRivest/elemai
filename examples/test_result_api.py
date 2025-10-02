#!/usr/bin/env python3
"""Test the new Result API similar to functai's design."""

from elemai import ai, _ai, Result

# Example 1: Simple function - returns just the value
@ai
def analyze_sentiment(text: str) -> str:
    """Analyze the sentiment of the text"""
    return _ai

# Example 2: Function with intermediate thinking
@ai
def analyze_sentiment_with_thinking(text: str) -> str:
    """Analyze the sentiment of the text"""
    thinking: str = _ai["Think about the emotional tone"]
    return _ai


def demo_simple():
    """Demonstrate simple case - just returns the value."""
    print("=" * 60)
    print("Example 1: Simple function (no intermediate outputs)")
    print("=" * 60)

    # Mock the result for demonstration
    # In reality this would call the LLM
    # result = analyze_sentiment("This is amazing")

    # Create a mock result to show the behavior
    result = "positive"
    print(f"Result: {result}")
    print(f"Type: {type(result)}")
    print(f"repr: {repr(result)}")
    print()


def demo_with_thinking():
    """Demonstrate function with intermediate thinking."""
    print("=" * 60)
    print("Example 2: Function with intermediate thinking")
    print("=" * 60)

    # Create mock results to demonstrate the behavior
    print("\n--- Default behavior (all=False) ---")
    # result = analyze_sentiment_with_thinking("This is amazing")
    # This would return just the sentiment string: "positive"
    mock_result = "positive"
    print(f"Result: {mock_result}")
    print(f"Type: {type(mock_result)}")
    print(f"repr: {repr(mock_result)}")

    print("\n--- With all=True parameter ---")
    # full_result = analyze_sentiment_with_thinking("This is amazing", all=True)
    # This returns a Result object with all fields
    mock_full_result = Result(
        thinking="The text uses positive words like 'amazing' indicating positive sentiment",
        result="positive"
    )
    print(f"Result object: {mock_full_result}")
    print(f"Type: {type(mock_full_result)}")
    print(f"repr: {repr(mock_full_result)}")
    print(f"\nAccess thinking: {mock_full_result.thinking}")
    print(f"Access result: {mock_full_result.result}")
    print()


def demo_result_representations():
    """Demonstrate different Result representations."""
    print("=" * 60)
    print("Example 3: Result object representations")
    print("=" * 60)

    # Single field Result
    print("\n--- Single field Result ---")
    r1 = Result(result="positive")
    print(f"repr: {repr(r1)}")
    print(f"str: {str(r1)}")

    # Multiple fields Result
    print("\n--- Multiple fields Result ---")
    r2 = Result(
        thinking="The text expresses strong positive emotion",
        sentiment="positive",
        result="positive"
    )
    print(f"repr: {repr(r2)}")
    print(f"str:\n{str(r2)}")

    # Markdown representation (for Jupyter)
    print("\n--- Markdown representation ---")
    print(r2._repr_markdown_())
    print()


if __name__ == "__main__":
    demo_simple()
    demo_with_thinking()
    demo_result_representations()

    print("=" * 60)
    print("Summary of the API design:")
    print("=" * 60)
    print("""
1. By default, @ai functions return just the final result value
   - Simple, clean API: result = func(input)

2. Use all=True to get the complete Result object with all fields
   - Access intermediate steps: full = func(input, all=True)
   - full.thinking, full.result, etc.

3. Result object has nice representations:
   - repr() shows all fields
   - str() shows just the result for single-field, or all for multi-field
   - _repr_markdown_() for Jupyter notebooks

This matches functai's design philosophy!
    """)
