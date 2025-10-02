#!/usr/bin/env python3
"""
Complete example demonstrating the Result API inspired by functai.

This shows how the all= parameter works and how Result objects behave.
"""

from elemai import ai, _ai, Result, configure

# Configure the library (you'll need to set your API key)
# configure(model="claude-sonnet-4-5-20250929")


# Example 1: Simple function with no intermediate outputs
@ai
def summarize(text: str) -> str:
    """Summarize the text in one sentence"""
    return _ai


# Example 2: Function with intermediate thinking
@ai
def analyze_sentiment(text: str) -> str:
    """Analyze the sentiment of the text"""
    thinking: str = _ai["Think about the emotional tone"]
    return _ai


# Example 3: Multiple intermediate outputs
@ai
def solve_math_problem(question: str) -> float:
    """Solve a math word problem"""
    understanding: str = _ai["First, understand what the problem is asking"]
    approach: str = _ai["Explain your step-by-step approach"]
    calculation: str = _ai["Show the calculation"]
    answer: float = _ai["The final numeric answer"]
    return answer


# Example 4: Complex analysis with multiple steps
@ai
def deep_analysis(text: str, context: str = "") -> str:
    """Perform deep analysis of text"""
    background: str = _ai["Consider the background and context"]
    key_points: str = _ai["Identify the key points"]
    connections: str = _ai["Find connections to the context"]
    summary: str = _ai["Provide a comprehensive summary"]
    return summary


def demo():
    """Demonstrate the Result API."""
    print("=" * 70)
    print("Result API Demo - functai-inspired design")
    print("=" * 70)

    # Since we're not actually calling the LLM, we'll demonstrate with mock Results

    print("\n1. SIMPLE FUNCTION (no intermediate outputs)")
    print("-" * 70)
    print("Code: summarize('Long text...')")
    print("Returns: Just the string value")
    # result = summarize("Long text...")
    # In reality: result = "This is a concise summary"
    mock_result = "This is a concise summary"
    print(f"result = {repr(mock_result)}")
    print(f"Type: {type(mock_result).__name__}")

    print("\n2. FUNCTION WITH THINKING (default behavior)")
    print("-" * 70)
    print("Code: analyze_sentiment('This is amazing!')")
    print("Returns: Just the final result (thinking is hidden)")
    # result = analyze_sentiment("This is amazing!")
    mock_result = "positive"
    print(f"result = {repr(mock_result)}")
    print(f"Type: {type(mock_result).__name__}")

    print("\n3. FUNCTION WITH THINKING (all=True)")
    print("-" * 70)
    print("Code: analyze_sentiment('This is amazing!', all=True)")
    print("Returns: Result object with ALL fields")
    # full_result = analyze_sentiment("This is amazing!", all=True)
    mock_full_result = Result(
        thinking="The text uses enthusiastic language with 'amazing', indicating strong positive sentiment",
        result="positive"
    )
    print(f"\nResult object:")
    print(f"  repr: {repr(mock_full_result)}")
    print(f"  Type: {type(mock_full_result).__name__}")
    print(f"\nAccessing fields:")
    print(f"  full_result.thinking = {repr(mock_full_result.thinking)}")
    print(f"  full_result.result = {repr(mock_full_result.result)}")

    print("\n4. MULTIPLE INTERMEDIATE OUTPUTS (default)")
    print("-" * 70)
    print("Code: solve_math_problem('What is 15 * 23?')")
    print("Returns: Just the final answer")
    # answer = solve_math_problem("What is 15 * 23?")
    mock_answer = 345.0
    print(f"answer = {mock_answer}")
    print(f"Type: {type(mock_answer).__name__}")

    print("\n5. MULTIPLE INTERMEDIATE OUTPUTS (all=True)")
    print("-" * 70)
    print("Code: solve_math_problem('What is 15 * 23?', all=True)")
    print("Returns: Result object with complete reasoning chain")
    # full = solve_math_problem("What is 15 * 23?", all=True)
    mock_full = Result(
        understanding="The problem asks us to multiply 15 by 23",
        approach="I will multiply these two numbers step by step",
        calculation="15 × 23 = 15 × 20 + 15 × 3 = 300 + 45 = 345",
        result=345.0
    )
    print(f"\nResult object with all steps:")
    print(f"  {repr(mock_full)}\n")
    print("Accessing individual steps:")
    print(f"  understanding: {mock_full.understanding}")
    print(f"  approach: {mock_full.approach}")
    print(f"  calculation: {mock_full.calculation}")
    print(f"  result: {mock_full.result}")

    print("\n6. STRING REPRESENTATIONS")
    print("-" * 70)

    # Single field Result
    r1 = Result(result="positive")
    print("Single field Result:")
    print(f"  repr(r1) = {repr(r1)}")
    print(f"  str(r1) = {str(r1)}")

    # Multiple fields Result
    r2 = Result(thinking="analysis", sentiment="positive", result="positive")
    print("\nMultiple fields Result:")
    print(f"  repr(r2) = {repr(r2)}")
    print(f"  str(r2):")
    for line in str(r2).split('\n'):
        print(f"    {line}")

    print("\n7. JUPYTER NOTEBOOK RENDERING")
    print("-" * 70)
    print("In Jupyter notebooks, Result objects render as formatted markdown:\n")
    print(r2._repr_markdown_())

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
1. DEFAULT BEHAVIOR: Functions return just the final result value
   ✓ Clean, simple API
   ✓ result = func(input)

2. USE all=True TO GET EVERYTHING: Get the complete Result object
   ✓ Access all intermediate steps
   ✓ full = func(input, all=True)
   ✓ full.thinking, full.reasoning, full.result, etc.

3. NICE REPRESENTATIONS: Result objects display cleanly
   ✓ repr() shows all fields clearly
   ✓ str() adapts to single vs. multiple fields
   ✓ Jupyter markdown rendering for notebooks

4. FOLLOWS functai DESIGN: Same philosophy and API patterns
   ✓ Function-is-the-prompt paradigm
   ✓ Intuitive result handling
   ✓ Easy access to intermediate outputs when needed
    """)


if __name__ == "__main__":
    demo()
