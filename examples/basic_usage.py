"""Basic usage examples for elemai."""

from elemai import ai, _ai, Chat


# Example 1: Simple AI function
@ai
def summarize(text: str) -> str:
    """Summarize the text in one sentence"""
    return _ai


# Example 2: With intermediate reasoning
@ai
def analyze_sentiment(text: str) -> str:
    """Analyze the sentiment of the text"""
    thinking: str = _ai["Think about the emotional tone"]
    return _ai


# Example 3: Structured output with Pydantic
from pydantic import BaseModel


class Analysis(BaseModel):
    sentiment: str
    themes: list[str]
    summary: str


@ai
def deep_analysis(text: str) -> Analysis:
    """Perform deep analysis of the text"""
    thinking: str = _ai["Analyze the text carefully"]
    return _ai


# Example 4: Custom messages template
@ai(
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Text: {text}\n\nPlease summarize this."},
    ]
)
def custom_summarize(text: str) -> str:
    """Summarize with custom template"""
    return _ai


# Example 5: Using template functions
@ai(
    messages=[
        {
            "role": "system",
            "content": "Task: {instruction}\n\nExpected output:\n{outputs(style='schema')}"
        },
        {"role": "user", "content": "{inputs(style='yaml')}"},
    ]
)
def structured_task(text: str, context: str) -> Analysis:
    """Analyze text with context"""
    return _ai


# Example 6: Assistant prefill
@ai(
    messages=[
        {"role": "system", "content": "You are concise"},
        {"role": "user", "content": "{text}"},
        {"role": "assistant", "content": "Here's my analysis:\n\n1. "},
    ]
)
def prefilled_analysis(text: str) -> str:
    """Analysis with prefilled start"""
    return _ai


# Example 7: Chat mode
def chat_example():
    """Demonstrate chat functionality"""
    chat = Chat(system="You are a helpful assistant")

    response1 = chat("My name is Alice")
    print(f"Bot: {response1}")

    response2 = chat("What's my name?")
    print(f"Bot: {response2}")


# Example 8: Chat with custom model
def chat_with_config():
    """Chat with custom configuration"""
    chat = Chat(model="opus", temperature=0.3, system="You are very precise")

    return chat("Explain quantum computing")


# Example 9: Multiple outputs
@ai
def solve_problem(problem: str) -> float:
    """Solve a math problem"""
    understanding: str = _ai["First, understand the problem"]
    approach: str = _ai["Explain your approach"]
    answer: float = _ai["The final numeric answer"]
    return answer


# Example 10: Configuration override
from elemai import configure


def with_config_override():
    """Use configuration override"""

    @ai
    def task(input: str) -> str:
        """Do something"""
        return _ai

    # Normal usage
    result1 = task("input 1")

    # Override for specific call
    with configure(model="haiku", temperature=0):
        result2 = task("input 2")

    return result1, result2


if __name__ == "__main__":
    # Test simple function
    result = summarize("This is a long text about artificial intelligence...")
    print(f"Summary: {result}")

    # Test with reasoning
    result = analyze_sentiment("I love this product! It's amazing!")
    print(f"Sentiment: {result}")

    # Preview what would be sent
    print("\nPreview of structured_task:")
    preview = structured_task.preview(
        text="Great product!",
        context="Customer review"
    )
    print(preview.prompt)
