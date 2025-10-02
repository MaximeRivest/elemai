"""Advanced usage examples for elemai."""

from elemai import ai, _ai, template_fn, MessageTemplate
from dataclasses import dataclass
from typing import List


# Example 1: Custom template function
@template_fn
def render_with_emphasis(inputs: dict, highlight: List[str] = None) -> str:
    """Render inputs with highlighted fields"""
    lines = []
    for k, v in inputs.items():
        if highlight and k in highlight:
            lines.append(f">>> {k.upper()}: {v}")
        else:
            lines.append(f"    {k}: {v}")
    return '\n'.join(lines)


@ai(
    messages=[
        {"role": "system", "content": "Analyze carefully"},
        {"role": "user", "content": "{render_with_emphasis(inputs, highlight=['text'])}"},
    ]
)
def highlighted_analysis(text: str, context: str) -> str:
    """Analysis with highlighted inputs"""
    return _ai


# Example 2: Complex structured output
@dataclass
class DetailedAnalysis:
    """Detailed text analysis"""
    sentiment: str
    confidence: float
    key_themes: List[str]
    entities: List[str]
    summary: str


@ai
def comprehensive_analysis(text: str) -> DetailedAnalysis:
    """Perform comprehensive text analysis"""
    reasoning: str = _ai["Think through all aspects"]
    entity_extraction: str = _ai["Identify key entities"]
    return _ai


# Example 3: Multi-step reasoning
@ai(
    messages=[
        {
            "role": "system",
            "content": "You solve problems step by step"
        },
        {
            "role": "user",
            "content": "Problem: {problem}"
        },
        {
            "role": "assistant",
            "content": """I'll solve this step by step:

UNDERSTANDING:
{understanding}

APPROACH:
{approach}

SOLUTION:
{solution}

VERIFICATION:
{verification}

FINAL ANSWER:
{answer}"""
        }
    ]
)
def solve_with_verification(problem: str) -> str:
    """Solve problem with full reasoning and verification"""
    understanding: str = _ai
    approach: str = _ai
    solution: str = _ai
    verification: str = _ai
    answer: str = _ai
    return answer


# Example 4: Dynamic message construction
def build_custom_messages(fn_name, instruction, demos, **inputs):
    """Build messages with custom logic"""
    messages = [
        {"role": "system", "content": f"You are performing: {instruction}"}
    ]

    # Add few-shot examples
    for demo in demos:
        messages.append({"role": "user", "content": demo['input']})
        messages.append({"role": "assistant", "content": demo['output']})

    # Add actual input
    messages.append({"role": "user", "content": inputs.get('text', '')})

    return messages


@ai(messages=build_custom_messages)
def dynamic_task(text: str) -> str:
    """Task with dynamically built messages"""
    return _ai


# Example 5: Using demos for few-shot learning
@ai
def classify(text: str) -> str:
    """Classify text into categories"""
    return _ai


def classify_with_examples():
    """Use few-shot examples"""
    demos = [
        {"text": "I love this!", "result": "positive"},
        {"text": "This is terrible", "result": "negative"},
        {"text": "It's okay I guess", "result": "neutral"},
    ]

    result = classify("This is amazing!", demos=demos)
    return result


# Example 6: Chat with tasks
from elemai import Chat


def chat_with_tasks():
    """Chat that can use AI tasks"""
    chat = Chat(system="You are a helpful analyst")

    # Register a task
    @chat.task
    def analyze(text: str) -> DetailedAnalysis:
        """Analyze text in detail"""
        return _ai

    # Now chat can reference or use this task
    response = chat("Can you analyze this text for me: 'Great product!'")
    return response


# Example 7: Reusable template sections
from elemai import MessageTemplate


expert_system = {
    "role": "system",
    "content": "You are an expert with 20 years of experience."
}


def create_expert_task(instruction: str):
    """Create a task with expert system prompt"""

    @ai(
        messages=[
            expert_system,
            {"role": "user", "content": "{inputs()}"},
        ]
    )
    def expert_task(text: str) -> str:
        return _ai

    expert_task.__doc__ = instruction
    return expert_task


# Example 8: Inspection and debugging
@ai
def inspectable_task(text: str) -> str:
    """A task we can inspect"""
    thinking: str = _ai["Analyze carefully"]
    return _ai


def inspect_task():
    """Demonstrate inspection capabilities"""
    # See the template
    print("Template messages:", inspectable_task.template.messages)

    # See rendered prompt
    rendered = inspectable_task.render(text="example input")
    print("\nRendered prompt:")
    print(rendered)

    # See actual messages
    messages = inspectable_task.to_messages(text="example input")
    print("\nMessages to send:")
    print(messages)

    # Preview everything
    preview = inspectable_task.preview(text="example input")
    print("\nPreview:")
    print(f"Model: {preview.config.model}")
    print(f"Temperature: {preview.config.temperature}")


# Example 9: Conditional outputs
@ai(
    messages=[
        {"role": "system", "content": "{instruction}"},
        {
            "role": "user",
            "content": """Text: {inputs.text}

Analyze this and provide:
- Sentiment
- Key themes
- Summary
{if inputs.get('include_entities'): '- Named entities'}
"""
        }
    ]
)
def conditional_analysis(text: str, include_entities: bool = False) -> str:
    """Analysis with optional entity extraction"""
    return _ai


# Example 10: Complex workflow
@ai
def extract_data(text: str) -> dict:
    """Extract structured data"""
    return _ai


@ai
def validate_data(data: dict) -> bool:
    """Validate extracted data"""
    return _ai


@ai
def summarize_data(data: dict) -> str:
    """Summarize validated data"""
    return _ai


def pipeline_example(text: str):
    """Multi-step AI pipeline"""
    # Step 1: Extract
    data = extract_data(text)

    # Step 2: Validate
    is_valid = validate_data(data)

    if is_valid:
        # Step 3: Summarize
        summary = summarize_data(data)
        return summary
    else:
        return "Invalid data extracted"


if __name__ == "__main__":
    # Test custom template function
    result = highlighted_analysis(
        text="This is important",
        context="background info"
    )
    print(f"Result: {result}")

    # Test inspection
    print("\n--- Inspection Demo ---")
    inspect_task()

    # Test few-shot
    print("\n--- Few-shot Demo ---")
    classification = classify_with_examples()
    print(f"Classification: {classification}")
