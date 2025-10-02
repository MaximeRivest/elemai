# elemai

**elemai** is a productive, ergonomic Python library for working with LLMs. It provides a clean interface that feels like writing normal Python functions, while giving you full control over prompts and message construction.

## Philosophy

**The function definition is the prompt, and the function body is the program.**

- 🎯 **Function-as-prompt**: Define AI tasks as regular Python functions
- 📝 **Direct message control**: Use standard OpenAI/Anthropic message format with `{template}` variables
- 🔧 **Template functions**: Programmatic control over input/output rendering
- 🧩 **Structured outputs**: Pydantic models and dataclasses work automatically
- 🎓 **Few-shot learning**: Built-in demos parameter for in-context learning
- 🔀 **Conditional templates**: Ternary expressions `{var ? "yes" : "no"}` for dynamic prompts
- 💬 **Stateful chat**: Easy conversational interfaces
- ⚙️ **Flexible configuration**: Global, contextual, and per-function settings
- 🎨 **Progressive disclosure**: Simple by default, powerful when needed

## Installation

```bash
pip install elemai
```

Set your API key:

```bash
export ANTHROPIC_API_KEY="your-key-here"
# or
export OPENAI_API_KEY="your-key-here"
```

## Quick Start

### Simple AI Function

```python
from elemai import ai, _ai

@ai
def summarize(text: str) -> str:
    """Summarize the text in one sentence"""
    return _ai

result = summarize("Long text here...")
```

### Multi-Step Reasoning

Variable names + type hints + descriptions = automatic structure:

```python
@ai
def analyze_sentiment(text: str) -> str:
    """Analyze the sentiment of the text"""
    thinking: str = _ai["Think about the emotional tone"]
    return _ai

# Get just the result
result = analyze_sentiment("This is amazing!")

# Get all intermediate outputs
full = analyze_sentiment("This is amazing!", all=True)
print(full.thinking)
print(full.result)
```

### Structured Output with Pydantic

```python
from pydantic import BaseModel

class Analysis(BaseModel):
    sentiment: str
    themes: list[str]
    summary: str

@ai
def deep_analysis(text: str) -> Analysis:
    """Perform deep analysis of the text"""
    thinking: str = _ai["Analyze carefully"]
    return _ai

result = deep_analysis("I love this product! It's innovative and well-designed.")
print(result.sentiment)  # "positive"
print(result.themes)     # ["product satisfaction", "innovation", ...]
```

### Custom Message Templates

Direct control using standard message format with variable interpolation:

```python
@ai(
    messages=[
        {"role": "system", "content": "You are a helpful pirate. {instruction}"},
        {"role": "user", "content": "Text: {text}\n\n"}
    ]
)
def pirate_summarize(text: str) -> str:
    """Summarize in 10 words"""
    return _ai

result = pirate_summarize("Foundation models are now mature enough...")
# "Arrr! Foundation models help collect real-world data for scientific discovery, matey!"
```

### Conditional Templates

Use ternary expressions for dynamic prompts:

```python
@ai(
    messages=[
        {
            "role": "system",
            "content": "You are helpful{formal ? \" who speaks formally\" : \"\"}"
        },
        {
            "role": "user",
            "content": "{urgent ? \"[URGENT] \" : \"\"}Question: {question}"
        }
    ]
)
def ask_question(question: str, urgent: bool = False, formal: bool = False) -> str:
    """Ask a question with optional urgency and formality"""
    return _ai

result = ask_question("What is AI?", urgent=True)
# Includes [URGENT] prefix in the prompt
```

### Template Functions

Automatic input/output formatting without hardcoding field names:

```python
@ai(
    messages=[
        {
            "role": "system",
            "content": "Task: {instruction}\n\nExpected output:\n{outputs(style='schema')}"
        },
        {"role": "user", "content": "{inputs(style='yaml')}"}
    ]
)
def structured_task(text: str, context: str) -> Analysis:
    """Analyze text with context"""
    return _ai

result = structured_task(
    text="Great product!",
    context="Customer review"
)
```

Available template functions:
- `{inputs()}` - All inputs, auto-formatted
- `{inputs(style='yaml')}` - YAML format
- `{outputs(style='schema')}` - JSON schema
- `{inputs(exclude=['context'])}` - Exclude fields
- `{demos()}` - Render few-shot examples

### Few-Shot Learning with Demos

```python
@ai
def classify(text: str) -> str:
    """Classify text sentiment"""
    return _ai

demos = [
    {"text": "I love this!", "sentiment": "positive"},
    {"text": "This is terrible", "sentiment": "negative"},
    {"text": "It's okay I guess", "sentiment": "neutral"}
]

result = classify("This is amazing!", demos=demos)
# Demos are automatically included in the prompt
```

### Assistant Prefill

Guide the LLM's response format by prefilling the assistant message:

```python
@ai(
    messages=[
        {"role": "system", "content": "You are concise"},
        {"role": "user", "content": "{text}"},
        {"role": "assistant", "content": "Here's my analysis in 5 words:"}
    ]
)
def concise_analysis(text: str) -> str:
    """Brief analysis"""
    return _ai

result = concise_analysis("The economy is growing steadily.")
# "Strong fundamentals drive sustained expansion."
```

### Chat Mode

```python
from elemai import Chat

chat = Chat(system="You are a helpful assistant")

chat("My name is Alice")
# > "Hello Alice! How can I help you today?"

chat("What's my name?")
# > "Your name is Alice."
```

### Configuration

```python
from elemai import configure, set_config

# Global config
set_config(model="claude-opus-4-20250514", temperature=0.3)

# Context override
with configure(model="claude-haiku-4-20250514", temperature=0):
    result = some_task(input)

# Per-function config
@ai(model="claude-opus-4-20250514", temperature=0)
def precise_task(input: str) -> str:
    return _ai
```

## Design Philosophy

### 1. Messages Are the Template

No abstraction layers - use the standard message format everyone knows:

```python
@ai(
    messages=[
        {"role": "system", "content": "{instruction}"},
        {"role": "user", "content": "{inputs()}"}
    ]
)
def task(text: str) -> str:
    return _ai
```

### 2. Template Functions for Control

Use Python functions to control rendering:

```python
{inputs()}                    # All inputs, auto-formatted
{inputs(style='yaml')}        # YAML format
{outputs(style='schema')}     # JSON schema
{inputs(only=['text'])}       # Subset of inputs
```

### 3. Progressive Disclosure

Start simple, add complexity only when needed:

```python
# Beginner - just works
@ai
def task(text: str) -> str:
    """Do something"""
    return _ai

# Intermediate - add reasoning
@ai
def task(text: str) -> str:
    thinking: str = _ai["Reason through this"]
    return _ai

# Advanced - full control
@ai(
    messages=[
        {"role": "system", "content": "You are an expert"},
        {"role": "user", "content": "{inputs(style='yaml')}"},
        {"role": "assistant", "content": "Analysis:\n\n"}
    ]
)
def task(text: str) -> Analysis:
    thinking: str = _ai
    draft: str = _ai
    return _ai
```

### 4. Custom Template Functions

Define your own rendering logic:

```python
def render_with_emphasis(inputs: dict, highlight: list[str] = None) -> str:
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
        {"role": "user", "content": "{render_with_emphasis(inputs, highlight=['text'])}"}
    ]
)
def highlighted_analysis(text: str, context: str) -> str:
    """Analysis with highlighted inputs"""
    return _ai

# Register the custom function
highlighted_analysis.template.register_function('render_with_emphasis', render_with_emphasis)
```

## Examples

See the documentation for comprehensive examples:

- `examples/basic_usage.qmd` - Simple tasks, chat, configuration, structured outputs
- `examples/advanced_usage.qmd` - Custom templates, multi-step reasoning, pipelines, optimization
- `docs/design_vision.qmd` - Complete design philosophy and vision

## Inspection & Debugging

```python
@ai
def task(text: str) -> str:
    return _ai

# See the messages template
print(task.messages)

# See rendered prompt with actual values
print(task.render(text="example"))

# Preview what will be sent to the LLM
preview = task.preview(text="example")
print(preview.prompt)      # Full prompt
print(preview.messages)    # API messages
print(preview.config)      # Model config
```

## Supported Providers

elemai uses [litellm](https://github.com/BerriAI/litellm) as its backend, giving you access to **100+ LLM providers** including:

- Anthropic (Claude)
- OpenAI (GPT-4, GPT-4o, GPT-3.5)
- Google (Gemini)
- Groq
- Cohere
- Azure OpenAI
- AWS Bedrock
- And many more!

Just specify the model name:

```python
@ai(model="gpt-4o")
def task(text: str) -> str:
    return _ai

@ai(model="gemini/gemini-2.0-flash-exp")
def task(text: str) -> str:
    return _ai

@ai(model="groq/llama-3.3-70b-versatile")
def task(text: str) -> str:
    return _ai
```

## Key Features

### Structured Outputs
- Full Pydantic model support
- Dataclass support with automatic schema generation
- Automatic JSON parsing and validation

### Few-Shot Learning
- `demos` parameter for in-context learning
- Automatic integration into templates
- Works with all template types

### Conditional Templates
- Ternary expressions: `{var ? "yes" : "no"}`
- Short form: `{var ? "yes"}` (empty string if false)
- Nested variable access: `{inputs.warning ? "Alert!" : ""}`

### Multi-Step Reasoning
- Intermediate output fields with descriptions
- Automatic field detection and extraction
- `all=True` parameter to access all outputs

## Development

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

## License

MIT

## Contributing

Contributions welcome! Please see issues for planned features.

## Inspiration

elemai is inspired by:

- **functai** - Function-as-prompt philosophy
- **claudette/fastai** - Sensible defaults, progressive disclosure
- **dspy** - Structured prompting and optimization
- **ggplot2/dplyr** - Composable, layered design
