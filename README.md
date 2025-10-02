# elemai

**elemai** is a productive, ergonomic Python library for working with LLMs. It provides a clean interface that feels like writing normal Python functions, while giving you full control over prompts and message construction.

## Features

- 🎯 **Function-as-prompt**: Define AI tasks as regular Python functions
- 📝 **Direct message control**: Use standard OpenAI/Anthropic message format
- 🔧 **Template functions**: Programmatic control over prompt rendering
- 💬 **Stateful chat**: Easy conversational interfaces
- ⚙️ **Flexible configuration**: Global config with context overrides
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

### With Intermediate Reasoning

```python
@ai
def analyze(text: str) -> str:
    """Analyze the sentiment and themes"""
    thinking: str = _ai["Think through the emotional tone"]
    themes: str = _ai["Identify key themes"]
    return _ai

result = analyze("Product review text...")
```

### Structured Output

```python
from pydantic import BaseModel

class Analysis(BaseModel):
    sentiment: str
    confidence: float
    themes: list[str]

@ai
def deep_analysis(text: str) -> Analysis:
    """Perform comprehensive analysis"""
    return _ai

result = deep_analysis("Text to analyze...")
print(result.sentiment)  # Access structured fields
```

### Custom Message Template

```python
@ai(
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Analyze: {text}"},
        {"role": "assistant", "content": "Let me analyze step by step:\n\n"}
    ]
)
def custom_analysis(text: str) -> str:
    """Analysis with prefilled assistant message"""
    return _ai
```

### Template Functions

```python
@ai(
    messages=[
        {
            "role": "system",
            "content": "Task: {instruction}\n\nOutputs:\n{outputs(style='schema')}"
        },
        {"role": "user", "content": "{inputs(style='yaml')}"}
    ]
)
def structured_task(text: str, context: str) -> Analysis:
    """Automatic input/output formatting"""
    return _ai
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
set_config(model="opus", temperature=0.3)

# Context override
with configure(model="haiku", temperature=0):
    result = some_task(input)

# Per-function config
@ai(model="opus", temperature=0)
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
        {"role": "user", "content": "{inputs()}"},
    ]
)
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
# Beginner
@ai
def task(text: str) -> str:
    """Do something"""
    return _ai

# Intermediate
@ai
def task(text: str) -> str:
    thinking: str = _ai["Reason through this"]
    return _ai

# Advanced
@ai(messages=custom_messages, model="opus")
def task(text: str) -> Analysis:
    thinking: str = _ai
    draft: str = _ai
    return _ai
```

## Examples

See the `examples/` directory for comprehensive examples:

- `basic_usage.py` - Simple tasks, chat, configuration
- `advanced_usage.py` - Custom templates, multi-step reasoning, pipelines

## Inspection & Debugging

```python
@ai
def task(text: str) -> str:
    return _ai

# See the template
print(task.template.messages)

# See rendered prompt
print(task.render(text="example"))

# See actual messages
print(task.to_messages(text="example"))

# Full preview
preview = task.preview(text="example")
print(preview.prompt)
print(preview.config)
```

## Supported Providers

elemai uses [litellm](https://github.com/BerriAI/litellm) as its backend, giving you access to **100+ LLM providers** including:

- Anthropic (Claude) - default
- OpenAI (GPT-4, GPT-4o, GPT-3.5)
- Google (Gemini)
- Cohere
- Azure OpenAI
- AWS Bedrock
- And many more!

Just use the model name and litellm handles the rest:

```python
set_config(model="gpt-4-turbo")
set_config(model="gemini-pro")
set_config(model="command-nightly")
```

## Model Aliases

Convenient shortcuts for the latest models (as of 2025):

```python
# Claude 4 (latest)
set_config(model="sonnet")      # claude-sonnet-4-20250514
set_config(model="opus")        # claude-opus-4-20250514
set_config(model="haiku")       # claude-3-5-haiku-20241022

# Claude 3.7 & 3.5
set_config(model="sonnet-3.7")  # claude-3-7-sonnet-20250219
set_config(model="sonnet-3.5")  # claude-3-5-sonnet-20241022

# OpenAI
set_config(model="gpt4o")       # gpt-4o
set_config(model="gpt4o-mini")  # gpt-4o-mini
set_config(model="gpt4")        # gpt-4-turbo

# Google Gemini
set_config(model="gemini-pro")  # gemini-2.5-pro
set_config(model="gemini-flash") # gemini-2.5-flash
```

## Development

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

Format code:

```bash
black .
isort .
```

## License

MIT

## Contributing

Contributions welcome! Please see issues for planned features.

## Inspiration

elemai is inspired by:

- **claudette/fastai** - Sensible defaults, progressive disclosure
- **functai** - Function-as-prompt philosophy
- **dspy** - Structured prompting as first-class
- **ggplot2/dplyr** - Composable, layered design
