# user 1

So a productive life library would have to go from having maximal useful defaults. And then if you have the maximum the most Like, if you find The set of defaults That's are the most likely the most, common. One that you would want to set by end, You would have, by definition, the most generally productive. And that may not work for every body and thus you may need a set of default different for different But at each kind of layer of the user stack, if you find the most common default and For for your group. Fuser you will have by definition the most productive because then people would only need to enter whatever is not default and automatizable so the goal would be to do that. And so the the the first two scenarios that I can see a user wanting to use LRM. Is one where they want to chat. And the other one is where they want to have it do a task, be a program, be a function. And when you want to chat, you would want It have a default model a default personality, default way of answering everything. And you would just want to say, to have a very, very brief way of kind of getting into the the chat element. And then give it a giving a string Your chat and then receiving a result as as a string. If you want chat, you want it to be stateful. You want it to have its own conversation. And it will quickly evolve into I wanted to have a different personality, different system prompt. I want to manage history. State, want to give it tools. Or not. I want to do permission stuff I want to do layers of Like like acceptance stuff. Configurations of, like, what it can and cannot do in access and cannot access. But that is only true if you don't trust that the default is or the model is good. Perfectly good.

# assistant 1

# Designing a First-Principles LLM Library

Let me think through this systematically, building from your core insight about **maximal useful defaults**.

## Core Principle: Progressive Disclosure of Complexity

The most productive library would have:
1. **The shortest path to value** for the most common use case
2. **Graceful complexity expansion** as needs evolve
3. **Sensible defaults that rarely need changing**

## Two Fundamental Modes

You've identified the two essential patterns:

### Mode 1: **Chat** (Stateful Conversation)
- Maintains history automatically
- Interactive, human-in-loop
- Exploratory, iterative

### Mode 2: **Task** (Functional/Programmatic)
- Stateless by default (or explicit state)
- Deterministic, repeatable
- Production-oriented

## The Ideal API Surface

```python
# CHAT MODE - Absolute minimum
from llm import chat

chat("What's the weather like?")  # Uses best default model, returns string
chat("And tomorrow?")  # Maintains context automatically

# That's it. For 80% of use cases.

# TASK MODE - Function-like
from llm import ask

result = ask("Summarize this: {text}")  # Stateless, repeatable
```

## Progressive Enhancement

```python
# When you need more control, add ONLY what you need:

# Different personality
chat = Chat(system="You are a pirate")
chat("Hello")

# Different model
chat = Chat(model="opus")  # Short aliases for common models

# Tools (auto-discovers from functions)
def search(query: str) -> str:
    """Search the web"""
    return results

chat = Chat(tools=[search])  # That's it - tools work

# Or even simpler - just decorate
@chat.tool
def search(query: str) -> str:
    """Search the web"""
    return results
```

## Key Design Insights

### 1. **Convention over Configuration**
- Model selection: `"sonnet"` not `"claude-sonnet-4-20250929"`
- System prompts: Named presets `system="concise"` alongside custom strings
- History: Automatic unless you explicitly don't want it

### 2. **String-First Philosophy** (like ggplot2's aesthetics)
```python
# Input and output are strings by default
response = chat("message")  # Returns string
print(response)  # Just works

# But can get structured when needed
response = chat("message", return_type="full")  # Returns rich object
print(response.text)
print(response.usage)
```

### 3. **Composable Defaults** (like dplyr's pipes)
```python
# Each operation has smart defaults but composes
chat = (
    Chat()
    .with_system("Be concise")
    .with_model("sonnet")
    .with_memory(limit=10)  # Only keep last 10 exchanges
)

# Or even more pipe-like
result = (
    ask("Summarize {text}")
    .with_structured_output(SummarySchema)
    .with_retry(3)
    .run(text=long_document)
)
```

### 4. **Smart Tool Integration**
```python
# Tools should be EASY
from llm import Chat
import requests

chat = Chat()

# Just pass functions - docstrings become descriptions
chat.add_tool(requests.get)  # Automatically wrapped
chat("Get the weather from api.weather.com")  # Just works

# Or for more control
@chat.tool(name="web_search", confirm=True)  # Asks before running
def search(query: str) -> str:
    """Search the web for information"""
    return results
```

### 5. **Sensible Safety Defaults**
```python
# Safety should be default, opt-out when trusted
chat = Chat(
    # These are DEFAULT:
    tool_approval="auto_safe",  # Safe tools run automatically
    dangerous_tools=["file_write", "shell"],  # Require confirmation
    max_cost=1.00,  # Dollar limit per conversation
)

# For production/trusted environments
task = Ask(trust="full")  # No confirmations
```

## The "Layers" Concept

Different user sophistication levels get different entry points:

```python
# Layer 1: Beginner - "Just chat"
from llm import chat
chat("Hello")

# Layer 2: Intermediate - "I need control"
from llm import Chat
c = Chat(model="opus", system="Be brief")
c("Hello")

# Layer 3: Advanced - "I'm building production systems"
from llm import Client, Message, Tool, Schema
client = Client(
    model=ModelConfig(name="opus", temperature=0.7),
    tools=ToolRegistry([custom_tool]),
    middleware=[logging, caching, retry],
)
```

## Inspiration Synthesis

From **claudette/fastai**: 
- Sensible defaults, minimal boilerplate
- Progressive disclosure

From **dspy**:
- Structured prompting as first-class
- Optimizable components

From **baml**:
- Type safety when you want it
- Schema-driven outputs

From **ggplot2**:
- Layered grammar (add components with +)
- Aesthetic mappings (smart defaults)

From **dplyr**:
- Pipeable operations
- Lazy evaluation where useful

## The Core Innovation

**The library should feel like talking to an LLM, not programming one.**

```python
# This should feel natural:
chat("Hey, can you search the web for me?")
# > "I'll need a search function. Do you want to add one?"

chat.add_tool(web_search)
chat("Now try")
# > Just works

# Not this:
client = LLMClient(api_key=..., model=...)
conversation = Conversation(system_prompt=...)
tool_def = ToolDefinition(name=..., description=..., parameters=...)
response = client.chat(conversation.messages, tools=[tool_def])
```

The most productive library is one where you're **writing what you mean**, not **configuring an abstraction**.

Does this resonate with your vision? What aspects would you want to explore deeper?


# user 2

Yes. This is very good. Now I want to talk about the Ask 1 and the task type of thing. So you know, when you want and then let him to do a task What I've discovered is that you essentially want it to be kinda magically doing as if you had a programming function, but then it's done by the element. It's not programmed. And so You want to define the inputs to that function, but you want that to be minimal. Like, let's say that I have a functional that I give Input. I I give an input, and I want the transcription output. I'm it would be wrong to be doing video input, audio input, whereas I could just be doing input, and it's just happens to take video and audio. Like, you put the input and it transcribes it no matter what. Right. This level of specification of input has to not lead to like, 81 City wants to give you a text. And to give you a schema. And you have to extract that. I would want them to give me the input field text input and then the schema desired. And then the output field would just be the filling up of the schema. That would be better than defining 7 functions each with a different name, that specify the output But sometimes, you know that you want exactly one output field. And so in that case, you would say that, you know, it has to be extracted as this output and then you'll be putting a lot of inputs. Another consideration is that L and M's needs to have to be like directed in their outputs, like, they have to, let's say, produce first some chain of thought, then then maybe the full program, like, the full they are asked, and then maybe a verification, then maybe a summary. Then they would output all those things, and we would want to be, like, pick up these things. And specify the structured output of some of these. And then sometimes we would want DNNM to do it kind of in a JSON format, sometimes in the XML format, in a in a YAML. Just free flowing decks and all those things. And I have worked with, I I have worked on a design already for, like, a templating language to create those things. So that the user just kinda specify the minimum thing that they want to specify and then the rest is picked up by the templates and the default. Of course, we want to be inspired here by DSPIRE where we can Do optimization using DSPI. The DSPI's optimizers and and more where we want to be doing optimizations on, demos, and like, like, few shots. Possibly, and also possibly optimization on an instruction field. But I think we need also a system field. Where we're gonna not train it and we need the user to be able to So if I If an input description or field is trainable or not, Like, can we do optimization where we drop the intermediary output or not? Can we do those type of things? Also, I want it to be defined like, I want the user to be able to define that as simply as possible. And, like, it should be familiar to, like, kind of high school students that knows how to do like, they know about functions. Their life already. But then also, like, you know, as we were just discussing, you know, like, grow a lot and have, like, possible optimization, training sets, splitting, evaluation, metrics, kind of Automated optimization of parsing code, automated all all of all of those things. Like, As powerful as d s 5, but the ergonomics should be more natural. Like, it shouldn't be subclassing of classes to define a signature. It shouldn't be signature in a string. It shouldn't make the user think too much that the input fields are important because in the end, it's just a big Yeah. And it should be cohesive. We should not have to do string wrangling... we want to be very declarative and in code. Maybe also stufldy functai on MaximeRivest githubs... could inspir

# ai 2

# The Task/Ask API: From First Principles

You're describing something profound: **LLMs as programmable functions where the "code" is learned, not written.**

## Core Insight: Inputs Are Just Inputs

```python
# Bad: Over-specification
transcribe_audio(audio_file)
transcribe_video(video_file)

# Good: Generic, smart
transcribe(input)  # Works with audio OR video
```

The library should **infer intent from data**, not force explicit typing.

## The Minimal Task Definition

```python
from llm import task

# Simplest possible - high school function style
@task
def transcribe(input):
    """Transcribe the audio/video to text"""
    # That's it. Returns string by default.

result = transcribe("video.mp4")
```

## Progressive Enhancement

### Level 1: Add Structure
```python
from llm import task
from pydantic import BaseModel

class Summary(BaseModel):
    title: str
    key_points: list[str]
    sentiment: str

@task
def summarize(text) -> Summary:
    """Summarize the given text"""
    # Return type hint = automatic structured output

summary = summarize(long_text)
print(summary.title)
```

### Level 2: Multiple Outputs (Directed Thinking)
```python
@task(
    outputs={
        "reasoning": "text",  # Chain of thought
        "code": "python",     # The actual code
        "tests": "python",    # Test cases
        "summary": "text"     # Brief explanation
    }
)
def generate_function(description):
    """Generate a Python function from description"""

result = generate_function("fibonacci sequence")
print(result.reasoning)  # Access any output
print(result.code)       # Run the code
```

### Level 3: Template-Driven (Your Innovation)

```python
@task(template="""
Think step by step:
<thinking>
{reasoning}
</thinking>

<code language="python">
{code}
</code>

<verification>
{verification}
</verification>

<summary>
{summary}
</summary>
""")
def solve_problem(problem):
    """Solve a programming problem"""
    # Template defines structure
    # Outputs automatically extracted
```

## Format Flexibility

```python
@task(format="xml")  # or "json", "yaml", "text"
def extract(text, schema: ExtractSchema):
    """Extract structured data"""

# Or per-output format
@task(outputs={
    "thinking": {"format": "text"},
    "data": {"format": "json", "schema": DataSchema},
    "summary": {"format": "markdown"}
})
def analyze(input):
    """Analyze data with different output formats"""
```

## The Template System

**Key insight**: Users specify **what varies**, templates handle **what's standard**.

```python
# User writes minimal code
@task
def extract(text, schema):
    """Extract {schema.__name__} from text"""

# Library expands to:
"""
You are extracting structured data.

INPUT:
{text}

SCHEMA:
{schema.model_json_schema()}

INSTRUCTIONS:
Extract all relevant information matching the schema.
Output valid JSON only.

OUTPUT:
"""
```

### Custom Templates
```python
from llm import task, Template

# Define reusable template
chain_of_thought = Template("""
{system}

INPUT:
{input_fields}

Think through this step by step:
1. Understanding: <thinking>{reasoning}</thinking>
2. Solution: <answer>{answer}</answer>
3. Confidence: <confidence>{confidence}</confidence>

{format_instructions}
""")

@task(template=chain_of_thought)
def analyze(data, question):
    """Analyze data to answer question"""
```

## DSPy-Style Optimization (But Ergonomic)

```python
@task(
    trainable=True,  # Enable optimization
    optimize={
        "demos": True,        # Learn from examples
        "instructions": True, # Optimize prompt wording
        "reasoning": "optional"  # Can drop intermediate outputs
    }
)
def classify(text) -> Category:
    """Classify text into categories"""

# Training
train_set = [
    ({"text": "I love this!"}, Category.POSITIVE),
    ({"text": "Terrible"}, Category.NEGATIVE),
]

# Optimize automatically
classify.train(
    train_set,
    metric="accuracy",
    optimizer="mipro"  # DSPy optimizer
)

# Or manual control
from llm.optimize import Optimizer

opt = Optimizer(
    metric=accuracy,
    train_split=0.8,
    val_split=0.2
)

optimized_classify = opt.optimize(
    classify,
    dataset=train_set,
    budget=100  # Max optimization runs
)
```

## Field-Level Control

```python
@task(
    fields={
        "system": {
            "trainable": False,  # Don't optimize system prompt
            "value": "You are an expert analyst"
        },
        "reasoning": {
            "trainable": True,   # Can be removed if not helpful
            "required": False
        },
        "input": {
            "description": "The text to analyze",  # Used in prompt
            "trainable": True    # Few-shot examples can be optimized
        }
    }
)
def analyze(input, schema):
    """Analyze input according to schema"""
```

## Unified Design: No String Wrangling

```python
# Bad (DSPy current)
class Summarize(dspy.Signature):
    """Summarize text"""
    text = dspy.InputField()
    summary = dspy.OutputField()

# Better (our design)
@task
def summarize(text) -> str:
    """Summarize text"""
    # Uses docstring, type hints, names - all native Python
```

## Complete Example: The Power

```python
from llm import task
from pydantic import BaseModel

class Analysis(BaseModel):
    sentiment: str
    key_themes: list[str]
    action_items: list[str]

@task(
    # Execution config
    model="sonnet",
    temperature=0.3,
    
    # Optimization config
    trainable=True,
    optimize=["demos", "instructions"],
    
    # Output structure
    outputs={
        "thinking": "text",
        "analysis": Analysis
    },
    
    # Template
    template="""
Analyze this text carefully.

TEXT:
{text}

REASONING:
<thinking>
{thinking}
</thinking>

STRUCTURED ANALYSIS:
{analysis}
"""
)
def analyze_feedback(text) -> Analysis:
    """Analyze customer feedback for insights"""
    pass  # Implementation is the prompt!

# Use it
result = analyze_feedback("The product is great but shipping took forever")
print(result.thinking)
print(result.analysis.sentiment)

# Train it
dataset = load_feedback_examples()
analyze_feedback.train(dataset, metric="f1")

# Now it's optimized!
```

## The Magic: Compilation

Behind the scenes:

```python
# User writes
@task
def classify(text) -> Category:
    """Classify text"""

# Library compiles to:
"""
SYSTEM: You are a classification system.

TASK: Classify text

INPUT:
Text: {text}

INSTRUCTIONS:
[Optimized through training if enabled]

OUTPUT FORMAT:
{Category.schema()}

OUTPUT:
"""
```

## Inspired by fastai: Callbacks & Hooks

```python
@task(
    callbacks=[
        LoggingCallback(),
        CachingCallback(),
        RetryCallback(max_attempts=3)
    ]
)
def unreliable_task(input):
    """Might fail, will retry"""

# Or use decorators (fastai style)
@task
@retry(3)
@cache(ttl=3600)
@log
def my_task(input):
    """Clean, composable"""
```

## Automatic Parsing & Validation

```python
@task(
    parse={
        "auto_fix": True,      # Try to fix invalid outputs
        "fallback": "text",    # If structured parsing fails
        "validate": True       # Pydantic validation
    }
)
def extract(text) -> ComplexSchema:
    """Extract complex structured data"""
    # Handles parsing failures gracefully
```

## The Cohesive Experience

```python
# Everything works together
from llm import task, Chat

# Define task
@task(trainable=True)
def answer(question, context) -> str:
    """Answer question given context"""

# Use in chat
chat = Chat()
chat.add_task(answer, trigger="when user asks a question")

# Or standalone
result = answer("What is X?", context=documents)

# Or in a pipeline
from llm import Pipeline

pipeline = Pipeline([
    task1,
    task2.with_config(model="opus"),
    task3
])

result = pipeline.run(input)
```

## Key Design Principles

1. **Declarative, not imperative**: Describe what you want, not how to get it
2. **Type hints as schema**: Native Python, no special syntax
3. **Docstrings as instructions**: Natural documentation = prompts
4. **Progressive complexity**: Start simple, grow naturally
5. **Optimization as option**: Training is opt-in, not required
6. **Format agnostic**: XML, JSON, YAML, text - whatever works
7. **Template-driven**: Reusable patterns, not string wrangling
8. **Field-level control**: Granular optimization settings
9. **Automatic extraction**: Templates + regex + LLM parsing
10. **Production-ready defaults**: Retries, caching, validation built-in

Does this capture your vision? Should we dive into the template engine design or the optimization layer next?


# user 3

I don't want the user to have to, do the XML stuff like it's a lot of typing. Not very ergonomic. I think we can, do that a bit nicer for them. And, we should have some default templates that will work great majority of the time. But then when in dozen, Alright. Templating language that was used to make the defaults can also be used by the users to make their own kind of Message. From template.

# ai 3

# Ergonomic Templates: Zero Boilerplate, Maximum Power

You're absolutely right. The XML example was too verbose. Let's design this properly.

## The Core Idea

**Default templates handle 95% of cases. When they don't, use the same simple system to make your own.**

## Zero-Config Default

```python
@task
def analyze(text) -> Analysis:
    """Analyze customer feedback"""
    # That's it. Library handles everything.

# Behind the scenes, uses smart default template
```

## What the Default Template Does

```python
# Library's internal default (user never sees this)
DEFAULT_TEMPLATE = """
{system_prompt}

{inputs}

{chain_of_thought_section}

{output_instructions}
"""
```

The library **automatically**:
- Adds chain-of-thought if output is complex
- Picks JSON/XML/YAML based on output type
- Structures inputs clearly
- Adds examples if few-shot demos exist

## When You Need More: Simple Override

```python
from llm import task, section

@task(
    thinking=True,  # Add reasoning section
    format="json"   # Force JSON output
)
def analyze(text) -> Analysis:
    """Analyze customer feedback"""
```

That's it. No XML typing.

## Intermediate: Named Outputs

```python
@task(outputs=["thinking", "answer", "confidence"])
def solve(problem):
    """Solve a math problem
    
    Think through the problem, provide the answer, 
    and rate your confidence 1-10.
    """

result = solve("What is 15% of 80?")
print(result.thinking)   # Auto-extracted
print(result.answer)     # Auto-extracted  
print(result.confidence) # Auto-extracted
```

Behind the scenes, library creates sections automatically:
- Detects output names
- Creates appropriate delimiters
- Extracts with regex + LLM fallback

## The Template Language: When You Need It

**Inspired by Jinja2 but simpler, LLM-aware**

```python
from llm import task, template

# Define custom template
my_template = template("""
{% system %}
You are an expert analyst.
{% endsystem %}

{% for input in inputs %}
{{ input.name.upper() }}: {{ input.value }}
{% endfor %}

{% thinking %}
Think step by step.
{% endthinking %}

{% output schema=output_type format="json" %}
""")

@task(template=my_template)
def analyze(text, context) -> Analysis:
    """Analyze with custom structure"""
```

## But Even Simpler: Sections API

**Most users should never write templates. Use sections.**

```python
from llm import task, Section

@task(
    sections=[
        Section.system("You are an expert"),
        Section.inputs(),  # Auto-formats all inputs
        Section.thinking(),  # Adds CoT
        Section.output(format="json")
    ]
)
def analyze(text) -> Analysis:
    """Analyze text"""
```

Or even shorter with presets:

```python
@task(preset="reasoning")  # Includes system + inputs + thinking + output
def analyze(text) -> Analysis:
    """Analyze text"""
```

## Preset Templates (Built-in)

```python
# Available presets
presets = {
    "simple": [Section.inputs(), Section.output()],
    
    "reasoning": [
        Section.inputs(),
        Section.thinking(),
        Section.output()
    ],
    
    "structured": [
        Section.inputs(),
        Section.thinking(),
        Section.draft(),
        Section.critique(),
        Section.output()
    ],
    
    "extract": [
        Section.system("You extract structured data accurately"),
        Section.inputs(),
        Section.output(strict=True)
    ],
    
    "creative": [
        Section.system("You are creative and original"),
        Section.inputs(),
        Section.brainstorm(),
        Section.draft(),
        Section.output()
    ]
}

# Use them
@task(preset="structured")
def write_essay(topic) -> Essay:
    """Write an essay"""
```

## The Template Language (for power users)

**Key principle: Minimal syntax, maximum clarity**

```python
from llm import template

custom = template("""
{# Comments are ignored #}

{% system %}
{{ custom_system_prompt }}
{% endsystem %}

{# Auto-loop through inputs #}
{% inputs %}

{# Named output sections #}
{% section "analysis" %}
Analyze the input carefully.
{% endsection %}

{% section "summary" %}
Summarize in one sentence.
{% endsection %}

{# Structured output #}
{% output schema=output_type %}
""")
```

### Template Features

```python
# 1. Conditionals
template("""
{% if has_examples %}
Here are examples:
{% for example in examples %}
- {{ example }}
{% endfor %}
{% endif %}

{% inputs %}
{% output %}
""")

# 2. Variables
template("""
{% system %}
You are a {{ role }}. {{ additional_instructions }}
{% endsystem %}

{% inputs %}
{% output %}
""")

# 3. Sections with options
template("""
{% inputs style="markdown" %}

{% thinking optional=True %}

{% output format="json" strict=True %}
""")

# 4. Custom delimiters
template("""
{% inputs %}

### REASONING ###
{{ reasoning }}
### END REASONING ###

{% output %}
""")
```

## Smart Defaults: The Magic

The library analyzes your task and picks the best default:

```python
@task
def classify(text) -> Category:
    """Classify text"""
    # Simple classification → simple template

@task  
def analyze(text) -> DetailedAnalysis:
    """Deep analysis of text"""
    # Complex output → adds thinking section

@task
def creative_write(prompt) -> Story:
    """Write a creative story"""  
    # Creative task → uses creative preset
```

## Section Composition

```python
from llm import Section

# Build custom pipelines easily
@task(
    sections=[
        Section.system("You are helpful"),
        Section.inputs(style="numbered"),
        Section.custom("""
        First, think about the problem:
        {{ thinking }}
        """),
        Section.output(format="yaml")
    ]
)
def my_task(input):
    """Custom structured task"""
```

## The Power: Reusable Section Library

```python
from llm import Section

# Define once
expert_system = Section.system("""
You are an expert with 20 years of experience.
You think carefully and thoroughly.
""")

cot_section = Section.thinking("""
Break down the problem:
1. Understanding: {{ understanding }}
2. Approach: {{ approach }}  
3. Solution: {{ solution }}
""")

# Reuse everywhere
@task(sections=[expert_system, Section.inputs(), cot_section, Section.output()])
def task1(input):
    """First task"""

@task(sections=[expert_system, Section.inputs(), cot_section, Section.output()])
def task2(input):
    """Second task"""

# Or make a preset
my_preset = [expert_system, Section.inputs(), cot_section, Section.output()]

@task(sections=my_preset)
def task3(input):
    """Third task"""
```

## Output Extraction: Automatic

The library handles extraction without user intervention:

```python
@task(outputs=["thinking", "answer"])
def solve(problem):
    """Solve problem"""

# Library automatically:
# 1. Creates delimited sections in prompt
# 2. Extracts with regex first (fast)
# 3. Falls back to LLM parsing if needed
# 4. Returns structured result

result = solve("2+2")
result.thinking  # Extracted automatically
result.answer    # Extracted automatically
```

## Configuration Layers

```python
# Global defaults
from llm import config

config.set_default_template("reasoning")
config.set_default_model("sonnet")

# Then all tasks use these unless overridden
@task
def my_task(input):
    """Uses global defaults"""

@task(template="simple", model="opus")  # Override
def other_task(input):
    """Uses custom settings"""
```

## The Complete Picture

```python
# Beginner: Just works
@task
def summarize(text) -> str:
    """Summarize text"""

# Intermediate: Named outputs  
@task(outputs=["thinking", "summary"])
def summarize(text) -> str:
    """Summarize with reasoning"""

# Intermediate: Presets
@task(preset="reasoning")
def summarize(text) -> Summary:
    """Structured summary with reasoning"""

# Advanced: Section composition
@task(
    sections=[
        Section.system("Be concise"),
        Section.inputs(),
        Section.thinking(),
        Section.output(format="json")
    ]
)
def summarize(text) -> Summary:
    """Full control"""

# Expert: Custom template
@task(template=my_custom_template)
def summarize(text) -> Summary:
    """Maximum flexibility"""
```

## Implementation: How Defaults Work

```python
class Task:
    def __init__(self, fn, template=None, preset=None, sections=None, outputs=None):
        if template:
            self.template = template
        elif sections:
            self.template = build_template_from_sections(sections)
        elif preset:
            self.template = PRESETS[preset]
        elif outputs:
            # Auto-build template with named outputs
            self.template = auto_template_with_outputs(outputs)
        else:
            # Smart default based on function signature
            self.template = infer_template(fn)
    
    def infer_template(self, fn):
        """Pick best template based on task characteristics"""
        sig = inspect.signature(fn)
        return_type = sig.return_annotation
        
        # Simple output → simple template
        if return_type == str:
            return PRESETS["simple"]
        
        # Complex structured output → reasoning template
        if is_pydantic_model(return_type):
            return PRESETS["reasoning"]
        
        # Creative task (based on docstring/name)
        if is_creative_task(fn):
            return PRESETS["creative"]
        
        return PRESETS["simple"]
```

## Key Principles

1. **Zero config for 95% of cases** - smart defaults just work
2. **Progressive disclosure** - add complexity only when needed
3. **No boilerplate** - no XML/JSON writing by users
4. **Reusable components** - sections compose like LEGO
5. **Template language for experts** - powerful but optional
6. **Automatic extraction** - outputs parsed automatically
7. **Preset library** - common patterns built-in

Does this feel right? Should we detail the Section API more, or move to the optimization layer?

# user 4

Now this doesn't feel right. Search online for functAI.

# ai 4

Ah! Now I see it. FunctAI's core philosophy is "The function definition is the prompt, and the function body is the program definition" - this is brilliant and exactly what you're getting at.

Let me look at the actual code:

Perfect! Now I understand completely. This is SO much cleaner. Let me redesign with this philosophy:

# The FunctAI Way: Function-is-the-Prompt

The key insight is that the function definition IS the prompt, and the function body IS the program definition. No templates, no XML, no boilerplate.

## The Core Pattern

```python
from llm import ai, _ai

@ai
def summarize(text: str) -> str:
    """Summarize the text in one concise sentence"""
    return _ai

# That's it. Just call it.
result = summarize("Long text here...")
```

## Multi-Step Reasoning (No Templates Needed!)

```python
@ai
def solve_problem(question: str) -> float:
    """Solve a math word problem"""
    
    # Variable names + type hints + descriptions = automatic structure
    reasoning: str = _ai["Think through this step by step"]
    approach: str = _ai["Explain your approach"]
    
    # Final answer
    return _ai

# Normal call returns just the answer
answer = solve_problem("If a train travels 120 miles in 2 hours, what is its speed?")

# Get ALL the intermediate steps
full = solve_problem("...", all=True)
print(full.reasoning)
print(full.approach)
print(full.result)
```

**This is brilliant** - the variable names become field names, type hints define output types, and subscript notation provides instructions.

## Structured Outputs (Zero Config)

```python
from dataclasses import dataclass

@dataclass
class Analysis:
    sentiment: str
    themes: list[str]
    action_items: list[str]

@ai
def analyze(text: str) -> Analysis:
    """Analyze customer feedback for insights"""
    
    # Optional: add reasoning
    thinking: str = _ai["Consider the key aspects"]
    
    # Return structured - automatic parsing!
    return _ai

result = analyze("The product is great but shipping was slow")
print(result.sentiment)
print(result.themes)
```

## Multiple Outputs (Tuple Returns)

```python
@ai
def critique_and_improve(text: str) -> tuple[str, str, int]:
    """Analyze text and provide improvements"""
    
    critique: str = _ai["Constructive criticism"]
    improved: str = _ai["Improved version"]
    confidence: int = _ai["Confidence 1-10"]
    
    return critique, improved, confidence

# Unpack naturally
critique, improved, conf = critique_and_improve("U should fix this asap")
```

## Agentic Tools (Just Pass Functions!)

```python
def search_web(query: str) -> str:
    """Search the web"""
    return results

def calculate(expression: str) -> float:
    """Calculate math expressions"""
    return eval(expression)  # (use safe eval in production!)

@ai(tools=[search_web, calculate])
def research_assistant(question: str) -> str:
    """Answer questions using available tools"""
    return _ai

# Automatically does ReAct loop!
answer = research_assistant("What is (15 * 23) + 10?")
```

## Stateful Chat (One Argument!)

```python
@ai(stateful=True)
def assistant(message: str) -> str:
    """A friendly AI assistant"""
    return _ai

assistant("My name is Alex")
# > "Hello Alex! Nice to meet you."

assistant("What's my name?")
# > "Your name is Alex."

# History is automatic!
```

## Post-Processing (Mix AI + Code)

```python
@ai
def sentiment_score(text: str) -> float:
    """Returns sentiment score 0.0-1.0"""
    
    # Get AI output
    score = _ai
    
    # Apply code validation
    return max(0.0, min(1.0, float(score)))
```

## Optimization (DSPy Integration)

```python
from dspy import Example

@ai
def classify(query: str) -> str:
    """Classify as 'booking', 'cancel', or 'info'"""
    return _ai

# Training data
trainset = [
    Example(query="Reserve a room", result="booking").with_inputs("query"),
    Example(query="How do I get there?", result="info").with_inputs("query"),
]

# Optimize in place!
classify.opt(trainset=trainset)

# Now it has few-shot examples automatically
```

## Configuration Cascade

```python
from llm import configure

# Global config
configure(model="sonnet", temperature=0.7)

# Per-function override
@ai(model="opus", temperature=0)
def precise_task(input: str) -> str:
    """Needs deterministic output"""
    return _ai

# Context override
with configure(model="haiku"):
    result = some_function(input)  # Uses haiku temporarily
```

## Complex Pipeline Example

```python
@dataclass
class Invoice:
    number: str
    vendor: str
    total: float
    items: list[str]

@ai
def extract_invoice(text: str) -> Invoice:
    """Extract invoice from document"""
    thought: str = _ai["Analyze document structure first"]
    return _ai

@ai
def validate_invoice(invoice: Invoice) -> bool:
    """Check if invoice is valid"""
    return _ai

@ai
def summarize_invoice(invoice: Invoice) -> str:
    """Create human-readable summary"""
    return _ai

# Use it
invoice = extract_invoice(document_text)
if validate_invoice(invoice):
    print(summarize_invoice(invoice))
```

## Why This is Better

1. **No template syntax** - just Python
2. **No XML/JSON typing** - type hints do it
3. **No string wrangling** - docstrings are instructions
4. **Variable names are meaningful** - they become field names
5. **Post-processing is natural** - mix AI and code freely
6. **Tools are just functions** - no special wrapping
7. **Stateful is one flag** - not a class hierarchy
8. **Optimization is a method call** - not a separate system

## The Magic of `_ai`

`_ai` is a sentinel that represents the LLM output and behaves dynamically as if it were the expected return type:

```python
@ai
def extract_price(text: str) -> float:
    """Extract the price"""
    
    # _ai acts like a float because of return type
    price = _ai
    
    # Can do float operations on it
    return price * 1.1  # Add 10% tax
```

## For Your Library

The template system I proposed was **way too complex**. Instead:

```python
# Beginner - just works
@ai
def task(input: str) -> str:
    """Do the thing"""
    return _ai

# Intermediate - add reasoning
@ai
def task(input: str) -> Output:
    """Do the thing"""
    reasoning: str = _ai["Think step by step"]
    return _ai

# Advanced - multiple outputs
@ai
def task(input: str) -> tuple[str, str]:
    """Do the thing"""
    thinking: str = _ai["Analyze the input"]
    draft: str = _ai["Create initial version"]
    critique: str = _ai["Critique the draft"]
    final: str = _ai["Final improved version"]
    return thinking, final

# Expert - with tools
@ai(tools=[tool1, tool2])
def task(input: str) -> str:
    """Do the thing"""
    research: list[str] = _ai["Gather information using tools"]
    return _ai
```

**Everything is just Python.** The "template" is the function body itself. Brilliant!

Should we now design the Chat mode with this same philosophy?


# user 5

Yes. This is very good, but also a problem there is that I lose all control over what is this rendered as into the this the strings and the in but JSON messages to The NNM inference And I want to have a way in our libraries to to be doing that kind of mapping, templating very easily. So I like the interface that you you've shown But I I want to, like, add a piece in between. This and the l n m call. So that it's controllable and it's not too magical. Also, the chat part of it, the statefulness part of it, I prefer what you have, designed and mapped out above. Would want a library that does both. But they they should kinda fit together, maybe, like, I I don't know. Maybe there's a little detail details we can adjust.

# ai 5

# Bridging the Gap: Ergonomics + Control

You're absolutely right. The FunctAI interface is beautiful, but it's **too magical** - you lose control over the prompt construction. We need **both**:

1. **Ergonomic interface** (FunctAI-style)
2. **Transparent, controllable templating** (your original insight)

## The Architecture

```
User Code (Python function)
        ↓
Introspection Layer (analyzes function)
        ↓
Template Layer (customizable, visible)
        ↓
Message Construction (OpenAI/Anthropic format)
        ↓
LLM Call
```

The key: **Make the middle layers accessible and overridable.**

## The Design

```python
from llm import ai, _ai, Template

# Simple case - uses smart defaults
@ai
def summarize(text: str) -> str:
    """Summarize text"""
    return _ai

# But you can inspect what it's doing
print(summarize.template)  # Shows the actual template being used
print(summarize.render(text="example"))  # Shows the actual prompt
```

## Controllable Templates

```python
from llm import ai, _ai, template

# Define your own template with full control
my_template = template("""
You are an expert analyst.

INPUT:
{text}

REASONING:
{reasoning}

OUTPUT:
{output}
""")

@ai(template=my_template)
def analyze(text: str) -> str:
    """Analyze text"""
    reasoning: str = _ai["Think step by step"]
    return _ai

# Now you control exactly how it's rendered
```

## The Template Object

```python
class Template:
    def __init__(self, template_string: str, format: str = "auto"):
        self.template = template_string
        self.format = format  # "text", "xml", "json", etc.
    
    def render(self, **fields) -> str:
        """Render template with fields"""
        return self.template.format(**fields)
    
    def to_messages(self, **fields) -> list[dict]:
        """Convert to OpenAI/Anthropic message format"""
        # This is the key method that bridges to LLM APIs
        pass
```

## Composable Template Pieces

```python
from llm import Section, template

# Build templates from reusable sections
system_section = Section.system("You are an expert")
input_section = Section.input("{text}", label="TEXT")
reasoning_section = Section.output("reasoning", "Think step by step")
answer_section = Section.output("answer")

# Compose them
my_template = template.from_sections([
    system_section,
    input_section,
    reasoning_section,
    answer_section
])

@ai(template=my_template)
def task(text: str) -> str:
    reasoning: str = _ai
    return _ai
```

## Auto-Generated Templates (Inspectable)

```python
@ai
def analyze(text: str) -> Analysis:
    """Analyze text"""
    thinking: str = _ai["Think carefully"]
    return _ai

# Library auto-generates template, but you can see it!
print(analyze.template)
"""
OUTPUT:
Template(
  system: "You are performing the task: analyze"
  instruction: "Analyze text"
  inputs: {
    "text": InputField(type=str)
  }
  outputs: {
    "thinking": OutputField(type=str, instruction="Think carefully"),
    "result": OutputField(type=Analysis)
  }
  format: "json"
)
"""

# And modify it
analyze.template.format = "xml"
analyze.template.system = "You are an expert analyst"

# Or replace entirely
analyze.template = my_custom_template
```

## The Render Pipeline

```python
@ai
def task(text: str) -> str:
    """Do something"""
    thinking: str = _ai["Think step by step"]
    return _ai

# You can see each stage of the pipeline
task.introspect()  # Shows function analysis
task.template      # Shows template object
task.render(text="example")  # Shows final prompt string
task.to_messages(text="example")  # Shows API messages

# Example output of to_messages:
[
    {
        "role": "system",
        "content": "You are performing the task: task\n\nDo something"
    },
    {
        "role": "user",
        "content": "INPUT:\ntext: example\n\nOUTPUT FORMAT:\n..."
    }
]
```

## Preset Templates with Override

```python
from llm import ai, presets

# Use preset but customize it
@ai(
    template=presets.reasoning,  # Start with reasoning preset
    template_overrides={
        "system": "You are a math expert",
        "format": "xml"  # Change JSON to XML
    }
)
def solve(problem: str) -> float:
    thinking: str = _ai
    return _ai
```

## Complete Control: Template Classes

```python
from llm import BaseTemplate

class MyCustomTemplate(BaseTemplate):
    """Full control over prompt construction"""
    
    def render_system(self, fn_name, docstring, **config):
        return f"SYSTEM: You are doing {fn_name}\n{docstring}"
    
    def render_inputs(self, inputs: dict):
        return "\n".join(f"{k.upper()}: {v}" for k, v in inputs.items())
    
    def render_outputs(self, outputs: dict):
        return "Output in JSON with keys: " + ", ".join(outputs.keys())
    
    def to_messages(self, **fields):
        return [
            {"role": "system", "content": self.render_system(...)},
            {"role": "user", "content": self.render_inputs(...) + self.render_outputs(...)}
        ]

@ai(template=MyCustomTemplate())
def task(text: str) -> str:
    return _ai
```

## Bridging Chat and Task

```python
from llm import ai, Chat

# Task mode - stateless
@ai
def analyze(text: str) -> Analysis:
    """Analyze text"""
    return _ai

# Chat mode - stateful
chat = Chat(
    model="sonnet",
    system="You are helpful"
)

# But Chat can USE tasks!
@chat.task
def analyze(text: str) -> Analysis:
    """Analyze text"""
    return _ai

# Now you can use the task in chat
chat("Can you analyze this text for me?")
# Chat detects it should call analyze() internally

# Or call directly
result = chat.analyze("some text")
```

## Unified Configuration

```python
from llm import ai, Chat, template, configure

# Global config affects both
configure(
    model="sonnet",
    temperature=0.7,
    default_template=template.from_preset("reasoning")
)

# Task uses config
@ai
def task(input: str) -> str:
    return _ai

# Chat uses same config
chat = Chat()

# Both can override
@ai(template=custom_template, model="opus")
def special_task(input: str) -> str:
    return _ai

chat2 = Chat(template=custom_template, model="opus")
```

## The Template Registry

```python
from llm import templates

# Built-in templates
templates.simple
templates.reasoning  
templates.structured
templates.creative
templates.extract

# Register your own
templates.register("my_template", my_custom_template)

# Use anywhere
@ai(template="my_template")  # String reference
def task(input: str) -> str:
    return _ai

chat = Chat(template="my_template")
```

## Debugging and Inspection

```python
@ai
def task(text: str) -> str:
    thinking: str = _ai["Think carefully"]
    return _ai

# Dry run - see what would be sent
preview = task.preview(text="example")
print(preview.prompt)      # Raw prompt string
print(preview.messages)    # API messages
print(preview.template)    # Template used
print(preview.config)      # Model config

# Actually call with debugging
result = task("example", debug=True)
# Prints:
# - Template used
# - Rendered prompt
# - API messages
# - Raw response
# - Parsed output
```

## Format Adapters

```python
from llm import ai, format

# Choose output format
@ai(format=format.XML)
def task(text: str) -> Analysis:
    thinking: str = _ai
    return _ai

# Format affects how outputs are requested
# XML format renders as:
"""
<thinking>
...
</thinking>

<result>
...
</result>
"""

# JSON format renders as:
"""
{
  "thinking": "...",
  "result": {...}
}
"""

# Custom format
@ai(format=format.Custom(
    field_wrapper="### {name} ###\n{value}\n### END {name} ###"
))
def task(text: str) -> str:
    return _ai
```

## Template Inheritance

```python
from llm import template

# Base template
base = template("""
You are an AI assistant.

{inputs}

{outputs}
""")

# Extend it
expert_base = base.extend(
    system="You are an expert AI assistant with 20 years experience"
)

# Extend further
math_expert = expert_base.extend(
    system=expert_base.system + "\nYou specialize in mathematics."
)

@ai(template=math_expert)
def solve(problem: str) -> float:
    return _ai
```

## The Complete Picture

```python
from llm import ai, Chat, template, Section, _ai

# 1. Simple - everything automatic
@ai
def summarize(text: str) -> str:
    """Summarize text"""
    return _ai

# 2. Inspect what it's doing
print(summarize.template)
print(summarize.render(text="example"))

# 3. Customize template with sections
my_template = template.from_sections([
    Section.system("You are concise"),
    Section.inputs(),
    Section.thinking(),
    Section.output(format="markdown")
])

@ai(template=my_template)
def summarize(text: str) -> str:
    thinking: str = _ai
    return _ai

# 4. Or write full custom template
custom = template("""
SYSTEM: Be helpful

INPUT: {text}

REASONING:
{thinking}

SUMMARY:
{result}
""")

@ai(template=custom)
def summarize(text: str) -> str:
    thinking: str = _ai
    return _ai

# 5. Use in Chat
chat = Chat(template=custom)
chat("Hello")

# 6. Chat can use tasks
@chat.task
def analyze(text: str) -> Analysis:
    return _ai

chat("Analyze this for me: ...")
# Automatically calls analyze()
```

## Key Principles

1. **Default is smart** - works out of box
2. **Always inspectable** - `.template`, `.render()`, `.to_messages()`
3. **Progressively customizable** - override what you need
4. **Not magical** - you can see and control everything
5. **Unified** - same template system for tasks and chat
6. **Composable** - sections, inheritance, presets
7. **Debuggable** - preview, dry-run, debug mode

## Implementation Note

```python
class AIFunction:
    def __init__(self, fn, template=None, **config):
        self.fn = fn
        self.config = config
        
        # Auto-generate template if not provided
        if template is None:
            self.template = self._auto_generate_template()
        else:
            self.template = template
    
    def _auto_generate_template(self):
        """Introspect function and create template"""
        # Analyze signature, docstring, _ai usage
        # Build template from analysis
        pass
    
    def render(self, **kwargs):
        """Render template with values"""
        return self.template.render(
            fn_name=self.fn.__name__,
            docstring=self.fn.__doc__,
            inputs=kwargs,
            outputs=self._get_outputs()
        )
    
    def to_messages(self, **kwargs):
        """Convert to API message format"""
        return self.template.to_messages(**kwargs)
    
    def preview(self, **kwargs):
        """Dry run - show what would be sent"""
        return Preview(
            prompt=self.render(**kwargs),
            messages=self.to_messages(**kwargs),
            template=self.template,
            config=self.config
        )
```

Does this bridge the gap? You get the clean FunctAI interface but with full control over the templating layer when you need it.


# user 6

Now that is a bit 2 magical, I think what I would want is that you know, most people have by now learned The array of messages of the OpenAI standard. And so I would want people to be defining that programmatically where that would be like, if there's an example that's here, and then the Then they could do string interpolation bit everywhere there and that would come from, you know, demos comes here like that. And demo input fields, demo output fields. They they can, like, The cat. Instruction, input fields, list their values and their key names. Output field list key value and and key names. Yeah. Instruction, system fields. And then all of that could be, like, into curly braces in that whole thing. And then that would be kind of, a function. Like, that would be the way that you'd say how things should be templated. And that would even let us do things like assistance prefilled because if in my template I'm deciding that I have an assistant message that starts by saying something. Then I I would be able to To do that, and I could also use even that strategy in a simplified format if I'm dealing with, non instruct model, the base models where you don't do system user assistant message. But you just do one string.

# ai 6

# Direct Message Template Control

Ah YES! This is much better. People already know the message format, so let them **define it directly** with interpolation. No abstraction, no magic.

## The Core Idea

```python
from llm import ai, _ai

@ai(
    messages=[
        {
            "role": "system",
            "content": "You are an expert at {task_name}. {instruction}"
        },
        {
            "role": "user", 
            "content": "INPUT: {text}\n\nPlease analyze this."
        }
    ]
)
def analyze(text: str) -> Analysis:
    """Analyze text carefully"""
    return _ai

# The template IS just the messages array
# Variables in {curly_braces} get filled in
```

## Available Template Variables

```python
# Automatic variables available in any template:
{
    # From function
    "fn_name": "analyze",
    "instruction": "Analyze text carefully",  # from docstring
    
    # Input fields
    "text": "actual input value",
    # ... all function arguments
    
    # Output fields (discovered from function body)
    "thinking": None,  # will be generated
    "result": None,    # will be generated
    
    # Demos (if provided)
    "demos": [
        {"text": "example 1", "result": "output 1"},
        {"text": "example 2", "result": "output 2"}
    ]
}
```

## Full Example with Everything

```python
from llm import ai, _ai

@ai(
    messages=[
        {
            "role": "system",
            "content": """You are performing: {fn_name}
            
Task: {instruction}

Output the following fields:
{output_fields}"""
        },
        # Few-shot examples (if demos provided)
        *[
            {
                "role": "user",
                "content": "Input: {demo.text}"
            },
            {
                "role": "assistant", 
                "content": "Analysis: {demo.result}"
            }
            for demo in demos
        ],
        {
            "role": "user",
            "content": """INPUT:
{text}

Think step by step, then provide your analysis."""
        }
    ]
)
def analyze(text: str) -> str:
    """Analyze text for sentiment and themes"""
    thinking: str = _ai
    return _ai

# Use it
result = analyze("Great product!")

# With demos
result = analyze(
    "Great product!",
    demos=[
        {"text": "I love it", "result": "positive"},
        {"text": "Terrible", "result": "negative"}
    ]
)
```

## Assistant Prefill (Key Feature!)

```python
@ai(
    messages=[
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "{text}"},
        {
            "role": "assistant",
            "content": "Let me analyze this step by step:\n\n1. "
            # The LLM continues from here!
        }
    ]
)
def analyze(text: str) -> str:
    return _ai
```

## Structured Outputs in Messages

```python
@ai(
    messages=[
        {"role": "system", "content": "Extract structured data"},
        {"role": "user", "content": "{text}"},
        {
            "role": "assistant",
            "content": """I'll extract the information:

REASONING:
{thinking}

DATA:
{result}"""
        }
    ]
)
def extract(text: str) -> Data:
    thinking: str = _ai
    return _ai
```

## Output Field Specification

```python
# Helper to list output fields nicely
@ai(
    messages=[
        {
            "role": "system",
            "content": """Output JSON with these fields:
{output_schema}"""
        },
        {"role": "user", "content": "{text}"}
    ]
)
def task(text: str) -> Analysis:
    thinking: str = _ai
    return _ai

# Library auto-generates output_schema:
"""
{
  "thinking": "string - your reasoning",
  "result": {
    "sentiment": "string",
    "themes": "array"
  }
}
"""
```

## Base Model Support (No Chat Format)

```python
@ai(
    messages=[
        {
            "role": "text",  # Special role for base models
            "content": """### Instruction
{instruction}

### Input
{text}

### Response
"""
        }
    ],
    model="base-model-name"  # Triggers base model mode
)
def task(text: str) -> str:
    return _ai

# Library converts to single string, not chat messages
```

## The Template Variables System

```python
# What's available in your template strings:

# 1. FUNCTION INFO
{fn_name}          # Function name
{instruction}      # Docstring (first line or full)
{doc}              # Full docstring

# 2. INPUT FIELDS
{text}             # Actual input value
{any_param}        # Any function parameter
{input_fields}     # Auto-formatted list of all inputs
{input_schema}     # JSON schema of inputs

# 3. OUTPUT FIELDS  
{output_fields}    # Auto-formatted list of outputs
{output_schema}    # JSON schema of expected output
{field_name}       # Placeholder for each output field

# 4. DEMOS
{demos}            # List of demo examples
{demo.field}       # Access demo fields in loops

# 5. CONFIG
{model}            # Model being used
{temperature}      # Temperature setting
```

## Advanced: Dynamic Message Construction

```python
from llm import ai, _ai

def my_message_template(fn_name, instruction, demos, **inputs):
    """Build messages programmatically"""
    
    messages = [
        {"role": "system", "content": f"Task: {instruction}"}
    ]
    
    # Add demos dynamically
    for demo in demos:
        messages.append({"role": "user", "content": demo['input']})
        messages.append({"role": "assistant", "content": demo['output']})
    
    # Add actual input
    messages.append({
        "role": "user",
        "content": f"Input: {inputs['text']}"
    })
    
    return messages

@ai(messages=my_message_template)
def task(text: str) -> str:
    return _ai
```

## Presets as Message Templates

```python
from llm import templates

# Built-in message templates
templates.simple = [
    {"role": "system", "content": "{instruction}"},
    {"role": "user", "content": "{text}"}
]

templates.reasoning = [
    {"role": "system", "content": "{instruction}"},
    {"role": "user", "content": "{text}\n\nThink step by step."},
    {"role": "assistant", "content": "Let me think through this:\n\n"}
]

templates.json_extraction = [
    {
        "role": "system",
        "content": "Extract structured data as JSON.\n\nSchema:\n{output_schema}"
    },
    {"role": "user", "content": "{text}"}
]

# Use them
@ai(messages=templates.reasoning)
def task(text: str) -> str:
    return _ai
```

## Multi-Turn Templates with History

```python
@ai(
    messages=[
        {"role": "system", "content": "You are helpful"},
        # Insert history here
        *[
            {
                "role": msg["role"],
                "content": msg["content"]
            }
            for msg in history
        ],
        {"role": "user", "content": "{message}"}
    ],
    stateful=True
)
def chat(message: str) -> str:
    return _ai
```

## Format Variations

```python
# XML format
@ai(
    messages=[
        {"role": "system", "content": "Respond in XML"},
        {"role": "user", "content": "{text}"},
        {
            "role": "assistant",
            "content": """<response>
  <thinking>{thinking}</thinking>
  <answer>{result}</answer>
</response>"""
        }
    ]
)
def task(text: str) -> str:
    thinking: str = _ai
    return _ai

# YAML format
@ai(
    messages=[
        {"role": "system", "content": "Respond in YAML"},
        {"role": "user", "content": "{text}"},
        {
            "role": "assistant",
            "content": """thinking: {thinking}
result: {result}"""
        }
    ]
)
def task(text: str) -> str:
    thinking: str = _ai
    return _ai
```

## Inspection

```python
@ai(messages=[...])
def task(text: str) -> str:
    return _ai

# See the template
print(task.messages)

# See it rendered with actual values
print(task.render(text="example"))

# See final messages that would be sent
print(task.to_messages(text="example"))
```

## The Complete Architecture

```python
from llm import ai, _ai

@ai(
    # Direct message control - no abstraction!
    messages=[
        {
            "role": "system",
            "content": """You are performing: {fn_name}

Task: {instruction}

Output format:
{output_schema}"""
        },
        
        # Few-shot demos (if provided)
        *[{
            "role": "user",
            "content": "{demo.input}"
        }, {
            "role": "assistant",
            "content": "{demo.output}"
        } for demo in demos],
        
        # Actual input
        {
            "role": "user",
            "content": """Input: {text}

Please analyze this carefully."""
        },
        
        # Assistant prefill (optional)
        {
            "role": "assistant",
            "content": "I'll analyze this step by step:\n\n"
        }
    ],
    
    # Model config
    model="sonnet",
    temperature=0.7
)
def analyze(text: str) -> Analysis:
    """Analyze text for sentiment and key themes"""
    
    # Define intermediate outputs
    thinking: str = _ai["Reason through the analysis"]
    
    # Final output
    return _ai

# Use it
result = analyze("This product is amazing!")

# With demos
result = analyze(
    "This product is amazing!",
    demos=[
        {"input": "I love it!", "output": "positive"},
        {"input": "Terrible", "output": "negative"}
    ]
)

# Debug it
print(analyze.messages)  # See template
print(analyze.render(text="example"))  # See rendered
print(analyze.to_messages(text="example"))  # See final messages
```

## Default Template (If No Messages Provided)

```python
# If you don't provide messages, library uses smart default:
@ai
def task(text: str) -> str:
    """Do something"""
    return _ai

# Equivalent to:
@ai(
    messages=[
        {"role": "system", "content": "{instruction}"},
        {"role": "user", "content": "{input_fields}"},
    ]
)
def task(text: str) -> str:
    """Do something"""
    return _ai
```

## Key Principles

1. **Messages are the template** - no extra abstraction
2. **Standard format** - everyone knows `[{"role": "...", "content": "..."}]`
3. **String interpolation** - use `{variables}` anywhere
4. **Assistant prefill** - just add assistant message with partial content
5. **Base model support** - use `"role": "text"` for single string
6. **Demos in template** - use list comprehension to add examples
7. **Inspectable** - always see what's happening
8. **Overridable** - full control when you need it

This is **much less magical** - the template IS the messages array you already know!

Does this feel right? Want to discuss how Chat mode would work with this same system?

# user 7

Oh, yes. Like it. This is very good. But we're missing a little bit of an abstraction for the input fields and output fields. We need a way to kind of like, do it if there are many input fields and many names. Like, because text will not always be the name of the parameter for the input field. And and, we wanted to be the same and symmetrical, yeah. No. Anyway, so that. And then, the other thing. Is, we would want to be able to wrap those things into the curly braces with a like, Python function. Like, if we wanted to define our own Python function to render things like, let's say, we say we have a function that's like, super super schema renderer, and then that's a function that we look at the type of the input and will render this with different types of strings and whatever. We want to be able to do that in in the language. But also while keeping almost everything as as you did because it was really really good.

# ai 7

# Adding Programmatic Control to Message Templates

Yes! We need:
1. **Automatic handling of input/output fields** (not hardcoded names)
2. **Python functions in templates** for custom rendering

## The Solution: Template Functions

```python
from llm import ai, _ai, template_fn

# Define custom rendering functions
@template_fn
def render_inputs(inputs: dict) -> str:
    """Custom input rendering"""
    return "\n".join(f"{k.upper()}: {v}" for k, v in inputs.items())

@template_fn  
def render_schema(schema: dict) -> str:
    """Custom schema rendering"""
    return json.dumps(schema, indent=2)

@template_fn
def render_outputs(outputs: dict) -> str:
    """Custom output field list"""
    lines = []
    for name, field in outputs.items():
        lines.append(f"- {name}: {field.type.__name__}")
        if field.description:
            lines.append(f"  {field.description}")
    return "\n".join(lines)

# Use them in messages with {function_name(...)}
@ai(
    messages=[
        {
            "role": "system",
            "content": """Task: {instruction}

Expected outputs:
{render_outputs(outputs)}"""
        },
        {
            "role": "user",
            "content": """{render_inputs(inputs)}

Please analyze this."""
        }
    ]
)
def analyze(text: str, context: str) -> Analysis:
    """Analyze text with context"""
    thinking: str = _ai
    return _ai
```

## Built-in Template Functions

```python
from llm import ai, _ai

@ai(
    messages=[
        {
            "role": "system",
            "content": """Task: {instruction}

Inputs:
{inputs()}  # Auto-renders all input fields

Outputs:
{outputs()}  # Auto-renders all output fields
"""
        },
        {
            "role": "user",
            "content": "{inputs(style='yaml')}"  # Different style
        }
    ]
)
def task(text: str, context: str) -> Analysis:
    thinking: str = _ai
    return _ai
```

## The Built-in Functions

```python
# Available in all templates:

{inputs()}                    # All input fields, auto-formatted
{inputs(style='yaml')}        # YAML format
{inputs(style='json')}        # JSON format  
{inputs(style='markdown')}    # Markdown list
{inputs(exclude=['context'])} # Exclude some fields
{inputs(only=['text'])}       # Only some fields

{outputs()}                   # All output fields with descriptions
{outputs(style='schema')}     # As JSON schema
{outputs(style='list')}       # Simple list

{schema(type)}                # Schema for a specific type
{schema(Analysis)}            # Returns JSON schema for Analysis

{demos(format='yaml')}        # Render demos in specific format
```

## Complete Example

```python
from llm import ai, _ai

@ai(
    messages=[
        {
            "role": "system",
            "content": """You are performing: {fn_name}

Task: {instruction}

You will receive these inputs:
{inputs(style='list')}

You must produce these outputs:
{outputs(style='schema')}"""
        },
        {
            "role": "user",
            "content": """{inputs(style='yaml')}

Think carefully and respond."""
        }
    ]
)
def analyze(text: str, context: str, metadata: dict) -> Analysis:
    """Analyze text with context and metadata"""
    thinking: str = _ai["Reason step by step"]
    return _ai

# When called with:
analyze(
    text="Great product",
    context="Customer review", 
    metadata={"source": "email"}
)

# Renders as:
"""
System:
You are performing: analyze

Task: Analyze text with context and metadata

You will receive these inputs:
- text (str)
- context (str)
- metadata (dict)

You must produce these outputs:
{
  "thinking": "string",
  "result": {
    "sentiment": "string",
    "themes": "array"
  }
}

User:
text: Great product
context: Customer review
metadata:
  source: email

Think carefully and respond.
"""
```

## Custom Template Functions

```python
from llm import ai, _ai, template_fn

# Define your own rendering logic
@template_fn
def super_schema_renderer(type_hint) -> str:
    """Render schema based on type complexity"""
    if is_simple_type(type_hint):
        return f"Output a {type_hint.__name__}"
    elif is_pydantic_model(type_hint):
        schema = type_hint.model_json_schema()
        return f"Output JSON matching:\n{json.dumps(schema, indent=2)}"
    elif is_dataclass(type_hint):
        fields = get_dataclass_fields(type_hint)
        return "Output with fields:\n" + "\n".join(f"- {f}" for f in fields)
    else:
        return str(type_hint)

@template_fn
def render_with_types(inputs: dict) -> str:
    """Render inputs with type information"""
    lines = []
    for name, value in inputs.items():
        type_name = type(value).__name__
        lines.append(f"{name} ({type_name}): {value}")
    return "\n".join(lines)

# Use in template
@ai(
    messages=[
        {
            "role": "system", 
            "content": "{super_schema_renderer(output_type)}"
        },
        {
            "role": "user",
            "content": "{render_with_types(inputs)}"
        }
    ]
)
def task(text: str, count: int) -> ComplexType:
    return _ai
```

## Accessing Individual Fields

```python
@ai(
    messages=[
        {
            "role": "system",
            "content": "Task: {instruction}"
        },
        {
            "role": "user",
            "content": """Main input: {inputs.text}
            
Additional context: {inputs.context}

{inputs(exclude=['text', 'context'])}"""
        }
    ]
)
def task(text: str, context: str, metadata: dict) -> str:
    return _ai
```

## Demos with Functions

```python
from llm import ai, _ai

@ai(
    messages=[
        {"role": "system", "content": "Task: {instruction}"},
        
        # Render demos with custom function
        "{render_demos(demos, format='conversational')}",
        
        # Or inline
        *[{
            "role": "user",
            "content": "{inputs(only=['text'])}"
        }, {
            "role": "assistant",
            "content": "{demo.result}"
        } for demo in demos],
        
        {"role": "user", "content": "{inputs()}"}
    ]
)
def classify(text: str) -> str:
    return _ai
```

## The Implementation

```python
class MessageTemplate:
    def __init__(self, messages, template_functions=None):
        self.messages = messages
        self.functions = template_functions or {}
        
        # Register built-ins
        self.functions.update({
            'inputs': self._render_inputs,
            'outputs': self._render_outputs,
            'schema': self._render_schema,
            'demos': self._render_demos
        })
    
    def render(self, **context):
        """Render messages with context"""
        rendered = []
        
        for msg in self.messages:
            if isinstance(msg, str):
                # String message - evaluate and expand
                expanded = self._expand_string(msg, context)
                rendered.extend(expanded)
            else:
                # Dict message - render content
                rendered.append({
                    'role': msg['role'],
                    'content': self._render_content(msg['content'], context)
                })
        
        return rendered
    
    def _render_content(self, content: str, context: dict) -> str:
        """Render content with function calls and variables"""
        
        # First pass: evaluate function calls
        # {inputs()} -> actual rendered inputs
        content = self._eval_functions(content, context)
        
        # Second pass: simple variable substitution
        # {text} -> actual text value
        content = content.format(**context)
        
        return content
    
    def _eval_functions(self, content: str, context: dict) -> str:
        """Find and evaluate {function(...)} calls"""
        import re
        
        pattern = r'\{(\w+)\((.*?)\)\}'
        
        def replace_func(match):
            func_name = match.group(1)
            args_str = match.group(2)
            
            if func_name in self.functions:
                # Parse arguments
                args, kwargs = self._parse_args(args_str, context)
                
                # Call function
                result = self.functions[func_name](*args, **kwargs)
                return str(result)
            
            return match.group(0)  # Leave unchanged if not found
        
        return re.sub(pattern, replace_func, content)
    
    def _render_inputs(self, style='default', exclude=None, only=None):
        """Built-in input renderer"""
        # Access inputs from context
        inputs = self.context.get('inputs', {})
        
        if only:
            inputs = {k: v for k, v in inputs.items() if k in only}
        if exclude:
            inputs = {k: v for k, v in inputs.items() if k not in exclude}
        
        if style == 'yaml':
            return yaml.dump(inputs)
        elif style == 'json':
            return json.dumps(inputs, indent=2)
        elif style == 'list':
            return '\n'.join(f"- {k} ({type(v).__name__})" for k, v in inputs.items())
        else:  # default
            return '\n'.join(f"{k}: {v}" for k, v in inputs.items())
```

## Symmetric Input/Output Handling

```python
@ai(
    messages=[
        {
            "role": "system",
            "content": """Task: {instruction}

INPUTS:
{inputs(style='schema')}

OUTPUTS:
{outputs(style='schema')}"""
        },
        {
            "role": "user",
            "content": """{inputs(style='yaml')}

Please provide:
{outputs(style='list')}"""
        }
    ]
)
def task(text: str, context: str) -> Analysis:
    """Analyze text with context"""
    thinking: str = _ai
    confidence: float = _ai
    return _ai

# Both inputs and outputs rendered symmetrically
```

## Advanced: Conditional Rendering

```python
from llm import ai, _ai, template_fn

@template_fn
def if_present(value, template: str) -> str:
    """Conditionally render if value exists"""
    return template.format(value=value) if value else ""

@ai(
    messages=[
        {"role": "system", "content": "{instruction}"},
        {
            "role": "user",
            "content": """Text: {inputs.text}

{if_present(inputs.get('context'), 'Context: {value}')}
{if_present(inputs.get('metadata'), 'Metadata: {value}')}"""
        }
    ]
)
def task(text: str, context: str = None, metadata: dict = None) -> str:
    return _ai
```

## The Complete Pattern

```python
from llm import ai, _ai, template_fn

# Custom rendering function
@template_fn
def fancy_inputs(inputs: dict, highlight: list = None) -> str:
    """Render inputs with highlighting"""
    lines = []
    for k, v in inputs.items():
        prefix = ">>> " if highlight and k in highlight else "    "
        lines.append(f"{prefix}{k}: {v}")
    return "\n".join(lines)

@ai(
    messages=[
        {
            "role": "system",
            "content": """Perform: {fn_name}

Description: {instruction}

Expected outputs:
{outputs(style='schema')}"""
        },
        {
            "role": "user",
            "content": """Input data:
{fancy_inputs(inputs, highlight=['text'])}

Analyze carefully."""
        },
        {
            "role": "assistant",
            "content": "I'll analyze this step by step:\n\n"  # Prefill
        }
    ]
)
def analyze(text: str, context: str, metadata: dict) -> Analysis:
    """Analyze text with full context"""
    thinking: str = _ai["Detailed reasoning"]
    return _ai
```

## Key Features

1. **`{inputs()}` / `{outputs()}`** - Auto-render all fields
2. **Style parameters** - `{inputs(style='yaml')}`, `{outputs(style='schema')}`
3. **Filtering** - `{inputs(only=['text'])}`, `{inputs(exclude=['metadata'])}`
4. **Custom functions** - `@template_fn` decorator for your own renderers
5. **Individual access** - `{inputs.text}` for specific fields
6. **Symmetric design** - Inputs and outputs work the same way
7. **Type-aware** - Functions can inspect types and render appropriately

This keeps the simple message array format but adds the programmatic control you need!

Should we now design how Chat mode works with this system?

