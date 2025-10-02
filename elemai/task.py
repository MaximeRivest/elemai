"""Task mode - AI functions with @ai decorator.

This module provides the @ai decorator that transforms regular Python functions
into AI-powered functions. The decorator introspects the function signature and
docstring to automatically generate appropriate prompts.

Example:
    >>> from elemai.task import ai
    >>> from elemai.sentinel import _ai
    >>> @ai
    ... def greet(name: str) -> str:
    ...     '''Say hello to someone'''
    ...     return _ai
    >>> # result = greet("Alice")  # Would call LLM
"""

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union
from .config import Config, get_config
from .client import get_client
from .template import MessageTemplate, templates, Field
from .sentinel import FunctionIntrospector, _ai


@dataclass
class Preview:
    """Preview of what would be sent to the LLM.

    Contains all the information that would be sent to the LLM when
    calling an AI function, useful for debugging and testing.

    Attributes:
        prompt: Human-readable formatted prompt
        messages: List of message dictionaries for the LLM
        template: MessageTemplate used for rendering
        config: Configuration that would be used

    Example:
        >>> from elemai.task import Preview, MessageTemplate
        >>> from elemai.config import Config
        >>> preview = Preview(
        ...     prompt="Test prompt",
        ...     messages=[{"role": "user", "content": "Hello"}],
        ...     template=MessageTemplate([]),
        ...     config=Config()
        ... )
        >>> preview.prompt
        'Test prompt'
    """
    prompt: str
    messages: List[Dict[str, str]]
    template: MessageTemplate
    config: Config


class AIFunction:
    """Wrapper for an AI-powered function.

    This class wraps a regular Python function and transforms it into an
    AI-powered function that uses an LLM to generate results based on
    the function's signature and docstring.

    Attributes:
        func: The original function being wrapped
        stateful: Whether to maintain conversation history
        tools: List of tool functions available to the AI
        config: Configuration for LLM calls
        introspector: FunctionIntrospector for metadata
        metadata: Extracted function metadata
        template: MessageTemplate for generating prompts

    Example:
        >>> from elemai.task import AIFunction
        >>> from elemai.sentinel import _ai
        >>> def greet(name: str) -> str:
        ...     '''Say hello'''
        ...     return _ai
        >>> ai_func = AIFunction(greet)
        >>> ai_func.metadata['fn_name']
        'greet'
    """

    def __init__(
        self,
        func: Callable,
        messages: Optional[Union[List[Dict[str, str]], Callable]] = None,
        template: Optional[MessageTemplate] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        stateful: bool = False,
        tools: Optional[List[Callable]] = None,
        **config_kwargs
    ):
        """Initialize AIFunction wrapper.

        Args:
            func: Function to wrap
            messages: Custom message list or callable
            template: Custom MessageTemplate
            model: Model to use (overrides global config)
            temperature: Temperature setting (overrides global config)
            stateful: Whether to maintain conversation history
            tools: List of tool functions
            **config_kwargs: Additional config parameters

        Example:
            >>> from elemai.task import AIFunction
            >>> from elemai.sentinel import _ai
            >>> def test() -> str:
            ...     '''Test function'''
            ...     return _ai
            >>> ai_func = AIFunction(test, model="claude-sonnet-4-5-20250929")
            >>> ai_func.config.model
            'claude-sonnet-4-5-20250929'
        """
        self.func = func
        self.stateful = stateful
        self.tools = tools or []
        self._history = [] if stateful else None

        # Config
        self.config = get_config().merge(
            model=model,
            temperature=temperature,
            **config_kwargs
        )

        # Introspect function
        self.introspector = FunctionIntrospector(func)
        self.metadata = self.introspector.introspect()

        # Template
        if template:
            self.template = template
        elif messages:
            self.template = MessageTemplate(messages)
        else:
            # Auto-generate template
            self.template = self._auto_generate_template()

    def _auto_generate_template(self) -> MessageTemplate:
        """Generate a default template based on function signature.

        Returns:
            MessageTemplate: Auto-generated template

        Example:
            >>> from elemai.task import AIFunction
            >>> from elemai.sentinel import _ai
            >>> def simple() -> str:
            ...     '''Test'''
            ...     return _ai
            >>> ai_func = AIFunction(simple)
            >>> ai_func.template  # doctest: +ELLIPSIS
            <elemai.template.MessageTemplate object at ...>
        """
        # Check if we have intermediate outputs (thinking, etc.)
        output_fields = self.metadata['output_fields']

        if len(output_fields) > 1:
            # Multiple outputs - use reasoning template
            return MessageTemplate(templates.reasoning)
        else:
            # Simple case
            return MessageTemplate(templates.simple)

    def _build_context(self, **kwargs) -> Dict[str, Any]:
        """Build template rendering context.

        Args:
            **kwargs: Input values for the function

        Returns:
            dict: Context dictionary for template rendering

        Example:
            >>> from elemai.task import AIFunction
            >>> from elemai.sentinel import _ai
            >>> def test(x: int) -> str:
            ...     '''Test'''
            ...     return _ai
            >>> ai_func = AIFunction(test)
            >>> ctx = ai_func._build_context(x=5)
            >>> ctx['inputs']['x']
            5
        """
        # Convert output fields to Field objects
        output_field_objs = []
        for field_dict in self.metadata['output_fields']:
            output_field_objs.append(Field(
                name=field_dict['name'],
                type=field_dict['type'],
                description=field_dict.get('description')
            ))

        context = {
            'fn_name': self.metadata['fn_name'],
            'instruction': self.metadata['instruction'],
            'doc': self.metadata['doc'],
            'inputs': kwargs,
            'input_fields': self.metadata['input_fields'],
            'output_fields': output_field_objs,
            'output_type': self.metadata['return_type'],
            'demos': kwargs.pop('demos', []),
        }

        return context

    def _parse_output(self, text: str, output_fields: List[Dict]) -> Any:
        """Parse LLM output to extract structured data.

        Args:
            text: Raw LLM response
            output_fields: List of expected output fields

        Returns:
            Parsed output (structured or raw text)

        Example:
            >>> from elemai.task import AIFunction
            >>> from elemai.sentinel import _ai
            >>> def test() -> int:
            ...     '''Test'''
            ...     return _ai
            >>> ai_func = AIFunction(test)
            >>> result = ai_func._parse_output("42", [{'name': 'result', 'type': int}])
            >>> result
            42
        """
        if len(output_fields) == 1 and output_fields[0]['name'] == 'result':
            # Single output - try to parse to return type
            return_type = self.metadata['return_type']
            return self._parse_to_type(text, return_type)

        # Multiple outputs - extract each field
        result = {}
        for field in output_fields:
            value = self._extract_field(text, field['name'], field['type'])
            result[field['name']] = value

        # If only one field and it's 'result', return just the value
        if len(result) == 1 and 'result' in result:
            return result['result']

        # Return as object with attributes
        return type('Result', (), result)()

    def _extract_field(self, text: str, field_name: str, field_type: type) -> Any:
        """Extract a specific field from text.

        Args:
            text: Text to extract from
            field_name: Name of field to extract
            field_type: Target type for the field

        Returns:
            Extracted and parsed value

        Example:
            >>> from elemai.task import AIFunction
            >>> from elemai.sentinel import _ai
            >>> def test() -> str:
            ...     '''Test'''
            ...     return _ai
            >>> ai_func = AIFunction(test)
            >>> result = ai_func._extract_field("answer: 42", "answer", int)
            >>> result
            42
        """
        # Try common patterns
        patterns = [
            rf'{field_name}:\s*(.*?)(?:\n\n|\n[A-Z]|$)',
            rf'<{field_name}>(.*?)</{field_name}>',
            rf'"{field_name}":\s*"(.*?)"',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                value = match.group(1).strip()
                return self._parse_to_type(value, field_type)

        # If not found, return the whole text
        return self._parse_to_type(text, field_type)

    def _parse_to_type(self, text: str, target_type: type) -> Any:
        """Parse text to target type.

        Args:
            text: Text to parse
            target_type: Target Python type

        Returns:
            Parsed value of target_type

        Example:
            >>> from elemai.task import AIFunction
            >>> from elemai.sentinel import _ai
            >>> def test() -> str:
            ...     '''Test'''
            ...     return _ai
            >>> ai_func = AIFunction(test)
            >>> ai_func._parse_to_type("123", int)
            123
            >>> ai_func._parse_to_type("hello", str)
            'hello'
        """
        # Handle basic types
        if target_type == str or target_type == Any:
            return text

        if target_type == int:
            # Extract first number
            match = re.search(r'-?\d+', text)
            return int(match.group(0)) if match else 0

        if target_type == float:
            match = re.search(r'-?\d+\.?\d*', text)
            return float(match.group(0)) if match else 0.0

        if target_type == bool:
            return text.lower() in ('true', 'yes', '1', 'correct')

        # Try JSON parsing for complex types
        if hasattr(target_type, 'model_validate_json'):
            # Pydantic model
            try:
                # Extract JSON if embedded in text
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    return target_type.model_validate_json(json_match.group(0))
                else:
                    return target_type.model_validate_json(text)
            except:
                pass

        # Try generic JSON parse
        try:
            json_match = re.search(r'\{.*\}|\[.*\]', text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                if hasattr(target_type, '__annotations__'):
                    # Dataclass or similar
                    return target_type(**data)
                return data
        except:
            pass

        return text

    def __call__(self, *args, **kwargs):
        """Execute the AI function.

        Args:
            *args: Positional arguments matching function signature
            **kwargs: Keyword arguments matching function signature

        Returns:
            Parsed LLM response matching return type

        Example:
            >>> from elemai.task import AIFunction
            >>> from elemai.sentinel import _ai
            >>> def greet(name: str) -> str:
            ...     '''Say hello'''
            ...     return _ai
            >>> ai_func = AIFunction(greet)
            >>> # result = ai_func("Alice")  # Would call LLM
        """
        # Convert positional args to kwargs
        input_fields = self.metadata['input_fields']
        for i, arg in enumerate(args):
            if i < len(input_fields):
                kwargs[input_fields[i]['name']] = arg

        # Build context
        context = self._build_context(**kwargs)

        # Render template
        messages = self.template.render(**context)

        # Add history if stateful
        if self.stateful and self._history:
            # Insert history before last message
            messages = messages[:-1] + self._history + messages[-1:]

        # Get client
        client = get_client(self.config)

        # Call LLM
        response = client.complete(messages, model=self.config.model,
                                   temperature=self.config.temperature)
        text = response['text']

        # Store in history if stateful
        if self.stateful:
            self._history.append({'role': 'user', 'content': str(kwargs)})
            self._history.append({'role': 'assistant', 'content': text})

        # Parse output
        output_fields = self.metadata['output_fields']
        result = self._parse_output(text, output_fields)

        return result

    def render(self, **kwargs) -> str:
        """Render the prompt with given inputs.

        Args:
            **kwargs: Input values for the function

        Returns:
            str: Human-readable formatted prompt

        Example:
            >>> from elemai.task import AIFunction
            >>> from elemai.sentinel import _ai
            >>> def greet(name: str) -> str:
            ...     '''Say hello'''
            ...     return _ai
            >>> ai_func = AIFunction(greet)
            >>> prompt = ai_func.render(name="Alice")
            >>> 'Alice' in prompt
            True
        """
        context = self._build_context(**kwargs)
        messages = self.template.render(**context)
        return '\n\n'.join(f"{m['role'].upper()}:\n{m['content']}" for m in messages)

    def to_messages(self, **kwargs) -> List[Dict[str, str]]:
        """Get the message list that would be sent.

        Args:
            **kwargs: Input values for the function

        Returns:
            list: List of message dictionaries

        Example:
            >>> from elemai.task import AIFunction
            >>> from elemai.sentinel import _ai
            >>> def test(x: int) -> str:
            ...     '''Test'''
            ...     return _ai
            >>> ai_func = AIFunction(test)
            >>> msgs = ai_func.to_messages(x=5)
            >>> len(msgs) >= 1
            True
        """
        context = self._build_context(**kwargs)
        return self.template.render(**context)

    def preview(self, **kwargs) -> Preview:
        """Preview what would be sent to the LLM.

        Args:
            **kwargs: Input values for the function

        Returns:
            Preview: Preview object with prompt, messages, template, config

        Example:
            >>> from elemai.task import AIFunction
            >>> from elemai.sentinel import _ai
            >>> def test(x: str) -> str:
            ...     '''Test'''
            ...     return _ai
            >>> ai_func = AIFunction(test)
            >>> preview = ai_func.preview(x="hello")
            >>> 'hello' in preview.prompt
            True
        """
        context = self._build_context(**kwargs)
        messages = self.template.render(**context)
        prompt = self.render(**kwargs)

        return Preview(
            prompt=prompt,
            messages=messages,
            template=self.template,
            config=self.config
        )


def ai(
    func: Optional[Callable] = None,
    *,
    messages: Optional[Union[List[Dict[str, str]], Callable]] = None,
    template: Optional[MessageTemplate] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    stateful: bool = False,
    tools: Optional[List[Callable]] = None,
    **config_kwargs
) -> Union[AIFunction, Callable]:
    """Decorator to create an AI-powered function.

    This decorator transforms a regular Python function into an AI function
    that uses an LLM to generate results. The function's signature and
    docstring are used to automatically generate prompts.

    Args:
        func: Function to wrap (when used without parentheses)
        messages: Custom message list or callable
        template: Custom MessageTemplate
        model: Model to use (e.g., "claude-sonnet-4-5-20250929")
        temperature: Temperature setting (0.0 to 1.0)
        stateful: Whether to maintain conversation history
        tools: List of tool functions
        **config_kwargs: Additional config parameters

    Returns:
        AIFunction or decorator function

    Example:
        >>> from elemai.task import ai
        >>> from elemai.sentinel import _ai
        >>> @ai
        ... def summarize(text: str) -> str:
        ...     '''Summarize the text'''
        ...     return _ai
        >>> summarize.metadata['fn_name']
        'summarize'

        >>> @ai(model="claude-sonnet-4-5-20250929", temperature=0.5)
        ... def precise_task(input: str) -> str:
        ...     '''Do something precisely'''
        ...     return _ai
        >>> precise_task.config.temperature
        0.5
    """
    def decorator(f: Callable) -> AIFunction:
        return AIFunction(
            f,
            messages=messages,
            template=template,
            model=model,
            temperature=temperature,
            stateful=stateful,
            tools=tools,
            **config_kwargs
        )

    if func is None:
        # Called with arguments: @ai(...)
        return decorator
    else:
        # Called without arguments: @ai
        return decorator(func)
