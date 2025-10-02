"""Template system for message construction with function support.

This module provides a flexible templating system for constructing LLM messages
with dynamic content, variable substitution, and built-in template functions.

Example:
    >>> from elemai.template import MessageTemplate
    >>> messages = [{"role": "user", "content": "Hello {name}!"}]
    >>> template = MessageTemplate(messages)
    >>> result = template.render(name="World")
    >>> result[0]['content']
    'Hello World!'
"""

import json
import re
from dataclasses import dataclass, is_dataclass, fields as dataclass_fields
from typing import Any, Callable, Dict, List, Optional, Union
import yaml


@dataclass
class Field:
    """Represents an input or output field specification.

    Used to define expected inputs or outputs for LLM tasks with type information
    and optional descriptions.

    Attributes:
        name: The name of the field
        type: The Python type of the field (str, int, list, etc.)
        description: Optional human-readable description of the field

    Example:
        >>> from elemai.template import Field
        >>> field = Field(name="username", type=str, description="User's name")
        >>> field.name
        'username'
        >>> field.type
        <class 'str'>
    """
    name: str
    type: type
    description: Optional[str] = None


class TemplateFunctions:
    """Registry and implementation of template functions.

    This class manages custom and built-in template functions that can be
    called within message templates using the {function(args)} syntax.

    Attributes:
        _functions: Dictionary mapping function names to callables
        _context: Current rendering context with variables

    Example:
        >>> from elemai.template import TemplateFunctions
        >>> funcs = TemplateFunctions()
        >>> funcs.set_context({'inputs': {'name': 'Alice', 'age': 30}})
        >>> output = funcs.call('inputs', style='json')
        >>> '"name"' in output
        True
    """

    def __init__(self):
        """Initialize template functions registry.

        Example:
            >>> from elemai.template import TemplateFunctions
            >>> funcs = TemplateFunctions()
            >>> 'inputs' in funcs._functions
            True
        """
        self._functions: Dict[str, Callable] = {}
        self._context: Dict[str, Any] = {}
        self._register_builtins()

    def _register_builtins(self):
        """Register built-in template functions.

        Registers: inputs, outputs, schema, demos functions.

        Example:
            >>> from elemai.template import TemplateFunctions
            >>> funcs = TemplateFunctions()
            >>> funcs._functions['inputs']  # doctest: +ELLIPSIS
            <bound method TemplateFunctions._render_inputs of ...>
        """
        self._functions['inputs'] = self._render_inputs
        self._functions['outputs'] = self._render_outputs
        self._functions['schema'] = self._render_schema
        self._functions['demos'] = self._render_demos

    def register(self, name: str, func: Callable):
        """Register a custom template function.

        Args:
            name: Name to register the function under
            func: Callable to register

        Example:
            >>> from elemai.template import TemplateFunctions
            >>> funcs = TemplateFunctions()
            >>> def custom(): return "custom result"
            >>> funcs.register('custom', custom)
            >>> funcs.call('custom')
            'custom result'
        """
        self._functions[name] = func

    def set_context(self, context: Dict[str, Any]):
        """Set the current rendering context.

        Args:
            context: Dictionary with variables for rendering

        Example:
            >>> from elemai.template import TemplateFunctions
            >>> funcs = TemplateFunctions()
            >>> funcs.set_context({'key': 'value'})
            >>> funcs._context['key']
            'value'
        """
        self._context = context

    def _render_inputs(self, style: str = 'default', exclude: Optional[List[str]] = None,
                      only: Optional[List[str]] = None) -> str:
        """Render input fields in various formats.

        Args:
            style: Output format ('default', 'yaml', 'json', 'list', 'schema')
            exclude: List of field names to exclude
            only: List of field names to include (overrides exclude)

        Returns:
            str: Formatted input fields

        Example:
            >>> from elemai.template import TemplateFunctions
            >>> funcs = TemplateFunctions()
            >>> funcs.set_context({'inputs': {'name': 'Alice', 'age': 30}})
            >>> output = funcs._render_inputs(style='default')
            >>> 'name: Alice' in output
            True
        """
        inputs = self._context.get('inputs', {})

        if only:
            inputs = {k: v for k, v in inputs.items() if k in only}
        if exclude:
            inputs = {k: v for k, v in inputs.items() if k not in exclude}

        if style == 'yaml':
            return yaml.dump(inputs, default_flow_style=False)
        elif style == 'json':
            return json.dumps(inputs, indent=2)
        elif style == 'list':
            return '\n'.join(f"- {k} ({type(v).__name__})" for k, v in inputs.items())
        elif style == 'schema':
            schema = {}
            for k, v in inputs.items():
                schema[k] = self._type_to_schema(type(v))
            return json.dumps(schema, indent=2)
        else:  # default
            return '\n'.join(f"{k}: {v}" for k, v in inputs.items())

    def _render_outputs(self, style: str = 'default') -> str:
        """Render output field specifications.

        Args:
            style: Output format ('default', 'schema', 'list')

        Returns:
            str: Formatted output field specifications

        Example:
            >>> from elemai.template import TemplateFunctions, Field
            >>> funcs = TemplateFunctions()
            >>> fields = [Field('result', str, 'The result')]
            >>> funcs.set_context({'output_fields': fields})
            >>> output = funcs._render_outputs()
            >>> 'result: str' in output
            True
        """
        output_fields = self._context.get('output_fields', [])

        if style == 'schema':
            schema = {}
            for field in output_fields:
                schema[field.name] = self._type_to_schema(field.type)
            return json.dumps(schema, indent=2)
        elif style == 'list':
            lines = []
            for field in output_fields:
                line = f"- {field.name}: {field.type.__name__}"
                if field.description:
                    line += f" ({field.description})"
                lines.append(line)
            return '\n'.join(lines)
        else:  # default
            lines = []
            for field in output_fields:
                lines.append(f"{field.name}: {field.type.__name__}")
                if field.description:
                    lines.append(f"  {field.description}")
            return '\n'.join(lines)

    def _render_schema(self, type_hint: type) -> str:
        """Render JSON schema for a type.

        Args:
            type_hint: Python type to convert to schema

        Returns:
            str: JSON schema representation

        Example:
            >>> from elemai.template import TemplateFunctions
            >>> funcs = TemplateFunctions()
            >>> schema = funcs._render_schema(str)
            >>> '"string"' in schema
            True
        """
        schema = self._type_to_schema(type_hint)
        return json.dumps(schema, indent=2)

    def _render_demos(self, format: str = 'default') -> str:
        """Render demonstration examples.

        Args:
            format: Output format ('default', 'yaml', 'json')

        Returns:
            str: Formatted demonstration examples

        Example:
            >>> from elemai.template import TemplateFunctions
            >>> funcs = TemplateFunctions()
            >>> demos = [{'input': 'test', 'output': 'result'}]
            >>> funcs.set_context({'demos': demos})
            >>> output = funcs._render_demos()
            >>> 'Example 1:' in output
            True
        """
        demos = self._context.get('demos', [])
        if not demos:
            return ""

        if format == 'yaml':
            return yaml.dump(demos, default_flow_style=False)
        elif format == 'json':
            return json.dumps(demos, indent=2)
        else:
            lines = []
            for i, demo in enumerate(demos, 1):
                lines.append(f"Example {i}:")
                for k, v in demo.items():
                    lines.append(f"  {k}: {v}")
            return '\n'.join(lines)

    def _type_to_schema(self, type_hint: type) -> Any:
        """Convert a type hint to a schema representation.

        Args:
            type_hint: Python type to convert

        Returns:
            Schema representation (str or dict)

        Example:
            >>> from elemai.template import TemplateFunctions
            >>> funcs = TemplateFunctions()
            >>> funcs._type_to_schema(str)
            'string'
            >>> funcs._type_to_schema(int)
            'integer'
        """
        if type_hint == str:
            return "string"
        elif type_hint == int:
            return "integer"
        elif type_hint == float:
            return "number"
        elif type_hint == bool:
            return "boolean"
        elif type_hint == list or getattr(type_hint, '__origin__', None) == list:
            return "array"
        elif type_hint == dict or getattr(type_hint, '__origin__', None) == dict:
            return "object"
        elif hasattr(type_hint, 'model_json_schema'):
            # Pydantic model
            return type_hint.model_json_schema()
        elif is_dataclass(type_hint):
            # Dataclass
            schema = {"type": "object", "properties": {}}
            for field in dataclass_fields(type_hint):
                schema["properties"][field.name] = self._type_to_schema(field.type)
            return schema
        else:
            return str(type_hint)

    def call(self, name: str, *args, **kwargs) -> Any:
        """Call a registered template function.

        Args:
            name: Name of the function to call
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            The result from the called function

        Raises:
            ValueError: If function name is not registered

        Example:
            >>> from elemai.template import TemplateFunctions
            >>> funcs = TemplateFunctions()
            >>> funcs.set_context({'inputs': {'x': 1}})
            >>> result = funcs.call('inputs')
            >>> 'x: 1' in result
            True
        """
        if name in self._functions:
            return self._functions[name](*args, **kwargs)
        raise ValueError(f"Unknown template function: {name}")


class MessageTemplate:
    """Message-based template with function support.

    This class allows creating flexible message templates with variable substitution
    and function calls. Templates can include placeholders like {variable} and
    function calls like {inputs(style='json')}.

    Attributes:
        messages: Template messages (list or callable)
        functions: TemplateFunctions instance for rendering

    Example:
        >>> from elemai.template import MessageTemplate
        >>> template = MessageTemplate([
        ...     {"role": "user", "content": "Name: {name}, Age: {age}"}
        ... ])
        >>> result = template.render(name="Bob", age=25)
        >>> result[0]['content']
        'Name: Bob, Age: 25'
    """

    def __init__(self, messages: Union[List[Dict[str, str]], Callable]):
        """Initialize message template.

        Args:
            messages: Either a list of message dicts or a callable that generates them

        Example:
            >>> from elemai.template import MessageTemplate
            >>> msgs = [{"role": "user", "content": "Hello {name}"}]
            >>> template = MessageTemplate(msgs)
            >>> template.messages == msgs
            True
        """
        self.messages = messages
        self.functions = TemplateFunctions()

    def register_function(self, name: str, func: Callable):
        """Register a custom template function.

        Args:
            name: Name to register the function under
            func: Callable to register

        Example:
            >>> from elemai.template import MessageTemplate
            >>> template = MessageTemplate([])
            >>> def greet(): return "Hello!"
            >>> template.register_function('greet', greet)
            >>> template.functions.call('greet')
            'Hello!'
        """
        self.functions.register(name, func)

    def render(self, **context) -> List[Dict[str, str]]:
        """Render messages with the given context.

        This method processes the template messages, substituting variables
        and evaluating function calls to produce final messages.

        Args:
            **context: Variables to use in rendering (inputs, outputs, etc.)

        Returns:
            list: List of rendered message dictionaries with 'role' and 'content'

        Example:
            >>> from elemai.template import MessageTemplate
            >>> msgs = [{"role": "user", "content": "Hello {name}!"}]
            >>> template = MessageTemplate(msgs)
            >>> result = template.render(name="World")
            >>> result[0]['role']
            'user'
            >>> result[0]['content']
            'Hello World!'
        """
        # Set context for template functions
        self.functions.set_context(context)

        # Get messages (either static list or callable)
        if callable(self.messages):
            msgs = self.messages(**context)
        else:
            msgs = self.messages

        # Render each message
        rendered = []
        for msg in msgs:
            if isinstance(msg, str):
                # String that should expand to messages
                expanded = self._expand_string(msg, context)
                rendered.extend(expanded)
            elif isinstance(msg, dict):
                # Standard message dict
                rendered_msg = {
                    'role': msg['role'],
                    'content': self._render_content(msg['content'], context)
                }
                rendered.append(rendered_msg)
            else:
                rendered.append(msg)

        return rendered

    def _render_content(self, content: str, context: Dict[str, Any]) -> str:
        """Render message content with function calls and variables.

        Args:
            content: Template content string
            context: Variables for substitution

        Returns:
            str: Rendered content with substitutions applied

        Example:
            >>> from elemai.template import MessageTemplate
            >>> template = MessageTemplate([])
            >>> result = template._render_content("Hi {name}", {"name": "Alice"})
            >>> result
            'Hi Alice'
        """
        # First pass: evaluate function calls {func(...)}
        content = self._eval_functions(content, context)

        # Second pass: simple variable substitution {var}
        # Use safe substitution to avoid KeyError on missing variables
        try:
            content = content.format(**context)
        except KeyError as e:
            # Try with nested access (inputs.text)
            content = self._format_with_nested_access(content, context)

        return content

    def _eval_functions(self, content: str, context: Dict[str, Any]) -> str:
        """Find and evaluate {function(...)} calls.

        Args:
            content: Template content with function calls
            context: Variables for evaluation

        Returns:
            str: Content with function calls replaced by results

        Example:
            >>> from elemai.template import MessageTemplate
            >>> template = MessageTemplate([])
            >>> template.functions.set_context({'inputs': {'x': 1}})
            >>> result = template._eval_functions("{inputs()}", {})
            >>> 'x: 1' in result
            True
        """
        # Pattern to match function calls like {inputs(style='yaml')}
        pattern = r'\{(\w+)\((.*?)\)\}'

        def replace_func(match):
            func_name = match.group(1)
            args_str = match.group(2)

            try:
                # Parse and evaluate arguments
                args, kwargs = self._parse_args(args_str, context)
                result = self.functions.call(func_name, *args, **kwargs)
                return str(result)
            except Exception:
                # If function call fails, leave it as-is
                return match.group(0)

        return re.sub(pattern, replace_func, content)

    def _parse_args(self, args_str: str, context: Dict[str, Any]):
        """Parse function arguments from string.

        Args:
            args_str: String representation of arguments
            context: Variables for evaluation

        Returns:
            tuple: (args list, kwargs dict)

        Example:
            >>> from elemai.template import MessageTemplate
            >>> template = MessageTemplate([])
            >>> args, kwargs = template._parse_args("'yaml'", {})
            >>> args
            ['yaml']
        """
        args = []
        kwargs = {}

        if not args_str.strip():
            return args, kwargs

        # Simple parser for key=value and positional args
        # This is a simplified version - a real implementation would need proper parsing
        parts = [p.strip() for p in args_str.split(',')]

        for part in parts:
            if '=' in part:
                key, value = part.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"\'')
                # Try to evaluate as Python literal
                try:
                    value = eval(value, {"__builtins__": {}}, context)
                except:
                    pass
                kwargs[key] = value
            else:
                # Positional arg
                value = part.strip().strip('"\'')
                try:
                    value = eval(value, {"__builtins__": {}}, context)
                except:
                    pass
                args.append(value)

        return args, kwargs

    def _format_with_nested_access(self, content: str, context: Dict[str, Any]) -> str:
        """Handle nested access like {inputs.text}.

        Args:
            content: Template content with nested placeholders
            context: Variables for substitution

        Returns:
            str: Content with nested accesses resolved

        Example:
            >>> from elemai.template import MessageTemplate
            >>> template = MessageTemplate([])
            >>> ctx = {'data': {'key': 'value'}}
            >>> result = template._format_with_nested_access("{data.key}", ctx)
            >>> result
            'value'
        """
        pattern = r'\{(\w+)\.(\w+)\}'

        def replace_nested(match):
            obj_name = match.group(1)
            attr_name = match.group(2)

            if obj_name in context:
                obj = context[obj_name]
                if isinstance(obj, dict) and attr_name in obj:
                    return str(obj[attr_name])
                elif hasattr(obj, attr_name):
                    return str(getattr(obj, attr_name))

            return match.group(0)

        return re.sub(pattern, replace_nested, content)

    def _expand_string(self, msg_str: str, context: Dict[str, Any]) -> List[Dict[str, str]]:
        """Expand a string into message dicts.

        Args:
            msg_str: String to expand
            context: Variables for expansion

        Returns:
            list: List of message dictionaries

        Example:
            >>> from elemai.template import MessageTemplate
            >>> template = MessageTemplate([])
            >>> result = template._expand_string("test", {})
            >>> result
            []
        """
        # This is for future expansion if needed
        return []


# Built-in message templates
class Templates:
    """Collection of built-in message templates.

    Provides pre-configured message templates for common LLM interaction patterns.

    Attributes:
        simple: Basic template with instruction and inputs
        reasoning: Template that encourages step-by-step thinking
        json_extraction: Template for extracting structured JSON data
        structured: Template with YAML inputs and schema outputs

    Example:
        >>> from elemai.template import templates, MessageTemplate
        >>> template = MessageTemplate(templates.simple)
        >>> msgs = template.render(instruction="Summarize", inputs={'text': 'Long text'})
        >>> len(msgs)
        2
        >>> msgs[0]['role']
        'system'
    """

    simple = [
        {"role": "system", "content": "{instruction}"},
        {"role": "user", "content": "{inputs()}"}
    ]

    reasoning = [
        {"role": "system", "content": "{instruction}"},
        {"role": "user", "content": "{inputs()}\n\nThink step by step."},
        {"role": "assistant", "content": "Let me think through this:\n\n"}
    ]

    json_extraction = [
        {
            "role": "system",
            "content": "Extract structured data as JSON.\n\nSchema:\n{outputs(style='schema')}"
        },
        {"role": "user", "content": "{inputs()}"}
    ]

    structured = [
        {
            "role": "system",
            "content": "Task: {instruction}\n\nExpected outputs:\n{outputs(style='schema')}"
        },
        {
            "role": "user",
            "content": "{inputs(style='yaml')}\n\nThink carefully and respond."
        }
    ]


templates = Templates()


def template_fn(func: Callable) -> Callable:
    """Decorator to mark a function as a template function.

    Args:
        func: Function to mark as template function

    Returns:
        Callable: The same function with _is_template_fn attribute set

    Example:
        >>> from elemai.template import template_fn
        >>> @template_fn
        ... def my_template():
        ...     return "test"
        >>> hasattr(my_template, '_is_template_fn')
        True
        >>> my_template._is_template_fn
        True
    """
    func._is_template_fn = True
    return func
