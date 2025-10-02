"""Template system for message construction with function support."""

import json
import re
from dataclasses import dataclass, is_dataclass, fields as dataclass_fields
from typing import Any, Callable, Dict, List, Optional, Union
import yaml


@dataclass
class Field:
    """Represents an input or output field."""
    name: str
    type: type
    description: Optional[str] = None


class TemplateFunctions:
    """Registry and implementation of template functions."""

    def __init__(self):
        self._functions: Dict[str, Callable] = {}
        self._context: Dict[str, Any] = {}
        self._register_builtins()

    def _register_builtins(self):
        """Register built-in template functions."""
        self._functions['inputs'] = self._render_inputs
        self._functions['outputs'] = self._render_outputs
        self._functions['schema'] = self._render_schema
        self._functions['demos'] = self._render_demos

    def register(self, name: str, func: Callable):
        """Register a custom template function."""
        self._functions[name] = func

    def set_context(self, context: Dict[str, Any]):
        """Set the current rendering context."""
        self._context = context

    def _render_inputs(self, style: str = 'default', exclude: Optional[List[str]] = None,
                      only: Optional[List[str]] = None) -> str:
        """Render input fields."""
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
        """Render output field specifications."""
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
        """Render JSON schema for a type."""
        schema = self._type_to_schema(type_hint)
        return json.dumps(schema, indent=2)

    def _render_demos(self, format: str = 'default') -> str:
        """Render demonstration examples."""
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
        """Convert a type hint to a schema representation."""
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
        """Call a registered template function."""
        if name in self._functions:
            return self._functions[name](*args, **kwargs)
        raise ValueError(f"Unknown template function: {name}")


class MessageTemplate:
    """Message-based template with function support."""

    def __init__(self, messages: Union[List[Dict[str, str]], Callable]):
        """
        Initialize message template.

        Args:
            messages: Either a list of message dicts or a callable that generates them
        """
        self.messages = messages
        self.functions = TemplateFunctions()

    def register_function(self, name: str, func: Callable):
        """Register a custom template function."""
        self.functions.register(name, func)

    def render(self, **context) -> List[Dict[str, str]]:
        """
        Render messages with the given context.

        Args:
            context: Variables to use in rendering (inputs, outputs, etc.)

        Returns:
            List of rendered message dicts
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
        """Render message content with function calls and variables."""
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
        """Find and evaluate {function(...)} calls."""
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
        """Parse function arguments from string."""
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
        """Handle nested access like {inputs.text}."""
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
        """Expand a string into message dicts."""
        # This is for future expansion if needed
        return []


# Built-in message templates
class Templates:
    """Collection of built-in message templates."""

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
    """Decorator to mark a function as a template function."""
    func._is_template_fn = True
    return func
