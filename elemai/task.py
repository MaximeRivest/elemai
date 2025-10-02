"""Task mode - AI functions with @ai decorator."""

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
    """Preview of what would be sent to the LLM."""
    prompt: str
    messages: List[Dict[str, str]]
    template: MessageTemplate
    config: Config


class AIFunction:
    """Wrapper for an AI-powered function."""

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
        """Generate a default template based on function signature."""
        # Check if we have intermediate outputs (thinking, etc.)
        output_fields = self.metadata['output_fields']

        if len(output_fields) > 1:
            # Multiple outputs - use reasoning template
            return MessageTemplate(templates.reasoning)
        else:
            # Simple case
            return MessageTemplate(templates.simple)

    def _build_context(self, **kwargs) -> Dict[str, Any]:
        """Build template rendering context."""
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
        """
        Parse LLM output to extract structured data.

        Args:
            text: Raw LLM response
            output_fields: List of expected output fields

        Returns:
            Parsed output (structured or raw text)
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
        """Extract a specific field from text."""
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
        """Parse text to target type."""
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
        """Execute the AI function."""
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
        """Render the prompt with given inputs."""
        context = self._build_context(**kwargs)
        messages = self.template.render(**context)
        return '\n\n'.join(f"{m['role'].upper()}:\n{m['content']}" for m in messages)

    def to_messages(self, **kwargs) -> List[Dict[str, str]]:
        """Get the message list that would be sent."""
        context = self._build_context(**kwargs)
        return self.template.render(**context)

    def preview(self, **kwargs) -> Preview:
        """Preview what would be sent to the LLM."""
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
    """
    Decorator to create an AI-powered function.

    Examples:
        @ai
        def summarize(text: str) -> str:
            '''Summarize the text'''
            return _ai

        @ai(model="opus", temperature=0)
        def precise_task(input: str) -> str:
            '''Do something precisely'''
            return _ai

        @ai(messages=[
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "{text}"}
        ])
        def custom(text: str) -> str:
            return _ai
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
