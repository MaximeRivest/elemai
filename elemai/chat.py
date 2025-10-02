"""Chat mode - stateful conversations."""

from typing import Any, Callable, Dict, List, Optional, Union
from .config import Config, get_config
from .client import get_client
from .template import MessageTemplate


class Chat:
    """
    Stateful chat interface.

    Examples:
        chat = Chat()
        chat("Hello!")
        chat("What's my name?")

        chat = Chat(model="opus", system="You are a pirate")
        chat("Ahoy!")
    """

    def __init__(
        self,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        template: Optional[Union[MessageTemplate, str]] = None,
        **config_kwargs
    ):
        """
        Initialize chat.

        Args:
            model: Model to use
            system: System prompt
            temperature: Temperature setting
            template: Message template or template name
            **config_kwargs: Additional config options
        """
        self.config = get_config().merge(
            model=model,
            temperature=temperature,
            **config_kwargs
        )

        self.system = system
        self.history: List[Dict[str, str]] = []
        self.tasks: Dict[str, Callable] = {}

        # Template
        if isinstance(template, str):
            # Load from templates registry
            from .template import templates
            self.template = getattr(templates, template, None)
        else:
            self.template = template

    def __call__(self, message: str, **kwargs) -> str:
        """
        Send a message and get response.

        Args:
            message: User message
            **kwargs: Additional parameters

        Returns:
            Assistant response
        """
        # Build messages
        messages = []

        # Add system message if present
        if self.system:
            messages.append({'role': 'system', 'content': self.system})

        # Add history
        messages.extend(self.history)

        # Add current message
        messages.append({'role': 'user', 'content': message})

        # Get client
        client = get_client(self.config)

        # Call LLM
        response = client.complete(
            messages,
            model=kwargs.get('model', self.config.model),
            temperature=kwargs.get('temperature', self.config.temperature)
        )

        response_text = response['text']

        # Update history
        self.history.append({'role': 'user', 'content': message})
        self.history.append({'role': 'assistant', 'content': response_text})

        return response_text

    def task(self, func: Callable) -> Callable:
        """
        Register an AI task that can be called from chat.

        Example:
            @chat.task
            def analyze(text: str) -> Analysis:
                return _ai
        """
        from .task import AIFunction

        # Create AIFunction
        ai_func = AIFunction(func)

        # Register it
        self.tasks[func.__name__] = ai_func

        return ai_func

    def reset(self):
        """Clear conversation history."""
        self.history = []

    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self.history.copy()

    def set_system(self, system: str):
        """Update system prompt."""
        self.system = system

    def add_task(self, task: Callable, trigger: Optional[str] = None):
        """
        Add a task that can be used in conversation.

        Args:
            task: AI task function
            trigger: Description of when to trigger (for documentation)
        """
        self.tasks[task.__name__] = task


def chat(message: str) -> str:
    """
    Simple stateful chat function.

    Maintains a global conversation state.

    Example:
        from elemai import chat
        chat("Hello!")
        chat("What's my name?")
    """
    global _global_chat

    if _global_chat is None:
        _global_chat = Chat()

    return _global_chat(message)


# Global chat instance for simple usage
_global_chat: Optional[Chat] = None
