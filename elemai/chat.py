"""Chat mode - stateful conversations.

This module provides a stateful chat interface that maintains conversation
history across multiple turns. Useful for building conversational AI applications.

Example:
    >>> from elemai.chat import Chat
    >>> chat = Chat(system="You are helpful")
    >>> # response = chat("Hello!")  # Would call LLM
    >>> # response2 = chat("What did I just say?")  # Maintains context
"""

from typing import Any, Callable, Dict, List, Optional, Union
from .config import Config, get_config
from .client import get_client
from .template import MessageTemplate


class Chat:
    """Stateful chat interface.

    Maintains conversation history across multiple turns, allowing for
    contextual multi-turn conversations with an LLM.

    Attributes:
        config: Configuration for LLM calls
        system: System prompt for the conversation
        history: List of message dictionaries (conversation history)
        tasks: Dictionary of registered AI tasks
        template: Optional MessageTemplate for custom formatting

    Example:
        >>> from elemai.chat import Chat
        >>> chat = Chat(system="You are a helpful assistant")
        >>> chat.system
        'You are a helpful assistant'
        >>> len(chat.history)
        0

        >>> chat2 = Chat(model="claude-sonnet-4-5-20250929", temperature=0.7)
        >>> chat2.config.temperature
        0.7
    """

    def __init__(
        self,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        template: Optional[Union[MessageTemplate, str]] = None,
        **config_kwargs
    ):
        """Initialize chat instance.

        Args:
            model: Model to use (overrides global config)
            system: System prompt for the conversation
            temperature: Temperature setting (overrides global config)
            template: Message template or template name
            **config_kwargs: Additional config options

        Example:
            >>> from elemai.chat import Chat
            >>> chat = Chat(model="claude-sonnet-4-5-20250929", system="Be concise")
            >>> chat.config.model
            'claude-sonnet-4-5-20250929'
            >>> chat.system
            'Be concise'
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
        """Send a message and get response.

        Sends a message to the LLM and returns the response, automatically
        maintaining conversation history.

        Args:
            message: User message to send
            **kwargs: Additional parameters (model, temperature, etc.)

        Returns:
            str: Assistant's response text

        Example:
            >>> from elemai.chat import Chat
            >>> chat = Chat(system="Echo the user's message")
            >>> # response = chat("Hello")  # Would call LLM and return response
            >>> len(chat.history)  # History starts empty
            0
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
        """Register an AI task that can be called from chat.

        Decorator to register an AI-powered function as a task within
        the chat context.

        Args:
            func: Function to register as AI task

        Returns:
            Callable: AIFunction wrapper

        Example:
            >>> from elemai.chat import Chat
            >>> from elemai.sentinel import _ai
            >>> chat = Chat()
            >>> @chat.task
            ... def analyze(text: str) -> str:
            ...     '''Analyze the text'''
            ...     return _ai
            >>> 'analyze' in chat.tasks
            True
        """
        from .task import AIFunction

        # Create AIFunction
        ai_func = AIFunction(func)

        # Register it
        self.tasks[func.__name__] = ai_func

        return ai_func

    def reset(self):
        """Clear conversation history.

        Removes all messages from history, starting fresh.

        Example:
            >>> from elemai.chat import Chat
            >>> chat = Chat()
            >>> chat.history.append({"role": "user", "content": "test"})
            >>> len(chat.history)
            1
            >>> chat.reset()
            >>> len(chat.history)
            0
        """
        self.history = []

    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history.

        Returns:
            list: Copy of the conversation history

        Example:
            >>> from elemai.chat import Chat
            >>> chat = Chat()
            >>> history = chat.get_history()
            >>> len(history)
            0
            >>> chat.history.append({"role": "user", "content": "hi"})
            >>> history = chat.get_history()
            >>> len(history)
            1
        """
        return self.history.copy()

    def set_system(self, system: str):
        """Update system prompt.

        Args:
            system: New system prompt

        Example:
            >>> from elemai.chat import Chat
            >>> chat = Chat(system="Old prompt")
            >>> chat.set_system("New prompt")
            >>> chat.system
            'New prompt'
        """
        self.system = system

    def add_task(self, task: Callable, trigger: Optional[str] = None):
        """Add a task that can be used in conversation.

        Args:
            task: AI task function
            trigger: Description of when to trigger (for documentation)

        Example:
            >>> from elemai.chat import Chat
            >>> from elemai.sentinel import _ai
            >>> chat = Chat()
            >>> def my_task() -> str:
            ...     '''Task'''
            ...     return _ai
            >>> chat.add_task(my_task)
            >>> 'my_task' in chat.tasks
            True
        """
        self.tasks[task.__name__] = task


def chat(message: str) -> str:
    """Simple stateful chat function.

    Convenience function that maintains a global conversation state.
    Useful for quick interactive sessions without managing a Chat instance.

    Args:
        message: User message to send

    Returns:
        str: Assistant's response

    Example:
        >>> from elemai.chat import chat, _global_chat
        >>> # First call initializes global chat
        >>> # response = chat("Hello!")  # Would call LLM
        >>> # response2 = chat("Hi again")  # Uses same conversation
        >>> _global_chat is None  # Initially None
        True
    """
    global _global_chat

    if _global_chat is None:
        _global_chat = Chat()

    return _global_chat(message)


# Global chat instance for simple usage
_global_chat: Optional[Chat] = None
