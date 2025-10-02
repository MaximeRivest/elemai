"""elemai - An AI LLM coding library."""

__version__ = "0.3.0"

# Core components
from .sentinel import _ai
from .task import ai, Result
from .chat import Chat, chat
from .config import configure, set_config, get_config
from .template import templates, template_fn, MessageTemplate, Field

__all__ = [
    # Sentinel
    '_ai',

    # Decorators
    'ai',

    # Results
    'Result',

    # Chat
    'Chat',
    'chat',

    # Configuration
    'configure',
    'set_config',
    'get_config',

    # Templates
    'templates',
    'template_fn',
    'MessageTemplate',
    'Field',
]
