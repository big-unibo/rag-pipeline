"""Answer generation stage contract and concrete classes."""

from .ollama_answerer import OllamaAnswerer
from .ollama_chat_answerer import OllamaChatAnswerer
from .interface import Answerer

__all__ = ["Answerer", "OllamaAnswerer", "OllamaChatAnswerer"]
