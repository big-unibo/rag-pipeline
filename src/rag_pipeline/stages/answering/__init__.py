"""Answer generation stage contract and concrete classes."""

from .ollama_answerer import OllamaAnswerer
from .interface import Answerer

__all__ = ["Answerer", "OllamaAnswerer"]
