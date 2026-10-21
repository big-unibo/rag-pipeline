"""Embedding stage contract and concrete classes."""

from .ollama_embedder import OllamaEmbedder
from .interface import Embedder

__all__ = ["Embedder", "OllamaEmbedder"]
