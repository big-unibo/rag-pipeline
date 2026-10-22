"""Document loading stage contract and concrete loaders."""

from .composite_loader import CompositeLoader
from .interface import DocumentLoader
from .json_loader import BasicJsonLoader
from .plain_text_pdf_loader import PlainTextPdfLoader
from .markdown_pdf_loader import MarkdownPdfLoader
from .docling_pdf_loader import DoclingMarkdownPdfLoader

__all__ = [
    "DocumentLoader",
    "PlainTextPdfLoader",
    "MarkdownPdfLoader",
    "DoclingMarkdownPdfLoader",
    "BasicJsonLoader",
    "CompositeLoader",
]