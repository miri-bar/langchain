from typing import (
    Any,
)

from langchain import TextSplitter
from ai21_base import AI21Base, DocumentType


class AI21SegmentationTextSplitter(TextSplitter, AI21Base):
    """Splitting text into coherent and readable units, based on distinct topics and lines"""
    class Config:
        arbitrary_types_allowed = True

    def __init__(
            self, source: str, source_type: DocumentType = DocumentType.URL, **kwargs: Any
    ) -> None:
        """Create a new TextSplitter.

        Args:
            source: Specifies the text or URL input for text segmentation
            source_type: Type of provided source
        """
        super().__init__(**kwargs)
        self._source = source
        self._source_type = source_type

