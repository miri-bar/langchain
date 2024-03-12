from typing import (
    Any,
    List,
)

from langchain.text_splitter import TextSplitter
from langchain_core.documents import Document
from langchain_ai21.ai21_base import AI21Base, DocumentType


class AI21SegmentationTextSplitter(TextSplitter):
    """Splitting text into coherent and readable units, based on distinct topics and lines"""

    def __init__(self, **kwargs: Any):
        """Create a new AI21SegmentationTextSplitter."""
        """Create a new TextSplitter."""
        super().__init__(**kwargs)
        self._client = AI21Base()
    def split_text_to_documents(self, source: str) -> List[Document]:
        """Split text into multiple documents.

        Args:
            source: Specifies the text input for text segmentation
        """
        response = self.client.segmentation.create(source=source, source_type=DocumentType.TEXT)
        documents = []
        for segment in response.segments:
            documents.append(Document(page_content=segment.segment_text, metadata={"source_type": segment.segment_type}))
        return documents

    def split_text(self, source: str) -> List[str]:
        """Split text into multiple components.

        Args:
            source: Specifies the text input for text segmentation
        """
        response = self.client.segmentation.create(source=source, source_type=DocumentType.TEXT)
        segments = [segment.segment_text for segment in response.segments]
        return segments

