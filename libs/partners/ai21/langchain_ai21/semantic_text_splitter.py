from typing import (
    Any,
    List,
    Optional, Iterable,
)

import copy
import logging
import re
from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter
from langchain_ai21.ai21_base import AI21Base, DocumentType

logger = logging.getLogger(__name__)


class AI21SemanticTextSplitter(TextSplitter):
    """Splitting text into coherent and readable units, based on distinct topics and lines"""

    def __init__(
            self,
            chunk_size: int = 0,
            chunk_overlap: int = 0,
            **kwargs: Any
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(chunk_size=chunk_size,chunk_overlap=chunk_overlap, **kwargs)
        self._ai21 = AI21Base()

    def split_text(self, source: str) -> List[str]:
        """Split text into multiple components.

        Args:
            source: Specifies the text input for text segmentation
        """
        response = self._ai21.client.segmentation.create(source=source, source_type=DocumentType.TEXT)
        segments = [segment.segment_text for segment in response.segments]
        if self._chunk_size > 0:
            return self._merge_splits_no_seperator(segments)
        return segments

    def split_text_to_documents(self, source: str) -> List[Document]:
        """Split text into multiple documents.

        Args:
            source: Specifies the text input for text segmentation
        """
        response = self._ai21.client.segmentation.create(source=source, source_type=DocumentType.TEXT)
        documents = []
        for segment in response.segments:
            documents.append(Document(page_content=segment.segment_text, metadata={"source_type": segment.segment_type}))
        return documents

    def create_documents(
        self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> List[Document]:
        """Create documents from a list of texts."""
        _metadatas = metadatas or [{}] * len(texts)
        documents = []
        for i, text in enumerate(texts):
            normalized_text = self._replace_continued_newlines(text)
            index = 0
            previous_chunk_len = 0
            for chunk in self.split_text_to_documents(text):
                metadata = dict(copy.deepcopy(_metadatas[i]) | chunk.metadata)
                if self._add_start_index:
                    offset = index + previous_chunk_len - self._chunk_overlap
                    normalized_chunk = self._replace_continued_newlines(chunk.page_content)
                    index = normalized_text.find(normalized_chunk, max(0, offset))
                    metadata["start_index"] = index
                    previous_chunk_len = len(normalized_chunk)
                new_doc = Document(page_content=chunk.page_content, metadata=metadata)
                documents.append(new_doc)
        return documents

    def _replace_continued_newlines(self, string: str) -> str:
        """Use regular expression to replace sequences of '\n'"""
        return re.sub(r'\n{2,}', '\n', string)

    def _merge_splits(self, splits: Iterable[str], separator: str) -> List[str]:
        return self._merge_splits_no_seperator(splits)

    def _merge_splits_no_seperator(self, splits: Iterable[str]) -> List[str]:
        """Merge splits into chunks."""
        chunks = []
        current_chunk = ""
        for split in splits:
            split_len = self._length_function(split)
            if split_len > self._chunk_size:
                logger.warning(f"Split of length {split_len} exceeds chunk size {self._chunk_size}.")
            if self._length_function(current_chunk) + split_len > self._chunk_size:
                if current_chunk != "":
                    chunks.append(current_chunk)
                    current_chunk = ""
            current_chunk += split
        if current_chunk != "":
            chunks.append(current_chunk)
        return chunks

