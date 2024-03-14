from unittest.mock import Mock

import pytest

from langchain_ai21 import AI21SemanticTextSplitter


@pytest.mark.parametrize(
    ids=[
    "when_chunk_size_is_zero",
    "when_chunk_size_is_large",
    "when_chunk_size_is_small",
    ],
    argnames=["chunk_size"],
    argvalues=[
        (0,),
        (1000,),
        (10,),
    ],
)
def test_split_text__on_chunk_size(
    chunk_size: int,
    mock_client_with_semantic_text_splitter: Mock,
) -> None:
    sts = AI21SemanticTextSplitter(chunk_size=chunk_size, client=mock_client_with_semantic_text_splitter)

    response = sts.split_text("This is a test")
    assert len(response) > 0


def test_split_text__on_large_chunk_size__should_merge_chunks(
    mock_client_with_semantic_text_splitter: Mock,
) -> None:
    sts_no_merge = AI21SemanticTextSplitter(client=mock_client_with_semantic_text_splitter)
    sts_merge = AI21SemanticTextSplitter(client=mock_client_with_semantic_text_splitter, chunk_size=1000)
    segments_no_merge = sts_no_merge.split_text("This is a test")
    segments_merge = sts_merge.split_text("This is a test")
    assert len(segments_merge) > 0
    assert len(segments_merge) > 0
    assert len(segments_no_merge) > len(segments_merge)

def test_split_text__on_small_chunk_size__should_not_merge_chunks(
    mock_client_with_semantic_text_splitter: Mock,
) -> None:
    sts_no_merge = AI21SemanticTextSplitter(client=mock_client_with_semantic_text_splitter)
    segments_merge = sts_no_merge.split_text("This is a test")
    assert len(segments_merge) == 2
