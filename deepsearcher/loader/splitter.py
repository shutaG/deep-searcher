## Sentence Window splitting strategy, ref:
#  https://github.com/milvus-io/bootcamp/blob/master/bootcamp/RAG/advanced_rag/sentence_window_with_langchain.ipynb

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class Chunk:
    def __init__(
        self,
        text: str,
        reference: str,
        metadata: dict = None,
        embedding: List[float] = None,
    ):
        self.text = text
        self.reference = reference
        self.metadata = metadata or {}
        self.embedding = embedding or None


def _sentence_window_split(
    split_docs: List[Document], original_document: Document, offset: int = 200
) -> List[Chunk]:
    '''
    增加metadata["wider_text"]的范围
    '''
    chunks = []
    original_text = original_document.page_content
    for doc in split_docs:
        doc_text = doc.page_content
        start_index = original_text.index(doc_text)
        end_index = start_index + len(doc_text) - 1
        wider_text = original_text[
            max(0, start_index - offset) : min(len(original_text), end_index + offset)
        ]
        reference = doc.metadata.pop("reference", "")
        doc.metadata["wider_text"] = wider_text
        chunk = Chunk(text=doc_text, reference=reference, metadata=doc.metadata)
        chunks.append(chunk)
    return chunks


def split_docs_to_chunks(
    documents: List[Document], chunk_size: int = 1500, chunk_overlap=100
) -> List[Chunk]:
    # 初始化文本分割器，默认分割长度为1500，重叠长度为100
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    all_chunks = []
    for doc in documents:
        split_docs = text_splitter.split_documents([doc])
        # 扩展切割后文档的上下文信息
        split_chunks = _sentence_window_split(split_docs, doc, offset=300)
        all_chunks.extend(split_chunks)
    return all_chunks
