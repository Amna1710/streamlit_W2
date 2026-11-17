from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document
class BaseVectorStore(ABC):
    @abstractmethod
    def similarity_search(
        self,
        query: str,
        user_id: str,
        k: int = 4,
    ) -> List[Document]:
        """Retrieve relevant chunks for a specific user."""
        raise NotImplementedError
    @abstractmethod
    def delete_document(
        self,
        document_id: str,
        user_id: str,
    ) -> None:
        """Delete all chunks belonging to a document."""
        raise NotImplementedError
    @abstractmethod
    def add_documents(
        self,
        documents: List[Document],
        user_id: str,
        document_id: str,
    ) -> List[str]:
        """Store document chunks for a specific user."""
        raise NotImplementedError
