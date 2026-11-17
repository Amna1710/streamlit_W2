from pathlib import Path
from uuid import UUID
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

from rag_system.models.model_factory import ModelFactory
from rag_system.services.conversation_memory import (
    ConversationMemory,
)
from rag_system.services.retrieval_service import RetrievalService
from rag_system.retrievers.query_rewriter import QueryRewriter


PROMPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "prompts"
    / "rag_prompt.txt"
)


class ConversationalRAG:

    def __init__(self):

        if not PROMPT_PATH.exists():
            raise FileNotFoundError(
                f"RAG prompt not found: {PROMPT_PATH}"
            )

        self.retrieval_service = RetrievalService()
        self.memory = ConversationMemory(max_messages=20)
        self.query_rewriter = QueryRewriter()

        self.system_prompt = PROMPT_PATH.read_text(
            encoding="utf-8"
        )

    def _format_documents(
        self,
        documents: list[Document],
    ) -> str:

        formatted_documents = []

        for index, document in enumerate(documents, start=1):

            source = document.metadata.get(
                "source",
                "Unknown source",
            )

            page = document.metadata.get("page")

            source_info = f"Source: {source}"

            if page is not None:
                source_info += f", Page: {page}"

            formatted_documents.append(
                f"[Document {index}]\n"
                f"{source_info}\n"
                f"Content:\n{document.page_content}"
            )

        return "\n\n".join(formatted_documents)

    def _build_chain(
        self,
        provider: str | None = None,
        api_key: str | None = None,
    ):

        llm = ModelFactory.get_chat_model(
            provider=provider,
            temperature=0.3,
            api_key=api_key,
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])

        return prompt | llm | StrOutputParser()

    def answer(
        self,
        query: str,
        user_id: str,
        conversation_id: UUID,
        k: int = 4,
        provider: str | None = None,
        api_key: str | None = None,
    ) -> dict[str, Any]:

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        chat_history = self.memory.get_history(
            user_id=user_id,
            conversation_id=conversation_id,
        )

        search_query = self.query_rewriter.rewrite_query(
            question=query,
            chat_history=chat_history,
            provider=provider,
            api_key=api_key,
        )

        documents = self.retrieval_service.retrieve(
            query=search_query,
            user_id=user_id,
            k=k,
        )

        if not documents:

            fallback = (
                "I could not find sufficient information "
                "in your uploaded documents."
            )

            self.memory.add_message(
                user_id,
                conversation_id,
                "user",
                query,
            )

            self.memory.add_message(
                user_id,
                conversation_id,
                "assistant",
                fallback,
            )

            return {
                "answer": fallback,
                "rewritten_query": search_query,
                "sources": [],
                "total_sources": 0,
            }

        context = self._format_documents(documents)

        chain = self._build_chain(
            provider=provider,
            api_key=api_key,
        )

        response = chain.invoke(
            {
                "context": context,
                "question": query,
                "chat_history": chat_history,
            },
            config={
                "run_name": "conversational_rag_answer",
                "tags": ["rag", "answer-generation"],
                "metadata": {
                    "component": "conversational_rag"
                },
            },
        )

        self.memory.add_message(
            user_id,
            conversation_id,
            "user",
            query,
        )

        self.memory.add_message(
            user_id,
            conversation_id,
            "assistant",
            response,
        )

        sources = []

        for document in documents:

            sources.append({
                "source": document.metadata.get(
                    "source",
                    "Unknown source",
                ),
                "page": document.metadata.get("page"),
                "document_id": document.metadata.get("document_id"),
                "chunk_index": document.metadata.get("chunk_index"),
                "content": document.page_content,
            })

        return {
            "answer": response,
            "rewritten_query": search_query,
            "sources": sources,
            "total_sources": len(sources),
        }

    def clear_conversation(
        self,
        user_id: str,
        conversation_id: UUID,
    ) -> bool:

        return self.memory.clear_history(
            user_id=user_id,
            conversation_id=conversation_id,
        )