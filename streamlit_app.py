
import os
import uuid

import requests
import streamlit as st


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="RAG Knowledge Base",
    page_icon="📚",
    layout="wide",
)

st.title("📚 RAG Knowledge Base")
st.caption(
    "Upload documents, retrieve information, chat with AI, "
    "and evaluate answers."
)


# --------------------------------------------------
# API CONFIGURATION
# --------------------------------------------------

try:
    API_BASE_URL = st.secrets["API_BASE_URL"].rstrip("/")
except Exception:
    API_BASE_URL = os.getenv(
        "API_BASE_URL",
        "https://streamlitw2-production.up.railway.app",
    ).rstrip("/")


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

defaults = {
    "user_id": "",
    "conversation_id": str(uuid.uuid4()),
    "messages": [],
    "last_retrieval": None,
    "last_chat_answer": "",
    "last_chat_question": "",
    "last_chat_context": "",
    "evaluation_result": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --------------------------------------------------
# API HELPER FUNCTIONS
# --------------------------------------------------

def get_headers():
    """Return the current user's identity header."""

    if not st.session_state.user_id:
        return {}

    return {
        "X-User-ID": st.session_state.user_id,
    }


def get_llm_headers():
    """
    Return user identity and the selected user's API key.

    The API key is sent only with LLM requests.
    It is never added to the request body or URL.
    """

    headers = get_headers()

    selected_provider = st.session_state.get(
        "selected_provider",
        "gemini",
    )

    if selected_provider == "gemini":
        api_key = st.session_state.get(
            "gemini_api_key_input",
            "",
        )

    elif selected_provider == "openai":
        api_key = st.session_state.get(
            "openai_api_key_input",
            "",
        )

    else:
        api_key = ""

    if api_key and api_key.strip():
        headers["X-LLM-API-Key"] = api_key.strip()

    return headers


def validate_provider_key():
    """Check whether the selected provider has its required API key."""

    selected_provider = st.session_state.get(
        "selected_provider",
        "gemini",
    )

    if selected_provider == "gemini":

        api_key = st.session_state.get(
            "gemini_api_key_input",
            "",
        )

        if not api_key or not api_key.strip():
            st.warning(
                "Please enter your personal Gemini API key "
                "in the sidebar."
            )
            return False

    elif selected_provider == "openai":

        api_key = st.session_state.get(
            "openai_api_key_input",
            "",
        )

        if not api_key or not api_key.strip():
            st.warning(
                "Please enter your personal OpenAI API key "
                "in the sidebar."
            )
            return False

    return True


def show_api_error(response):
    """Display a backend API error."""

    try:
        detail = response.json().get(
            "detail",
            response.text,
        )

    except (ValueError, AttributeError):
        detail = response.text

    st.error(
        f"API Error ({response.status_code}): {detail}"
    )


def make_request(method, endpoint, **kwargs):
    """Send an HTTP request to the FastAPI backend."""

    try:

        response = requests.request(
            method=method,
            url=f"{API_BASE_URL}{endpoint}",
            timeout=600,
            **kwargs,
        )

        if response.ok:
            return response.json()

        show_api_error(response)

    except requests.RequestException as exc:
        st.error(f"Backend connection failed: {exc}")

    except ValueError as exc:
        st.error(f"Invalid API response: {exc}")

    return None


def display_sources(sources):
    """Display source citations returned by RAG."""

    if not sources:
        return

    with st.expander(f"View Sources ({len(sources)})"):

        for index, source in enumerate(sources, start=1):

            st.markdown(
                f"**Source {index}: "
                f"{source.get('source', 'Unknown')}**"
            )

            if source.get("page") is not None:
                st.caption(
                    f"Page: {source['page']}"
                )

            if source.get("document_id"):
                st.caption(
                    f"Document ID: {source['document_id']}"
                )

            if source.get("chunk_index") is not None:
                st.caption(
                    f"Chunk: {source['chunk_index']}"
                )

            st.write(
                source.get("content", "")
            )

            st.divider()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("Your Account")

    entered_user_id = st.text_input(
        "Enter your existing User ID",
        value=st.session_state.user_id,
        placeholder="USR-...",
    )

    st.session_state.user_id = entered_user_id.strip()

    if st.session_state.user_id:

        st.success("User ID is active.")

        st.code(
            st.session_state.user_id
        )

    else:

        st.info(
            "Upload your first document to generate a User ID."
        )

    st.divider()

    st.subheader("AI Settings")

    provider = st.selectbox(
        "LLM Provider",
        [
            "gemini",
            "openai",
            "anthropic",
            "ollama",
        ],
        index=0,
        key="selected_provider",
    )

    # ----------------------------------------------
    # PERSONAL API KEY INPUT
    # ----------------------------------------------

    if provider == "gemini":

        st.text_input(
            "Your Gemini API Key",
            type="password",
            placeholder="Enter your Gemini API key",
            key="gemini_api_key_input",
            help=(
                "Enter your personal Gemini API key. "
                "It is sent to the backend with your "
                "LLM requests."
            ),
        )

        st.caption(
            "Using your own Gemini API key."
        )

    elif provider == "openai":

        st.text_input(
            "Your OpenAI API Key",
            type="password",
            placeholder="Enter your OpenAI API key",
            key="openai_api_key_input",
            help=(
                "Enter your personal OpenAI API key. "
                "It is sent to the backend with your "
                "LLM requests."
            ),
        )

        st.caption(
            "Using your own OpenAI API key."
        )

    elif provider == "anthropic":

        st.info(
            "Anthropic currently uses the API key "
            "configured in your backend .env file."
        )

    elif provider == "ollama":

        st.info(
            "Ollama runs locally and does not require "
            "an API key."
        )

    st.divider()

    retrieval_k = st.slider(
        "Number of retrieved chunks",
        min_value=1,
        max_value=20,
        value=4,
    )

    st.divider()

    if st.button(
        "Start a New Conversation",
        use_container_width=True,
    ):

        st.session_state.conversation_id = str(uuid.uuid4())

        st.session_state.messages = []

        st.session_state.last_chat_answer = ""
        st.session_state.last_chat_question = ""
        st.session_state.last_chat_context = ""

        st.session_state.evaluation_result = None

        st.rerun()


# --------------------------------------------------
# MAIN TABS
# --------------------------------------------------

(
    upload_tab,
    chat_tab,
    retrieval_tab,
    evaluation_tab,
    history_tab,
) = st.tabs(
    [
        "📁 Upload Documents",
        "💬 Chat",
        "🔎 Retrieval",
        "📊 Evaluation",
        "🕘 Conversation History",
    ]
)


# ==================================================
# TAB 1: DOCUMENT UPLOAD
# ==================================================

with upload_tab:

    st.header("Upload Documents")

    uploaded_files = st.file_uploader(
        "Choose PDF, DOCX, TXT, or Markdown files",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )

    if st.button(
        "Upload Documents",
        type="primary",
        disabled=not uploaded_files,
    ):

        for uploaded_file in uploaded_files:

            with st.spinner(
                f"Processing {uploaded_file.name}..."
            ):

                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type or "application/octet-stream",
                    )
                }

                result = make_request(
                    "POST",
                    "/documents/upload",
                    headers=get_headers(),
                    files=files,
                )

                if result:

                    st.session_state.user_id = result["user_id"]

                    st.success(
                        f"Uploaded: {result['filename']}"
                    )

                    st.write(
                        f"Chunks created: {result['chunks_created']}"
                    )

                    st.write(
                        f"Document ID: {result['document_id']}"
                    )

                    st.info(
                        f"Your User ID: {result['user_id']}"
                    )

                    st.code(
                        result["user_id"]
                    )


# ==================================================
# TAB 2: CONVERSATIONAL RAG CHAT
# ==================================================

with chat_tab:

    st.header("Ask Questions")

    if not st.session_state.user_id:

        st.warning(
            "Upload a document first to generate your User ID."
        )

    else:

        st.caption(
            f"Conversation ID: {st.session_state.conversation_id}"
        )

        # Display previous messages.

        for message in st.session_state.messages:

            with st.chat_message(message["role"]):

                st.markdown(
                    message["content"]
                )

                display_sources(
                    message.get("sources", [])
                )

        prompt = st.chat_input(
            "Ask something about your uploaded documents..."
        )

        if prompt and validate_provider_key():

            st.session_state.messages.append({
                "role": "user",
                "content": prompt,
            })

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):

                with st.spinner(
                    "Searching documents and generating an answer..."
                ):

                    payload = {
                        "query": prompt,
                        "k": retrieval_k,
                        "provider": provider,
                        "conversation_id": (
                            st.session_state.conversation_id
                        ),
                    }

                    result = make_request(
                        "POST",
                        "/chat/answer",
                        headers=get_llm_headers(),
                        json=payload,
                    )

                    if result:

                        answer = result["answer"]

                        sources = result.get(
                            "sources",
                            [],
                        )

                        st.markdown(answer)

                        display_sources(sources)

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        })

                        st.session_state.last_chat_question = prompt

                        st.session_state.last_chat_answer = answer

                        st.session_state.last_chat_context = "\n\n".join(
                            source.get("content", "")
                            for source in sources
                        )


# ==================================================
# TAB 3: DOCUMENT RETRIEVAL
# ==================================================

with retrieval_tab:

    st.header("Document Retrieval")

    st.write(
        "Search your uploaded documents without generating "
        "an AI answer."
    )

    if not st.session_state.user_id:

        st.warning(
            "Upload a document first."
        )

    else:

        retrieval_query = st.text_area(
            "Enter your search query",
            placeholder="What is Retrieval Augmented Generation?",
            key="retrieval_query",
        )

        if st.button(
            "Retrieve Documents",
            type="primary",
        ):

            if not retrieval_query.strip():

                st.warning(
                    "Please enter a search query."
                )

            else:

                payload = {
                    "query": retrieval_query,
                    "k": retrieval_k,
                }

                with st.spinner(
                    "Retrieving relevant document chunks..."
                ):

                    result = make_request(
                        "POST",
                        "/chat/retrieve",
                        headers=get_headers(),
                        json=payload,
                    )

                if result:

                    st.session_state.last_retrieval = result

                    st.success(
                        f"Retrieved {result['total_results']} chunks."
                    )

        result = st.session_state.last_retrieval

        if result:

            st.divider()

            st.write(
                f"Query: {result['query']}"
            )

            st.write(
                f"Total results: {result['total_results']}"
            )

            for index, chunk in enumerate(
                result["results"],
                start=1,
            ):

                with st.expander(
                    f"Retrieved Chunk {index}"
                ):

                    st.markdown("**Content**")

                    st.write(
                        chunk["content"]
                    )

                    st.markdown("**Retrieval Score**")

                    if chunk.get("score") is not None:
                        st.write(chunk["score"])
                    else:
                        st.write("Not available")

                    st.write(
                        f"Score Type: "
                        f"{chunk.get('score_type') or 'Not available'}"
                    )

                    st.markdown("**Metadata**")

                    st.json(
                        chunk.get("metadata", {})
                    )


# ==================================================
# TAB 4: ANSWER EVALUATION
# ==================================================

with evaluation_tab:

    st.header("RAG Answer Evaluation")

    st.write(
        "Evaluate an AI-generated answer for relevance "
        "and faithfulness."
    )

    # Use the latest chat answer.

    if st.session_state.last_chat_answer:

        if st.button(
            "Use Latest Chat Answer"
        ):

            st.session_state.eval_question = (
                st.session_state.last_chat_question
            )

            st.session_state.eval_answer = (
                st.session_state.last_chat_answer
            )

            st.session_state.eval_context = (
                st.session_state.last_chat_context
            )

            st.rerun()

    with st.form("evaluation_form"):

        question = st.text_area(
            "Question",
            value=st.session_state.get(
                "eval_question",
                "",
            ),
            max_chars=2000,
        )

        answer = st.text_area(
            "Generated Answer",
            value=st.session_state.get(
                "eval_answer",
                "",
            ),
            max_chars=10000,
        )

        context = st.text_area(
            "Reference Context",
            value=st.session_state.get(
                "eval_context",
                "",
            ),
            max_chars=30000,
            help=(
                "Provide the document context used "
                "to generate the answer."
            ),
        )

        evaluate_button = st.form_submit_button(
            "Evaluate Answer",
            type="primary",
        )

    if evaluate_button:

        if not question.strip() or not answer.strip():

            st.warning(
                "Question and answer are required."
            )

        elif not context.strip():

            st.warning(
                "Please provide reference context for evaluation."
            )

        elif validate_provider_key():

            payload = {
                "question": question,
                "answer": answer,
                "context": context,
            }

            with st.spinner(
                "Evaluating answer..."
            ):

                result = make_request(
                    "POST",
                    "/chat/evaluate",
                    headers=get_llm_headers(),
                    params={
                        "provider": provider,
                    },
                    json=payload,
                )

            if result:

                st.session_state.evaluation_result = result

    result = st.session_state.evaluation_result

    if result:

        st.divider()

        st.subheader("Evaluation Results")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Relevance Score",
                f"{result['relevance_score']:.2%}",
            )

            st.info(
                result["relevance_explanation"]
            )

        with col2:

            st.metric(
                "Faithfulness Score",
                f"{result['faithfulness_score']:.2%}",
            )

            st.info(
                result["faithfulness_explanation"]
            )

        if result["passed"]:

            st.success(
                "Evaluation Passed"
            )

        else:

            st.warning(
                "Evaluation Failed"
            )

        st.write(
            f"Evaluation Model: {result['evaluation_model']}"
        )


# ==================================================
# TAB 5: CONVERSATION HISTORY
# ==================================================

with history_tab:

    st.header("Conversation History")

    st.write(
        "View the messages stored in your current Streamlit session."
    )

    if not st.session_state.user_id:

        st.warning(
            "Enter your User ID first."
        )

    else:

        st.write(
            f"Conversation ID: {st.session_state.conversation_id}"
        )

        if not st.session_state.messages:

            st.info(
                "No messages in the current conversation."
            )

        else:

            st.write(
                f"Total messages: {len(st.session_state.messages)}"
            )

            for index, message in enumerate(
                st.session_state.messages,
                start=1,
            ):

                with st.expander(
                    f"{index}. {message['role'].capitalize()}"
                ):

                    st.markdown(
                        message["content"]
                    )

                    display_sources(
                        message.get("sources", [])
                    )

        st.divider()

        st.subheader("Clear Conversation")

        st.warning(
            "This will request deletion of the backend conversation "
            "memory and clear the messages in this Streamlit session."
        )

        if st.button(
            "Clear Conversation History",
            type="primary",
        ):

            conversation_id = (
                st.session_state.conversation_id
            )

            result = make_request(
                "DELETE",
                f"/chat/conversation/{conversation_id}",
                headers=get_headers(),
            )

            if result is not None:

                st.info(
                    result["message"]
                )

                st.session_state.messages = []

                st.session_state.conversation_id = str(uuid.uuid4())

                st.session_state.last_chat_answer = ""
                st.session_state.last_chat_question = ""
                st.session_state.last_chat_context = ""

                st.session_state.evaluation_result = None

                st.success(
                    "Local conversation history cleared."
                )

                st.rerun()
