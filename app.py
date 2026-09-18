import streamlit as st
import tempfile
import uuid

from langchain_core.messages import HumanMessage, AIMessage


#from agent import agent
from agent import agent
from ingestion import Create_embeddings, delete_index

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📄"
)

st.title("📄 PDF RAG Chatbot with Caching and Reranking")

if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False

if "current_file" not in st.session_state:
    st.session_state.current_file = None

if "current_index" not in st.session_state:
    st.session_state.current_index = None

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "display_messages" not in st.session_state:
    st.session_state.display_messages = []

if not st.session_state.pdf_uploaded:
    st.subheader("Upload a PDF to get started")
    st.info(
        "The chatbot currently works with text-based PDFs, "
        "not scanned/image-only PDFs."
    )

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"]
    )

    if uploaded_file:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name

        with st.spinner("Processing PDF and creating embeddings..."):
            result = Create_embeddings.invoke({
                "file_path": temp_path,
                "file_name": uploaded_file.name
            })

        st.session_state.current_file = uploaded_file.name
        st.session_state.current_index = result['index_name']
        st.session_state.pdf_uploaded = True

        st.success("PDF processed successfully!")
        st.rerun()

else:
    st.subheader(f"📄 {st.session_state.current_file}")

    config = {
        "configurable": {
            "thread_id": st.session_state.thread_id,
            "index_name": st.session_state.current_index
        },
        "recursion_limit": 20
    }

    col1, col2 = st.columns([3, 1])

    with col1:
        st.caption(
            f"Thread: {st.session_state.thread_id[:8]}..."
        )

    with col2:
        if st.button("＋ New Chat"):
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.display_messages = []
            st.rerun()

    for message in st.session_state.display_messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)

        elif isinstance(message, AIMessage) and message.content:
            with st.chat_message("assistant"):
                st.markdown(message.content)

    humanmsg = st.chat_input("Ask anything about your PDF...")

    if humanmsg:
        user_message = HumanMessage(content=humanmsg)

        st.session_state.display_messages.append(user_message)

        with st.chat_message("user"):
            st.markdown(humanmsg)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            for chunk in agent.stream(
                {"messages": [user_message]},
                config=config,
                stream_mode="messages"
            ):
                message, metadata = chunk

                if isinstance(message, AIMessage) and message.content:
                    full_response += message.content
                    response_placeholder.markdown(full_response)

            final_response = AIMessage(
                content=full_response
            )

        st.session_state.display_messages.append(
            final_response
        )

    st.divider()

    if st.button("End Session"):
        if st.session_state.current_index:
            delete_index.invoke({
                "index_name": st.session_state.current_index
            })

        st.session_state.clear()
        st.rerun()

