import streamlit as st
from rag_pipeline import get_answer

# PAGE CONFIG
st.set_page_config(
    page_title="AI Teaching Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI Teaching Assistant")

# CHAT HISTORY
if "messages" not in st.session_state:
    st.session_state.messages = []

# DISPLAY OLD MESSAGES
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# USER INPUT
query = st.chat_input("Ask your question...")

if query:

    # USER MESSAGE
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.markdown(query)

    # ASSISTANT RESPONSE
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):
            response = get_answer(query)

        st.markdown(response)

    # SAVE RESPONSE
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })