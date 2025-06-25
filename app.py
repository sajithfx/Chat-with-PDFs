import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

load_dotenv()

# Extract text from PDFs


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Split text into chunks


def get_text_chunks(text):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return splitter.split_text(text)

# Convert chunks into vector store (FAISS)


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(text_chunks, embedding=embeddings)

# Create LangChain retrieval-based conversation chain


def get_conversation_chain(vectorstore):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        return_messages=True
    )
    llm = OpenAI()
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

# Handle user input and display chat


def handle_userinput(user_question):
    response = st.session_state.chat.invoke({
        'question': user_question
    })

    st.session_state.chat_history.append(("You", user_question))
    st.session_state.chat_history.append(("Chat", response["answer"]))

    for speaker, message in st.session_state.chat_history:
        st.write(f"**{speaker}:** {message}")

# Main Streamlit app


def main():
    st.set_page_config("Chat with PDF")
    st.header("📖 Chat with PDF")

    if "chat" not in st.session_state:
        st.session_state.chat = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_question = st.text_input("Ask a question about your PDF:")

    if user_question and st.session_state.chat:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Upload your PDFs")
        pdf_docs = st.file_uploader("Upload files", accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.chat = get_conversation_chain(vectorstore)
                st.success("✅ Processing complete! You can now ask questions.")


if __name__ == '__main__':
    main()
