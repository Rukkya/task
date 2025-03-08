import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

# Initialize session state
if 'conversation' not in st.session_state:
    st.session_state.conversation = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def initialize_chain(uploaded_files):
    # Initialize document list
    documents = []
    
    # Process each uploaded file
    for file in uploaded_files:
        # Get file extension
        file_extension = os.path.splitext(file.name)[1]
        
        # Save the file temporarily
        with open(file.name, "wb") as f:
            f.write(file.getbuffer())
        
        # Load the file based on its extension
        if file_extension.lower() == ".pdf":
            loader = PyPDFLoader(file.name)
            documents.extend(loader.load())
        elif file_extension.lower() == ".txt":
            loader = TextLoader(file.name)
            documents.extend(loader.load())
            
        # Remove temporary file
        os.remove(file.name)
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    
    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(splits, embeddings)
    
    # Create conversation chain
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
    )
    
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        verbose=True
    )
    
    return conversation_chain

# Streamlit UI
st.title("📚 Document Chatbot")

# File uploader
uploaded_files = st.file_uploader(
    "Upload your documents (PDF or TXT)",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("Process Documents"):
        with st.spinner("Processing documents..."):
            st.session_state.conversation = initialize_chain(uploaded_files)
        st.success("Documents processed successfully!")

# Chat interface
if st.session_state.conversation:
    user_question = st.text_input("Ask a question about your documents:")
    
    if user_question:
        with st.spinner("Thinking..."):
            response = st.session_state.conversation({"question": user_question})
            st.session_state.chat_history.append((user_question, response['answer']))

    # Display chat history
    if st.session_state.chat_history:
        st.subheader("Chat History")
        for q, a in st.session_state.chat_history:
            st.write(f"👤 **You:** {q}")
            st.write(f"🤖 **Bot:** {a}")
            st.write("---")