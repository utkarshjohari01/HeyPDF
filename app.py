import streamlit as st
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS

from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract

# Use default tesseract command for Docker compatibility
pytesseract.pytesseract.tesseract_cmd = "tesseract"

from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_bytes = pdf.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            local_text = ""

            for i, page in enumerate(doc):
                content = page.get_text()
                if content.strip():
                    local_text += content
                else:
                    st.warning(f"No text found on page {i+1} of {pdf.name}. Using OCR...")
                    images = convert_from_bytes(
                        pdf_bytes,
                        first_page=i+1,
                        last_page=i+1
                    )
                    ocr_text = ""
                    for image in images:
                        ocr_text += pytesseract.image_to_string(image)
                    local_text += ocr_text

            text += local_text
        except Exception as e:
            st.error(f"Error reading {pdf.name}: {e}")
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,  
        chunk_overlap=200
    )
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    if not text_chunks:
        raise ValueError("No text chunks provided to build the vector store.")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversation_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    Make sure to only use the information from the context.
    If the answer is not in the provided context, say "Answer is not available in the context",
    don't try to make up an answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-latest", temperature=0.7)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def handle_user_question(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversation_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.session_state.chat_history.append({"user": user_question, "bot": response["output_text"]})

def main():
    st.set_page_config(page_title="📘HeyPDF", layout="wide")

    # Initialize session flags
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "clear_input_flag" not in st.session_state:
        st.session_state.clear_input_flag = False
    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False

    with st.sidebar:
        st.markdown("""        
            <h1 style='text-align: center;'>📂 Upload PDFs</h1>
        """, unsafe_allow_html=True)

        pdf_docs = st.file_uploader("Choose PDF files:", type="pdf", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing PDFs..."):
                raw_text = get_pdf_text(pdf_docs)
                if not raw_text.strip():
                    st.error("No text extracted from the uploaded PDFs.")
                    st.session_state.pdf_processed = False
                    return
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("✅ PDF files processed and vector store created!")
                st.session_state.pdf_processed = True

    st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h1 style='color:#4A90E2;'>📄 Chat With Your PDFs</h1>
            <p style='font-size: 18px;'>Upload multiple PDFs and ask anything.</p>
        </div>
    """, unsafe_allow_html=True)

    with st.form(key="question_form"):
        st.markdown("""<div style='margin-top: 20px;'><strong>Ask a Question:</strong></div>""", unsafe_allow_html=True)
        user_question = st.text_input("Ask a Question", value="", key="input_field", label_visibility="collapsed")
        submit = st.form_submit_button("Ask")
        if submit and user_question.strip():
            if not st.session_state.pdf_processed:
                st.warning("Please upload and process PDFs before asking questions.")
            else:
                handle_user_question(user_question)
                st.session_state.clear_input_flag = True

    if st.session_state.chat_history:
        st.markdown("<div style='padding: 10px;'>", unsafe_allow_html=True)
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            st.markdown(f"<div style='margin-bottom: 10px; color: #555;'><strong>You:  </strong> {chat['user']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='margin-left: 20px;'><strong>Response:  </strong> {chat['bot']}</div>", unsafe_allow_html=True)
            if i < len(st.session_state.chat_history) - 1:
                st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
