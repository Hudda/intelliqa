import tempfile
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from query import get_answer
from upsert_db import upsert_in_pinecone



def main():
    st.set_page_config(
        layout="wide",
        page_title="IntelliQA",
        page_icon=":graph:"
    )
    st.sidebar.image('./logo.png', use_container_width=True)
    with st.sidebar.expander("Expand Me"):
        st.markdown("""
    This application allows you to upload a PDF file, extract its content into a vector database, and perform queries using natural language.
    It leverages LangChain and OpenAI's GPT models to generate answers to your queries.
    """)
    st.title("IntelliQA: Realtime RAG App")

    company_name = st.text_input("Please enter the company name:")

    if company_name:
        uploaded_files = st.file_uploader("Please select PDF files.", type="pdf", accept_multiple_files=True)

        if uploaded_files:
            process = st.button("Upload Files")
            if process:
                with st.spinner("Uploading the files..."):
                    texts = []
                    for file in uploaded_files:
                        # st.write(f"File: {file.name}")
                        # Save uploaded file to temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(file.read())
                            tmp_file_path = tmp_file.name

                        # Load and split the PDF
                        loader = PyPDFLoader(tmp_file_path)
                        pages = loader.load_and_split()

                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
                        docs = text_splitter.split_documents(pages)
                        texts += [doc.page_content.replace("\n", "") for doc in docs]
                    
                    upsert_in_pinecone(texts, company_name)

                    st.success("Files uploaded successfully!")
        else:
            st.button("Upload Files", disabled=True)

        
        st.subheader("Ask a Question")
        with st.form(key='question_form'):
            question = st.text_input("Enter your question:")
            submit_button = st.form_submit_button(label='Submit')

        if submit_button and question:
            with st.spinner("Generating answer..."):
                answer = get_answer(question, company_name)
                st.write("\n**Answer:**\n" + answer)

if __name__ == "__main__":
    main()