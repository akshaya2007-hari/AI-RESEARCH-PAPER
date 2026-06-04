import streamlit as st
import os
import tempfile

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI

# Load API key
load_dotenv()

# Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)

st.set_page_config(page_title="AI Research Paper Summarizer")

st.title("📄 AI Research Paper Summarizer")

uploaded_file = st.file_uploader(
    "Upload Research Paper",
    type="pdf"
)

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    paper_text = ""

    for chunk in chunks:
        paper_text += chunk.page_content

    if st.button("Generate Summary"):

        with st.spinner("Analyzing Paper..."):

            prompt = f"""
            You are an AI Research Paper Summarizer.

            Analyze the paper and provide:

            1. Paper Title
            2. Abstract Summary
            3. Key Findings
            4. Methodology
            5. Future Scope
            6. Conclusion

            Paper:

            {paper_text[:15000]}
            """

            response = llm.invoke(prompt)

            st.subheader("Summary")
            st.write(response.content)

    st.divider()

    question = st.text_input(
        "Ask a question about the paper"
    )

    if st.button("Get Answer"):

        qa_prompt = f"""
        Answer only from the research paper.

        Paper:

        {paper_text[:15000]}

        Question:

        {question}
        """

        answer = llm.invoke(qa_prompt)

        st.subheader("Answer")
        st.write(answer.content)
