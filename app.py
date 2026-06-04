import streamlit as st
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI

# ---------------------------------
# PAGE CONFIG
# ---------------------------------

st.set_page_config(
    page_title="AI Research Paper Summarizer",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Research Paper Summarizer")
st.write("Upload a research paper PDF and generate a summary.")

# ---------------------------------
# LOAD API KEY FROM STREAMLIT SECRETS
# ---------------------------------

try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except Exception:
    api_key = None

if not api_key:
    st.error("Google API Key not found.")
    st.info(
        "Go to App Settings → Secrets and add GOOGLE_API_KEY"
    )
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key

# ---------------------------------
# GEMINI MODEL
# ---------------------------------

try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3
    )
except Exception as e:
    st.error(f"Gemini Error: {e}")
    st.stop()

# ---------------------------------
# FILE UPLOADER
# ---------------------------------

uploaded_file = st.file_uploader(
    "Upload Research Paper PDF",
    type=["pdf"]
)

if uploaded_file:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp_file:

        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    # ---------------------------------
    # LOAD PDF
    # ---------------------------------

    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        st.success(
            f"PDF Loaded Successfully ({len(docs)} pages)"
        )

    except Exception as e:
        st.error(f"PDF Loading Error: {e}")
        st.stop()

    # ---------------------------------
    # TEXT SPLITTING
    # ---------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    paper_text = ""

    for chunk in chunks:
        paper_text += chunk.page_content + "\n"

    st.write(
        f"Document processed into {len(chunks)} chunks."
    )

    # ---------------------------------
    # SUMMARY SECTION
    # ---------------------------------

    if st.button("Generate Summary"):

        with st.spinner("Analyzing Paper..."):

            summary_prompt = f"""
            You are an AI Research Paper Summarizer.

            Analyze the research paper and provide:

            1. Paper Title
            2. Abstract Summary
            3. Key Findings
            4. Methodology
            5. Future Scope
            6. Conclusion

            Research Paper:

            {paper_text[:15000]}
            """

            try:

                response = llm.invoke(summary_prompt)

                st.subheader("📌 Summary")
                st.write(response.content)

            except Exception as e:

                st.error(
                    f"Summary Generation Error: {e}"
                )

    st.divider()

    # ---------------------------------
    # QUESTION ANSWERING
    # ---------------------------------

    st.subheader("🤖 Ask Questions About the Paper")

    question = st.text_input(
        "Enter your question"
    )

    if st.button("Get Answer"):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            qa_prompt = f"""
            Answer only from the research paper.

            Research Paper:

            {paper_text[:15000]}

            Question:

            {question}
            """

            try:

                answer = llm.invoke(qa_prompt)

                st.subheader("Answer")
                st.write(answer.content)

            except Exception as e:

                st.error(
                    f"Question Answering Error: {e}"
                )
