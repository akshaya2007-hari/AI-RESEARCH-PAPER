# Generated from: AI_RESEARCH_PAPER.ipynb
# Converted at: 2026-06-04T09:23:33.000Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

!pip install -q langchain
!pip install -q langchain-community
!pip install -q langchain-google-genai
!pip install -q langchain-text-splitters
!pip install -q pypdf

from google.colab import files

uploaded = files.upload()

pdf_file = list(uploaded.keys())[0]

print(pdf_file)


from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(pdf_file)

docs = loader.load()

print("Pages =", len(docs))

print(docs[0].page_content[:500])

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)

print("Chunks =", len(chunks))

import os

os.environ["GOOGLE_API_KEY"] = "AQ.Ab8RN6JhzJX5mktIXhflJqYMN8pBzMEOy05z7YQlDr-o6SVXyw"

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

print("Gemini Connected")

paper_text = ""

for chunk in chunks:
    paper_text += chunk.page_content

prompt = f"""
You are an AI Research Paper Summarizer Agent.

Analyze the paper and provide:

1. Paper Title
2. Abstract Summary
3. Key Findings
4. Methodology
5. Conclusion

Paper:

{paper_text[:15000]}
"""

response = llm.invoke(prompt)

print(response.content)

question = input("Ask a question: ")

qa_prompt = f"""
Answer only from the research paper.

Paper:
{paper_text[:15000]}

Question:
{question}
"""

answer = llm.invoke(qa_prompt)

print(answer.content)