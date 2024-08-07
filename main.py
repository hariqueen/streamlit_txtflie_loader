import os
import openai
import streamlit as st
import tempfile
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

# OpenAI API 키 설정
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
else:
    st.write(f"OpenAI API Key: {openai_api_key}")  # 디버깅을 위해 API 키 출력 (보안을 위해 실제 사용 시 제거)
    openai.api_key = openai_api_key

    # 제목
    st.title("txt파일을 업로드 해보세요.")
    st.write("---")

    # 파일 업로드
    uploaded_file = st.file_uploader("텍스트 파일을 올려주세요!", type=['txt'])
    st.write("---")

    def pdf_to_document(uploaded_file):
        temp_dir = tempfile.TemporaryDirectory()
        temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
        with open(temp_filepath, "wb") as f:
            f.write(uploaded_file.getvalue())
        loader = TextLoader(temp_filepath)
        pages = loader.load_and_split()
        return pages

    # 업로드된 후 동작
    if uploaded_file is not None:
        pages = pdf_to_document(uploaded_file)

        # Split
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,  # 300글자 단위 정도
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
        )

        texts = text_splitter.split_documents(pages)

        # Embedding
        embeddings_model = OpenAIEmbeddings()

        # load it into chroma
        db = Chroma.from_documents(texts, embeddings_model)

        # Question
        st.header("첨부된 파일과 관련된 질문해보세요!")
        question = st.text_input('질문을 입력하세요')

        if st.button('질문하기'):
            with st.spinner('Wait for it...'):
                llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
                qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
                try:
                    result = qa_chain({"query": question})
                    st.write(result["result"])
                except openai.error.OpenAIError as e:
                    st.error(f"OpenAI API 요청 실패: {e}")
                    st.write(f"Error details: {e}")
