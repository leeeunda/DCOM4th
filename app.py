import streamlit as st
from dotenv import load_dotenv
from langchain.llms.openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.llms.openai import OpenAI
from langchain.chains.summarize import load_summarize_chain
import os

load_dotenv(verbose=True)
print(os.environ ['OPENAI_API_KEY'])

st.title('Text Summary App')
source_text = st.text_area("Source Text", label_visibility="collapsed", height=200)

llm=OpenAI(temperature=0)
result=llm("너는 누구야?")
print(result)

if st.button('입력'):
    try:
         with st.spinner('좀만 기다려~'):
            text_splitter = CharacterTextSplitter()
            texts = text_splitter.split_text(source_text)

              # Create Document objects for the texts (max 3 pages)
            docs = [Document(page_content=t) for t in texts[:3]]

            # Initialize the OpenAI module, load and run the summarize chain
            llm = OpenAI(temperature=0)
            chain = load_summarize_chain(llm, chain_type="map_reduce")
            summary = chain.run(docs)

            st.success(summary)
    except:
        pass

else:
    st.write('이제 곧 뜬다')
