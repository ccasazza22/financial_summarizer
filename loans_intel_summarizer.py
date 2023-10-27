import streamlit as st
from pathlib import Path
import os 
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ReduceDocumentsChain, MapReduceDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chat_models import ChatOpenAI
from langsmith import Client
from langchain.document_loaders import PyPDFLoader
from langchain import PromptTemplate, LLMChain
from langchain.chains import LLMChain
import langsmith
from langchain.document_loaders import Docx2txtLoader
from dotenv import load_dotenv
import tempfile
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain import OpenAI
from langchain.vectorstores import FAISS




load_dotenv()  # take environment variables from .env.
os.environ['OPENAI_API_KEY'] = st.secrets["general"]["OPENAI_API_KEY"]
os.environ["LANGCHAIN_API_KEY"]= st.secrets["general"]["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"]="loans_intel_transcripts"

client = Client()


llm = ChatOpenAI(model="gpt-4",temperature=0)

# Map
from langchain import hub
map_template = hub.pull("casazza/summarizer-a:4c223487", api_url="https://api.hub.langchain.com")


#map_prompt = PromptTemplate.from_template(prompt=map_template)
map_chain = LLMChain(llm=llm, prompt=map_template)

# Reduce

reduce_prompt= hub.pull("casazza/reduce-template",api_url="https://api.hub.langchain.com")

# Run chain
reduce_chain = LLMChain(llm=ChatOpenAI(model="gpt-4",max_tokens=4000), prompt=reduce_prompt)

# Takes a list of documents, combines them into a single string, and passes this to an LLMChain
combine_documents_chain = StuffDocumentsChain(
    llm_chain=reduce_chain, document_variable_name="doc_summaries"
)

# Combines and iteravely reduces the mapped documents
reduce_documents_chain = ReduceDocumentsChain(
    # This is final chain that is called.
    combine_documents_chain=combine_documents_chain,
    # If documents exceed context for `StuffDocumentsChain`
    collapse_documents_chain=combine_documents_chain,
    # The maximum number of tokens to group documents into.
    token_max=3500,
)

# Combining documents by mapping a chain over them, then combining results
map_reduce_chain = MapReduceDocumentsChain(
    # Map chain
    llm_chain=map_chain,
    # Reduce chain
    reduce_documents_chain=reduce_documents_chain,
    # The variable name in the llm_chain to put the documents in
    document_variable_name="docs",
    # Return the results of the map steps in the output
    return_intermediate_steps=False, tags=["Streamlit"]
)

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=100
)


@st.cache_data
def process_file(pages):
    try:
        # assuming text_splitter.split_text and map_reduce_chain.run accept text 
        split_docs = text_splitter.split_documents(pages)
        output = map_reduce_chain.run(split_docs)

        # Embed documents once they are processed
        embeddings = OpenAIEmbeddings()
        retriever_docs = FAISS.from_documents(split_docs, embeddings)
        
        return output, retriever_docs
    except Exception as e:
        st.error(f"An error occurred during processing: {str(e)}")

def process_query(query, retriever_docs):
    try:
        chain = load_qa_chain(OpenAI(temperature=0), chain_type="refine")
        return chain({"input_documents": retriever_docs, "question": query}, return_only_outputs=True)
    except Exception as e:
        st.error(f"An error occurred during processing the query: {str(e)}")


def main():

    uploaded_file = st.file_uploader("Choose a file", type="docx")

    if uploaded_file is not None:
        with st.spinner('Processing...This may take a few minutes'):
            try:
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_file.read())
                tfile.close()

                loader = Docx2txtLoader(tfile.name)
                pages = loader.load_and_split()
               
                st.session_state.output, st.session_state.retriever_docs = process_file(pages)

                st.subheader('Your summarized document:')
                st.code(st.session_state.output, language='')

                # Add a section for follow-up questions
                st.subheader('Ask a follow-up question:')
                query = st.text_input("Please enter your question here")
                submit_button = st.button("Submit Question")

                if submit_button:
                    def process_question():
                        with st.spinner('Processing your question...this may take a few minutes'):
                            try:
                                answer = process_query(query, st.session_state.retriever_docs)
                                st.subheader('Answer:')
                                st.write(answer)
                            except Exception as e:
                                st.error(f"An error occurred: {str(e)}")
                    process_question()

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
            finally:
                os.unlink(tfile.name)

if __name__ == "__main__":
    main()