from langchain.embeddings.openai import OpenAIEmbeddings
import streamlit as st
from pathlib import Path
import os 
from langchain.chains.mapreduce import MapReduceChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ReduceDocumentsChain, MapReduceDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langsmith import Client
from langchain.document_loaders import PyPDFLoader
from langchain import PromptTemplate, LLMChain
from langchain.chains import LLMChain
import langsmith
from langchain import chat_models, prompts, smith
import langsmith
from langchain.document_loaders import Docx2txtLoader
from dotenv import load_dotenv
import tempfile
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain import OpenAI




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
map_template = hub.pull("casazza/summarizer-a:daec89c5", api_url="https://api.hub.langchain.com")


#map_prompt = PromptTemplate.from_template(prompt=map_template)
map_chain = LLMChain(llm=llm, prompt=map_template)

# Reduce
reduce_template = """The following is set of summaries:
{doc_summaries}\n

-------------------------------------------------------------------\n

Based on these summaries, please do the following: 
(1) Prioritize information on Market Conditions and performance, cost management and savings, business strategy and future outlook, regulatory changes and impact. 
(2) Divide the relevant information into the following buckets:

Guidance:
Involves forward-looking financial metrics including revenue, EBITDA, free cash flow, leverage, KPIs, Capex, taxes, interest, and net working capital. It also covers market strategies, major product changes, industry trends, business performance, and operational updates.

Mergers & Acquisitions (M&A): 
Discusses M&A activities, providing reasons, funding details, and related financials. It includes commentary on buyer/seller expectations and future M&A plans.

Capital Allocation: 
Covers refinancing plans, recent transactions or issues, sponsor equity, availability of baskets in credit agreement, hedging activities, and maturity details of funding sources.

Other Information: 
Addresses management changes, ratings updates, and current post-quarter-end levels for cash and RC balance.

Make your final summary: 
1) As long as it needs to be to be comprehensive
2) In bulleted form :
3) Refer to the company by their name 
"""

reduce_prompt = PromptTemplate.from_template(reduce_template)

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
    return_intermediate_steps=False, tags=["Summarizer A"]
)

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=0
)

def process_file(pages):
    try:
        # assuming text_splitter.split_text and map_reduce_chain.run accept text 
        split_docs = text_splitter.split_documents(pages)
        output = map_reduce_chain.run(split_docs)

        # Embed documents once they are processed
        embeddings = OpenAIEmbeddings()
        retriever_docs = Chroma.from_texts(split_docs, embeddings, metadatas=[{"source": str(i)} for i in range(len(split_docs))]).as_retriever()
        
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
    st.title("Loans Intel Earnings Call Summarizer")
    st.subheader("Welcome to the Document Summarizer Application!")
    st.write("To get started, please upload a Word (.docx) file. The application will process the file and provide a summarized version of the document contents.")

    uploaded_file = st.file_uploader("Choose a Word file", type=['docx'])

    if uploaded_file is not None:
        with st.spinner('Processing...This may take a few minutes'):
            try:
                # Create a temporary file and save the uploaded file's content to this temporary file
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_file.read())
                tfile.close()

                # Now that you have a real file on your filesystem, you can pass its path to Docx2txtLoader
                print(f"Loading {tfile.name}")
                loader = Docx2txtLoader(tfile.name)
                pages = loader.load_and_split()
               
                # Process file
                output, retriever_docs = process_file(pages)

                # Show output
                st.subheader('Your summarized document:')
                st.code(output, language='')

                # Add a section for follow-up questions
                st.subheader('Ask a follow-up question:')
                query = st.text_input("Please enter your question here")

                if query:
                    with st.spinner('Processing your question...this may take a few minutes'):
                        try:
                            answer = process_query(query, retriever_docs)  # Pass the retriever_docs to the process_query function
                            st.subheader('Answer:')
                            st.write(answer)
                
                        except Exception as e:
                            st.error(f"An error occurred: {str(e)}")
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
            finally:
                # Delete the temporary file
                os.unlink(tfile.name)

if __name__ == "__main__":
    main()
