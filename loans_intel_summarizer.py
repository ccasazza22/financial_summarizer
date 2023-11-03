from langchain.callbacks.manager import collect_runs
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
from langchain.chains import RetrievalQA
from streamlit_feedback import streamlit_feedback
#from expression import get_expression_chain





load_dotenv()  # take environment variables from .env.
os.environ['OPENAI_API_KEY'] = st.secrets['general']["OPENAI_API_KEY"]
os.environ["LANGCHAIN_API_KEY"]= st.secrets['general']["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"]="loans_intel_transcripts"

client = Client()


llm = ChatOpenAI(model="gpt-3.5-turbo",temperature=0,max_tokens=350)

# Map
from langchain import hub
map_template = hub.pull("casazza/summarizer-a", api_url="https://api.hub.langchain.com")


#map_prompt = PromptTemplate.from_template(prompt=map_template)
map_chain = LLMChain(llm=llm, prompt=map_template)

# Reduce

reduce_prompt= hub.pull("casazza/reduce-template",api_url="https://api.hub.langchain.com")
collapse_prompt= hub.pull("casazza/collapse_prompt:90a46122",api_url="https://api.hub.langchain.com")

# Run chain
reduce_chain = LLMChain(llm=ChatOpenAI(model="gpt-4",max_tokens=4000), prompt=reduce_prompt)
collapse_chain=LLMChain(llm=ChatOpenAI(model="gpt-4",max_tokens=4000),prompt=collapse_prompt)

inputs = []

# Takes a list of documents, combines them into a single string, and passes this to an LLMChain
combine_documents_chain = StuffDocumentsChain(
    llm_chain=reduce_chain, document_variable_name="doc_summaries"
)

collapse_documents_chain=StuffDocumentsChain(
    llm_chain=collapse_chain, document_variable_name="doc_summaries"
)

# Combines and iteravely reduces the mapped documents
reduce_documents_chain = ReduceDocumentsChain(
    # This is final chain that is called.
    combine_documents_chain=combine_documents_chain,
    # If documents exceed context for `StuffDocumentsChain`
    collapse_documents_chain=collapse_documents_chain,
    # The maximum number of tokens to group documents into.
    token_max=3000,
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


@st.cache_data(ttl=300,max_entries=1)
def process_file(_pages):
    try:
        # assuming text_splitter.split_text and map_reduce_chain.run accept text 
        split_docs = text_splitter.split_documents(_pages)
        #output = map_reduce_chain.run(split_docs)

        with collect_runs() as cb:
            output = map_reduce_chain.run(split_docs)
            st.session_state.run_id = cb.traced_runs[0].id
        texts = [doc.page_content for doc in _pages]       
        return output, ''.join(texts)
    except Exception as e:
        st.error(f"An error occurred during processing: {str(e)}")



qa_prompt = hub.pull("casazza/qa-finance-prompt")

def process_query(query, _pages):
    try:
        # Embed documents once they are processed
        split_docs = text_splitter.split_documents(_pages)
        retriever = FAISS.from_documents(split_docs, OpenAIEmbeddings())
        llm = ChatOpenAI(model="gpt-4")
        chain_type_kwargs = {"prompt": qa_prompt}
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever.as_retriever(), chain_type_kwargs=chain_type_kwargs)
        answer = qa.run(query)
        #chain = load_qa_chain(OpenAI(temperature=0), chain_type="refine")
        return answer
    except Exception as e:
        st.error(f"An error occurred during processing the query: {str(e)}")

#def cacheBust():
    #st.cache_data.clear()



def main():
    st.title("Loans Intel Earnings Call Summarizer")
    st.title("AI Document Summarizer")
    st.subheader("Welcome to the Document Summarizer Application!")
    st.write("To get started, please upload a Word (.docx) file. The application will process the file and provide a summarized version of the document contents. After the summary is loaded you can ask follow up questions.")
    #st.button(label='Clear cache',on_click=cacheBust)
    uploaded_file = st.file_uploader("Choose a file", type="docx")
    
    if uploaded_file is not None:
        with st.spinner('Processing...This may take a few minutes'):
            try:
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_file.read())
                tfile.close()

                loader = Docx2txtLoader(tfile.name)
                pages = loader.load_and_split()
               
                output,original_text = process_file(pages)

                st.subheader('Your summarized document:')
                st.code(output, language='')
                with st.expander:
                    st.code(original_text)
                feedback_option = "faces"
                score_mappings = {"faces": {"😀": 1, "🙂": 0.75, "😐": 0.5, "🙁": 0.25, "😞": 0},}

                if st.session_state.get("run_id"):
                    run_id = st.session_state.run_id
                    feedback = streamlit_feedback(
                        feedback_type=feedback_option,
                        optional_text_label="[Optional] Please provide an explanation",
                        key=f"feedback_{run_id}",
                    )
                
                scores = score_mappings[feedback_option]

                if feedback:
                    score = scores.get(feedback["score"])
                    if score is not None:
                        # Formulate feedback type string incorporating the feedback option and score value
                        feedback_type_str = f"{feedback_option} {feedback['score']}"

                        # Record the feedback with the formulated feedback type string and optional comment
                        feedback_record = client.create_feedback(
                            run_id,
                            feedback_type_str,
                            score=score,
                            comment=feedback.get("text"),
                        )
                        st.session_state.feedback = {
                            "feedback_id": str(feedback_record.id),
                            "score": score,
                        }
                    else:
                        st.warning("Invalid feedback score.")

                # Add a section for follow-up questions
                st.subheader('Ask a follow-up question:')
                query = st.text_input("Please enter your question here")
                submit_button = st.button("Submit Question")

                if submit_button:
                    def process_question():
                        with st.spinner('Processing your question...this may take a few minutes'):
                            try:
                                answer = process_query(query, pages)
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