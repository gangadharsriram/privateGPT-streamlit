#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import os
import argparse
import time

import streamlit as st


from constants import CHROMA_SETTINGS


load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_batch = int(os.environ.get('MODEL_N_BATCH',8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))



st.title("Content Sensitive Chatbot")
if "messages" not in st.session_state:
    st.session_state.messages=[]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
   


@st.cache_resource
def main_function():
    args1 = parse_arguments()
    # Parse the command line arguments
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name,model_kwargs = {'device': 'cuda'})
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args1.mute_stream else [StreamingStdOutCallbackHandler()]
    # Prepare the LLM
    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, max_tokens=model_n_ctx, n_batch=model_n_batch, callbacks=callbacks, verbose=False)
        case "GPT4All":
            llm = GPT4All(model=model_path, max_tokens=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=True)
        case _default:
            # raise exception if model_type is not supported
            raise Exception(f"Model type {model_type} is not supported. Please choose one of the following: LlamaCpp, GPT4All")
        
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not args1.hide_source)
    return qa
    
    # Interactive questions and answers
@st.cache_resource
def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()




args1 = parse_arguments()
res = main_function()

query= st.chat_input(("whats is up : "))

if query:       
    if query == "exit":
        exit
        
    query=query.strip()
          
    with st.chat_message("user"):
        st.markdown(query)        
    st.session_state.messages.append({"role":"user","content":query})


    # Get the answer from the chain
    start = time.time()
    res = res(query)
    answer, docs = res['result'], [] if args1.hide_source else res['source_documents']
    end = time.time()

    # Print the result
    print("\n\n> Question:")
    print(query)
    print(f"\n> Answer (took {round(end - start, 2)} s.):")
    print(answer)
    
    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role":"assistant","content":answer})      

    # Print the relevant sources used for the answer
    for document in docs:
        print("\n> " + document.metadata["source"] + ":")
        print(document.page_content)
