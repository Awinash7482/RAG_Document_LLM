from langchain_community.llms.ollama import Ollama
from langchain.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import streamlit as st

def process_input(question):
    llm=Ollama(model="llama3")
    print(llm.invoke("what is an apple")) # just for checking that our llama3 is giving answer or not

    #load the pdf files
    directory='/content/data'
    loader=DirectoryLoader(directory)
    documents=loader.load()
    len(documents)

    #splits documents in to chunks
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=300)
    docs=text_splitter.split_documents(documents)
    print(len(docs))
    for i in range(3):
        print(docs[i].page_content)

    #store all chunks in to vectordatabase and create embeddings
    vectorstore = Chroma.from_documents(
        documents=docs,
        collection_name="rag-chroma",
        embedding=embeddings.ollama.OllamaEmbeddings(model='nomic-embed-text'),
    )
    retriever = vectorstore.as_retriever()

    # Define multiple prompt templates
    context_prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:
    {context}
    Question: {question}
    """
    )

    comparison_prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based on comparison between the documents:
    {context}
    Question: {question}
    """
    )
    
    # Create RAG chains for different prompts
    context_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | context_prompt_template
    | llm
    | StrOutputParser()
    )

    comparison_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | comparison_prompt_template
    | llm
    | StrOutputParser()
    )
    
    
    #asking some basic question from documents
    context_chain.invoke("what is the risk factor associated with Uber?")
    context_chain.invoke("what is the total revenue for Google search?")
    comparison_chain.invoke("what is the differnces in the buisness of Tesla and Uber?")
    context_chain.invoke("what is the total revenue for Uber?")

    if "compare" in question.lower() or "COMPARE" in question.upper():
        # Use the comparison prompt
        result = comparison_chain.invoke({"context": retriever, "question": question})
    else:
        # Use the context-based prompt
        result = context_chain.invoke({"context": retriever, "question": question})
    return result

  

#simple UI for intraction with documents
st.title("Document Query with Ollama")
st.write("Enter URLs (one per line) and a question to query the documents.")

# Input fields
urls = st.text_area("Enter URLs separated by new lines", height=150)
question = st.text_input("Question")

# Button to process input
if st.button('Query Documents'):
    with st.spinner('Processing...'):
        answer = process_input(question)
        st.text_area("Answer", value=answer, height=300, disabled=True) 