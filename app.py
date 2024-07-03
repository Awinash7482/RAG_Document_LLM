
from main import process_input
import streamlit as st

#simple UI for intraction with documents
st.title("Document Query with Ollama")
st.write("Enter the query which you want to ask from the documents.")

# Input fields
question = st.text_input("Question")

# Button to process input
if st.button('Query Documents'):
    with st.spinner('Processing...'):
        answer = process_input(question)
        st.text_area("Answer", value=answer, height=300, disabled=True) 
