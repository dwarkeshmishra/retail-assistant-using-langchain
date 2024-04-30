import os
import streamlit as st
import cryptography
from langchain_helper import get_few_shot_db_chain

st.title("AtliQ T Shirts: Database Q&A 👕")

question = st.text_input("Question: ")

if question:
    chain = get_few_shot_db_chain()
    response = chain.run(question)

    st.header("Answer")
    st.write(response)

# Ensure that you have the GOOGLE_API_KEY environment variable set correctly
google_api_key = os.environ.get('GOOGLE_API_KEY')
