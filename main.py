import streamlit as st
import os
from dotenv import load_dotenv
from langchain.llms import OpenAI

load_dotenv()
# openai_api_key = os.getenv ("OPENAI_API_KEY")

import requests






#Function to return the response
def load_answer(question):
    url = "https://api.worqhat.com/api/ai/content/v2"

    headers = {
        "x-api-key": "sk-6cecd17a7f3f4462ac134596eab033e5",
        "Authorization": "Bearer sk-6cecd17a7f3f4462ac134596eab033e5",
        "Content-Type": "application/json"
    }

    data = {
        "question": question,
        "randomness": 0.4
    }
    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        resp=response.json()
        return resp["content"]
   


#App UI starts here
st.set_page_config(page_title="LangChain Demo", page_icon=":robot:")
st.header("LangChain Demo")

#Gets the user input
def get_text():
    input_text = st.text_input("You: ", key="input")
    return input_text


user_input=get_text()
response = load_answer(user_input)

submit = st.button('Generate')  

#If generate button is clicked
if submit:

    st.subheader("Answer:")

    st.write(response)

