import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import requests

url_extract = "https://api.worqhat.com/api/ai/v2/pdf-extract"
url_summarize = "https://api.worqhat.com/api/ai/content/v2"
headers_extract = {
    "Authorization": "Bearer sk-95ac67907ba94319a3d6a6c7e3907421",
    "Content-Type": "multipart/form-data"
}
headers_summarize = {
    "x-api-key": "sk-95ac67907ba94319a3d6a6c7e3907421",
    "Authorization": "Bearer sk-95ac67907ba94319a3d6a6c7e3907421",
    "Content-Type": "application/json"
}
bot_template = "<div style='color: green;'><strong>Buddy:</strong> {{MSG}}</div>"
user_template = "<div style='color: blue;'><strong>You:</strong> {{MSG}}</div>"

def assess_answer(question, answer,dataa):
    data = {
    "question": f"The question is {question} The answer given is: {answer} Check if the answer to the question is correct and suggest improvements, also give marks out 100 to the answer",
  
    "training_data": dataa ,
    
    "randomness": 0.1
    }
    response = requests.post(url_summarize, headers=headers_summarize, json=data)
    st.write(response.json()["content"])

def generate_summary(raw_text):
    data = {
    "question": "Generate a short summary.",
  
    "training_data": raw_text,
    
    "randomness": 0.1
    }
    response = requests.post(url_summarize, headers=headers_summarize, json=data)
    return response

def generate_question(dataa):
    data = {
    "question": "Generate one question on the data. Return only that question.",
    "training_data": dataa, 
    "randomness": 0.1
    }
    response = requests.post(url_summarize, headers=headers_summarize, json=data)
    return response


def get_pdf_text(pdf_docs):
    text = ""
   
    pdf_reader= PdfReader(pdf_docs)
    for page in pdf_reader.pages:
        text += page.extract_text()

    return text

def handle_chat(history,question,data):
    data = {
    "question": question,
    "preserve_history": True,
    "training_data": data, 
    "history_object": history,
    "randomness": 0.1
    }
    response = requests.post(url_summarize, headers=headers_summarize, json=data)
    st.session_state["history"].append(question)
    st.session_state["history"].append(response.json()["content"])
    for i,message in enumerate(st.session_state["history"]):
        if i%2 == 0:
            with st.chat_message("user"):
                st.write(message)
        else: 
            with st.chat_message("user"):
                st.write(message)
    # for i in st.session_state["history"]:
    #     st.write(user_template.replace("{{MSG}}", i),unsafe_allow_html=True)
    # for i,message in enumerate(history):
    #     if i%2 == 0:
    #         st.write(user_template.replace("{{MSG}}", message.content),unsafe_allow_html=True)
    #     else:
    #         st.write(bot_template.replace("{{MSG}}", message.content),unsafe_allow_html=True)
    

    

def get_answer(question):
    return question

def main():
    load_dotenv()
    rawww_text = ""
    st.set_page_config(page_title="Study Buddy",page_icon=":books:")
    st.header("Study Buddy :books:")
    if "summary" not in st.session_state:
        st.session_state.summary=[]
    if "history" not in st.session_state:
        st.session_state.history=[]
    # ask_questions = st.button("Ask me questions")
    # if ask_questions:
    #     butt = st.empty()
    # if butt:
    #     assess_answer(question="",answer="",rawww_text="")
        # if answer:
        #     print(1)
        #     response=assess_answer(question,answer,rawww_text)   
        #     st.write(response.json()["content"])
    container = st.empty()
    with container.container():
        user_question = st.text_input("Ask a question")
        butt= st.button("Generate")
        if butt:

            handle_chat(st.session_state["history"], user_question,rawww_text)
        # generate = st.button("Chat with the bot")
        
        # if generate:
        #     inp = st.text_area("Enter")
        #     but = st.button("Generate response")
        #     if but:
        #         print(1)
        #         handle_chat(history=st.session_state["history"])
            
            

    with st.sidebar:
        st.subheader("document")
        pdf_docs = st.file_uploader("upload document and click on process",accept_multiple_files=False)
        
        if st.button("process"):
            with st.spinner("processing"):
                raw_text = get_pdf_text(pdf_docs)
                rawww_text=raw_text
                summary = generate_summary(raw_text).json()["content"]
                st.session_state.summary = summary
                st.write("Here is the summary:")
                st.write(st.session_state.summary)
    


    
        

                




if __name__ == '__main__':
    main()