import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain   
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from transformers import BartForConditionalGeneration, BartTokenizer
import speech_recognition as sr
#import random


css = """
    <style>
        /* Your CSS styles here */
    </style>
"""

# Templates for HTML rendering
bot_template = "<div style='color: green;'><strong>Buddy:</strong> {{MSG}}</div>"
user_template = "<div style='color: blue;'><strong>You:</strong> {{MSG}}</div>"


def generate_questions(summary):
    model_name = "sshleifer/distilbart-cnn-12-6"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    text = "Ask a question on the following summary: "+summary
    inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)
    question = model.generate(inputs.input_ids, max_length=150, num_beams=4, length_penalty=2.0, early_stopping=True)
    output = tokenizer.decode(question[0], skip_special_tokens=True)
    return output

def generate_summary(text):
    model_name = "sshleifer/distilbart-cnn-12-6"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)
    summary_ids = model.generate(inputs.input_ids, max_length=150, num_beams=4, length_penalty=2.0, early_stopping=True)

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary



def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="/n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(text)
    #keywords = [word for chunk in chunks for word in chunk.split()]
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl",model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever=vectorstore.as_retriever(),
        memory=memory

    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question':user_question})
    st.session_state.chat_history = response['chat_history']

    for i,message in enumerate(st.session_state.chat_history):
        if i%2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content),unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content),unsafe_allow_html=True)






def main():
    load_dotenv()
    st.set_page_config(page_title="Study Buddy",page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    st.header("Study Buddy :books:")
    user_question = st.text_input("Ask a question")

    st.write("Ask a question using voice:")
    if st.button("Start Voice Input"):
        user_voice_question = get_voice_input()
        if user_voice_question:
            handle_userinput(user_voice_question)

    if user_question:
        handle_userinput(user_question)


    
    with st.sidebar:
        st.subheader("document")
        pdf_docs = st.file_uploader("upload document and click on process",accept_multiple_files=True)
        if st.button("process"):
            with st.spinner("processing"):
                #get pdf text
                #raw_text= get_pdf_text(pdf_docs)
                #st.write(raw_text)

                #_, keywords = get_text_chunks(raw_text)
                #chunks, keywords = get_text_chunks(raw_text)
                #text_chunks, keywords = get_text_chunks(raw_text)
                #_, keywords = get_text_chunks(raw_text)


                


                #get the text chunks
                #text_chunks, _ = get_text_chunks(raw_text)
                #st.write(text_chunks)


                #create vector store
                #vectorstore = get_vectorstore(text_chunks)
                #vectorstore = get_vectorstore(raw_text)

                

                #create conversation chain
                #st.session_state.conversation = get_conversation_chain(vectorstore)


                #summary = generate_summary(raw_text)
                #st.session_state.summary = summary
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)

                # generate summary
                summary = generate_summary(raw_text)
                st.session_state.summary = summary


    st.write("Summary:")
    if "summary" in st.session_state:
        st.write(st.session_state.summary)

    read_summary = st.checkbox("Have you read the summary?")
    if read_summary:
        st.write("Great! Let's ask some questions about the summary.")
        # Ask questions based on the summary
        summary_questions = generate_questions(st.session_state.summary)
        st.write(summary_questions)
            # Process the user's answers or save them as needed


    st.session_state.conversation

def get_voice_input():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        st.write("Listening...")
        audio = recognizer.listen(source)

    st.write("Processing voice input...")
    try:
        user_voice_question = recognizer.recognize_google(audio)
        st.write("You said:", user_voice_question)
        return user_voice_question
    except sr.UnknownValueError:
        st.write("Could not understand audio.")
        return None
    except sr.RequestError as e:
        st.write("Could not request results; {0}".format(e))
        return None
   

if __name__ == '__main__':
    main()

