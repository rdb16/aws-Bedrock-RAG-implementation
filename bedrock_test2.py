from langchain.llms.bedrock import Bedrock
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import boto3
import streamlit as st

# bedrock client
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)

# ID du modèle chez Bedrock
model_id = "ai21.j2-mid-v1"

# Configuration du modèle Bedrock
llm = Bedrock(
    model_id=model_id,
    client=bedrock_client,
    model_kwargs={"temperature": 0.9}
)


def my_chatbot(language, user_text):
    prompt = PromptTemplate(
        input_variables=["language", "user_text"],
        template=f"You are a chatbot. You speak in {language}.\n\n{user_text}"
    )

    bedrock_chain = LLMChain(llm=llm, prompt=prompt)
    ans = bedrock_chain({'language': language, 'user_text': user_text})

    return ans


st.title("Bedrock Chatbot Demo")

language = st.sidebar.selectbox("Language", ["english", "spanish", "hindi"])
user_text = st.sidebar.text_area("What is your question?", max_chars=100)

if user_text:  # Cette condition assure que user_text n'est pas vide
    response = my_chatbot(language, user_text)
    st.write(response['text'])
