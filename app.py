import warnings
import requests
import os
import streamlit as st

warnings.filterwarnings("ignore")
from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain import PromptTemplate, LLMChain, HuggingFaceHub
from getpass import getpass

load_dotenv(find_dotenv())


# img2text
def img2text(url):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text(url)[0]["generated_text"]
    return text


# text2story
def text2story(text):
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
    headers = {"Authorization": "Bearer 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"}
    payload = {
        "inputs": f"You can generate a short story based on a simple narrative, the story should be no more than 300 "
                  f"words; The context is: {text}"
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()[0]["generated_text"].split("The context is: ")[1]


# story2speech
def text2speech(message):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": "Bearer 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"}
    payload = {
        "inputs": f"{message}"
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    with open('audio.mp3', 'wb') as file:
        file.write(response.content)


def main():
    st.set_page_config(page_title="Image to Audio Story", page_icon="🤖")

    st.header("Turn image into an audio story")
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        st.image(uploaded_file, caption="Uploaded Image.",
                 use_column_width=True)
        scenario = img2text(uploaded_file.name)
        story = text2story(scenario)
        text2speech(story)

        with st.expander("scenario"):
            st.write(scenario)
        with st.expander("story"):
            st.write(story)
        st.audio("audio.mp3")


if __name__ == '__main__':
    main()
