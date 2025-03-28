import subprocess
import time
import os
import streamlit as st
import requests

# Use Render's PORT environment variable
PORT = int(os.environ.get("PORT", 10000))
FASTAPI_PORT = PORT

def start_fastapi():
    try:
        response = requests.get(f"http://0.0.0.0:{FASTAPI_PORT}/")
        if response.status_code == 200:
            print("✅ FastAPI is already running.")
            return
    except requests.exceptions.ConnectionError:
        print("⚡ Starting FastAPI...")
        subprocess.Popen(
            ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", str(FASTAPI_PORT), "--reload"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(3)

start_fastapi()

# Update API URL to use 0.0.0.0
API_URL = f"http://0.0.0.0:{FASTAPI_PORT}/predict/"

st.title("Named Entity Recognition (NER) App 🚀")
st.markdown("Enter text below, and the model will identify named entities!")

text_input = st.text_area("Enter text:", "Google was created by Larry Page and Sergey Brin in California.")

if st.button("Analyze"):
    if text_input.strip():
        with st.spinner("Fetching entities..."):
            try:
                response = requests.post(API_URL, json={"text": text_input})
                if response.status_code == 200:
                    entities = response.json()["entities"]
                    if entities:
                        st.success("Entities identified:")
                        for entity in entities:
                            st.markdown(f"**{entity['word']}** → `{entity['entity_group']}` (Score: {entity['score']:.2f})")
                    else:
                        st.warning("No entities found.")
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect to the API: {e}")
    else:
        st.warning("Please enter some text.")
