import streamlit as st
import requests

# FastAPI endpoint
API_URL = "http://127.0.0.1:8000/predict/"

st.title("Named Entity Recognition (NER) App 🚀")
st.markdown("Enter text below, and the model will identify named entities!")

# User input text box
text_input = st.text_area("Enter text:", "Google was created by Larry Page and Sergey Brin in California.")

if st.button("Analyze"):
    if text_input.strip():
        with st.spinner("Fetching entities..."):
            # Send request to FastAPI backend
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
                st.error("Error fetching entities. Please check the API.")
    else:
        st.warning("Please enter some text.")
