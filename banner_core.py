
import streamlit as st
from google import genai

IMAGE_MODEL = "gemini-3.1-flash-image-preview"

def process_smk_streamlit(file, style):
    client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])
    response = client.models.generate_content(
        model=IMAGE_MODEL,
        contents=["Create a simple tech banner"],
    )
    image_bytes = response.candidates[0].content.parts[0].inline_data.data
    return {"image_bytes": image_bytes}
