
import streamlit as st
from banner_core import process_smk_streamlit

st.set_page_config(page_title="Tech Brief 이미지 생성기", layout="wide")
st.title("Tech Brief 이미지 생성기")

uploaded_file = st.file_uploader("SMK PDF 또는 이미지 업로드", type=["pdf","png","jpg","jpeg","webp"])
style_zip = st.file_uploader("스타일 ZIP 업로드(선택)", type=["zip"])

if st.button("이미지 생성"):
    if uploaded_file is None:
        st.warning("파일을 업로드하세요.")
    else:
        result = process_smk_streamlit(uploaded_file, style_zip)
        st.image(result["image_bytes"])
        st.download_button("다운로드", result["image_bytes"], "banner.png")
