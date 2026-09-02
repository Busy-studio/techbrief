import os

import streamlit as st

from banner_core import (
    IMAGE_QUALITY,
    IMAGE_SIZE,
    process_smk_streamlit,
)


st.set_page_config(page_title="Tech Brief 이미지 생성기", layout="wide")


def get_api_key() -> str:
    if "OPENAI_API_KEY" in st.secrets:
        return str(st.secrets["OPENAI_API_KEY"]).strip()
    if os.getenv("OPENAI_API_KEY"):
        return os.getenv("OPENAI_API_KEY", "").strip()
    raise RuntimeError(
        "OPENAI_API_KEY가 설정되지 않았습니다. Streamlit Cloud Secrets 또는 로컬 .streamlit/secrets.toml을 확인하세요."
    )


st.title("Tech Brief 이미지 생성기")
st.caption("SMK PDF를 업로드하면 부산대학교 Tech Brief 스타일 배너를 생성합니다.")

with st.sidebar:
    st.subheader("실행 설정")
    size_options = ["1280x720", "1536x1024", "2048x1152"]
    quality_options = ["medium", "high"]

    default_size_index = size_options.index(IMAGE_SIZE) if IMAGE_SIZE in size_options else size_options.index("2048x1152")
    default_quality_index = quality_options.index(IMAGE_QUALITY) if IMAGE_QUALITY in quality_options else quality_options.index("high")

    selected_image_size = st.selectbox(
        "이미지 크기",
        options=size_options,
        index=default_size_index,
        help="크기가 클수록 보통 더 오래 걸립니다.",
    )
    selected_image_quality = st.selectbox(
        "이미지 품질",
        options=quality_options,
        index=default_quality_index,
        help="high가 더 느릴 수 있습니다.",
    )

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader(
        "SMK PDF 업로드",
        type=["pdf", "png", "jpg", "jpeg", "webp", "bmp"],
    )
    style_zip = st.file_uploader(
        "스타일 ZIP 업로드 (선택)",
        type=["zip"],
        help="업로드하지 않으면 Yeonhee 스타일을 사용합니다.",
    )
    run = st.button("이미지 생성", type="primary", use_container_width=True)

with col2:
    st.info(
        "업로드 파일은 임시 파일로 처리되고, 결과 이미지는 다운로드할 수 있습니다."
    )

if run:
    if uploaded_file is None:
        st.warning("SMK PDF 를 먼저 업로드하세요.")
    else:
        try:
            api_key = get_api_key()
        except Exception as e:
            st.error(str(e))
            st.stop()

        with st.spinner("배너 생성 중입니다..."):
            result = process_smk_streamlit(
                uploaded_file=uploaded_file,
                style_zip_file=style_zip,
                api_key=api_key,
                image_size=selected_image_size,
                image_quality=selected_image_quality,
            )

        if not result.get("success"):
            st.error(result.get("error", "오류가 발생했습니다."))
            st.text_area("실행 로그", result.get("result_text", ""), height=360)
        else:
            st.success("생성이 완료되었습니다.")
            st.image(result["image_bytes"], caption=result["file_name"], use_container_width=True)
            st.download_button(
                label="PNG 다운로드",
                data=result["image_bytes"],
                file_name=result["file_name"],
                mime="image/png",
                use_container_width=True,
            )
            st.text_area("제목 + 요약", result["display_text"], height=180)
            st.text_area("분석 결과 + 실행 로그", result["result_text"], height=360)
            with st.expander("분석 JSON"):
                st.json(result["analysis_data"])
            with st.expander("최종 프롬프트"):
                st.code(result["final_prompt"])
