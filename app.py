import os

import streamlit as st

from banner_core import process_smk_streamlit


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

    PRESETS = {
        "빠른 시안": {"size": "1280x720", "quality": "medium"},
        "표준": {"size": "1536x864", "quality": "medium"},
        "최종 고품질": {"size": "2048x1152", "quality": "high"},
    }
    size_options = ["1280x720", "1536x864", "2048x1152"]
    quality_options = ["medium", "high"]

    if "render_preset" not in st.session_state:
        st.session_state.render_preset = "빠른 시안"
    if "selected_image_size" not in st.session_state:
        st.session_state.selected_image_size = PRESETS[st.session_state.render_preset]["size"]
    if "selected_image_quality" not in st.session_state:
        st.session_state.selected_image_quality = PRESETS[st.session_state.render_preset]["quality"]

    def apply_render_preset() -> None:
        preset = PRESETS[st.session_state.render_preset]
        st.session_state.selected_image_size = preset["size"]
        st.session_state.selected_image_quality = preset["quality"]

    st.selectbox(
        "생성 프리셋",
        options=list(PRESETS.keys()),
        key="render_preset",
        on_change=apply_render_preset,
    )
    st.caption("프리셋을 불러온 뒤 이미지 크기와 품질은 아래에서 개별 조정할 수 있습니다.")
    selected_image_size = st.selectbox(
        "이미지 크기",
        options=size_options,
        key="selected_image_size",
        help="모든 선택지는 16:9 비율입니다.",
    )
    selected_image_quality = st.selectbox(
        "이미지 품질",
        options=quality_options,
        key="selected_image_quality",
        help="medium은 시안/표준용, high는 최종본용입니다.",
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
