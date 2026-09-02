# Tech Brief 이미지 생성기 (Streamlit)

SMK PDF 또는 이미지를 업로드하면 부산대학교 Tech Brief 스타일의 16:9 배너를 생성하는 Streamlit 앱입니다.

## AI 모델 구성

- 기술 내용/스타일 레퍼런스 분석: `gpt-5.6-luna`
- 최종 이미지 생성: `gpt-image-2`
- 기본 출력: `2048x1152`, `high` quality

기존 Gemini 호출은 제거되어 Google API 키가 필요하지 않습니다.

## 구성 파일

- `app.py`: Streamlit UI 및 Secrets 로딩
- `banner_core.py`: PDF 추출, 스타일 ZIP 분석, OpenAI 텍스트/비전 분석, direct reference 기반 이미지 생성
- `requirements.txt`: 배포 의존성
- `.streamlit/config.toml`: Streamlit 기본 설정

## Streamlit Secrets

Streamlit Community Cloud의 **Advanced settings > Secrets** 또는 로컬 `.streamlit/secrets.toml`에 아래와 같이 입력합니다.

```toml
OPENAI_API_KEY = "sk-..."
```

`secrets.toml`은 GitHub에 커밋하지 마세요.

## 로컬 실행

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

로컬에서는 `OPENAI_API_KEY` 환경변수로도 실행할 수 있습니다.

## 선택 환경변수

필요할 때만 아래 값을 덮어쓸 수 있습니다. 기본값을 그대로 사용해도 됩니다.

```text
TEXT_MODEL=gpt-5.6-luna
IMAGE_MODEL=gpt-image-2
IMAGE_SIZE=2048x1152
IMAGE_QUALITY=high
OPENAI_MAX_RETRIES=3
OPENAI_BASE_WAIT=2.0
MAX_STYLE_IMAGES=6
MAX_STYLE_ANALYSIS_IMAGES_FOR_OPENAI=4
MAX_STYLE_REFERENCE_IMAGES_FOR_GENERATION=4
DEFAULT_STYLE_ZIP_URL=<기본 스타일 ZIP 주소>
```

## 처리 흐름

1. SMK PDF라면 텍스트와 첫 페이지 이미지를 추출합니다.
2. 스타일 ZIP의 대표 이미지들을 `gpt-5.6-luna`가 분석하여 스타일 JSON을 만듭니다.
3. 같은 스타일 ZIP에서 대표 레퍼런스 이미지를 몇 장 선별해 `gpt-image-2`에 직접 전달할 준비를 합니다.
4. SMK 텍스트 + 첫 페이지 이미지를 `gpt-5.6-luna`가 분석하여 기술/키워드/장면 JSON을 만듭니다.
5. 기존 Tech Brief 프롬프트 규칙을 결합합니다.
6. `gpt-image-2`가 선별된 레퍼런스 이미지 + 최종 프롬프트를 함께 받아 16:9 배너 PNG를 생성합니다.

## 참고

- Streamlit Cloud의 로컬 디스크는 영구 저장소가 아니므로 결과 파일은 캐시/다운로드 용도입니다.
- GPT Image 모델 사용 시 OpenAI 조직 인증이 요구될 수 있습니다.

## direct reference 동작 방식

- 스타일 ZIP이 있으면 내부에서 대표 이미지 2~4장(기본값 4장)을 선별합니다.
- 이 이미지는 `gpt-image-2`의 이미지 입력으로 직접 전달됩니다.
- 따라서 기존처럼 스타일을 텍스트로만 요약하는 것이 아니라, 실제 예시 이미지를 함께 참고하여 결과를 생성합니다.
- 스타일 ZIP이 없을 때만 일반 텍스트 기반 생성으로 폴백합니다.
