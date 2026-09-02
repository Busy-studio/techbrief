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
MAX_STYLE_REFERENCE_IMAGES_FOR_GENERATION=2
DEFAULT_STYLE_ZIP_URL=<기본 스타일 ZIP 주소>
```

## 처리 흐름

1. SMK PDF라면 텍스트와 첫 페이지 이미지를 추출합니다.
2. 스타일 ZIP을 따로 올리지 않으면 앱에 포함된 `assets/default_style_ref.json`을 즉시 사용하므로 스타일 분석용 `gpt-5.6-luna` 호출은 발생하지 않습니다.
3. 기본 스타일에서는 앱에 포함된 대표 레퍼런스 2장을 `gpt-image-2`에 직접 전달합니다.
4. 사용자가 새 스타일 ZIP을 올린 경우에만 대표 이미지를 `gpt-5.6-luna`가 분석하여 새 스타일 JSON을 만들고, 동일 ZIP 해시의 결과는 로컬 캐시에 재사용합니다.
5. SMK 텍스트 + 첫 페이지 이미지는 `gpt-5.6-luna`가 분석하여 기술/키워드/장면 JSON을 만듭니다.
6. 기존 Tech Brief 프롬프트 규칙을 결합합니다.
7. `gpt-image-2`가 레퍼런스 이미지 + 최종 프롬프트를 함께 받아 16:9 배너 PNG를 생성합니다.

## 참고

- Streamlit Cloud의 로컬 디스크는 영구 저장소가 아니므로 결과 파일은 캐시/다운로드 용도입니다.
- GPT Image 모델 사용 시 OpenAI 조직 인증이 요구될 수 있습니다.

## direct reference 동작 방식

- 기본 스타일은 앱에 내장된 대표 이미지 2장을 `gpt-image-2`에 직접 전달합니다.
- 사용자가 새 스타일 ZIP을 올리면 그 ZIP에서 대표 이미지 최대 2장을 선별해 `gpt-image-2`에 직접 전달합니다.
- 이 레퍼런스 이미지는 최종 생성 호출마다 `gpt-image-2`가 직접 처리합니다.
- 같은 사용자 스타일 ZIP은 ZIP 해시 기반 스타일 분석 JSON 캐시를 우선 재사용하므로, 캐시가 남아 있으면 `gpt-5.6-luna` 스타일 분석을 다시 하지 않습니다.
- 기본 스타일에서는 스타일 분석 JSON 자체가 앱에 내장되어 있으므로 컨테이너 캐시 유무와 관계없이 스타일 Luna 호출이 발생하지 않습니다.

## 기본 스타일 사전분석

- 사용자가 제공한 Tech Brief 레퍼런스 15장을 기준으로 기본 스타일을 사전 분석하여 `assets/default_style_ref.json`에 포함했습니다.
- 기본 direct reference는 해당 레퍼런스 중 포토리얼 기술 몽타주와 부산대학교 우상단 라벨 패턴을 대표하는 이미지 2장을 앱에 내장했습니다.
- 따라서 기본 스타일 사용 시 Google Drive ZIP 다운로드/압축 해제/스타일 Luna 분석을 모두 건너뜁니다.
- 새 스타일 ZIP을 업로드한 경우에만 그 ZIP을 새로 분석하고 해당 이미지들을 direct reference로 사용합니다.
