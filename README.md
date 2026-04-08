# Tech Brief 이미지 생성기 (Streamlit)

SMK PDF 또는 이미지를 업로드하면 부산대학교 Tech Brief 스타일 배너를 생성하는 Streamlit 앱입니다.

## 구성 파일

- `app.py`: Streamlit UI
- `banner_core.py`: PDF 추출, 스타일 ZIP 분석, Gemini 호출, 이미지 생성 로직
- `requirements.txt`: 배포용 의존성
- `.streamlit/config.toml`: Streamlit 기본 설정

## 로컬 실행

```bash
python -m venv .venv
source .venv/bin/activate  # Windows는 .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## 로컬 시크릿 설정

프로젝트 루트에 `.streamlit/secrets.toml` 파일을 만들고 아래처럼 입력합니다.

```toml
GOOGLE_API_KEY = "YOUR_NEW_GOOGLE_API_KEY"
```

중요: `secrets.toml`은 커밋하지 마세요.

## Streamlit Community Cloud 배포

1. 이 폴더를 GitHub 저장소에 업로드
2. Streamlit Community Cloud에서 저장소 연결
3. Entry point를 `app.py`로 지정
4. Advanced settings > Secrets에 아래 내용 입력

```toml
GOOGLE_API_KEY = "YOUR_NEW_GOOGLE_API_KEY"
```

## 주의사항

- 기존 코드에 들어 있던 API 키는 절대 GitHub에 올리면 안 됩니다.
- 이미 노출된 키가 있다면 반드시 폐기 후 새 키로 교체하세요.
- 스타일 ZIP을 업로드하지 않으면 `DEFAULT_STYLE_ZIP_URL` 환경변수/기본값을 사용합니다.
- Streamlit Cloud 환경에서는 로컬 디스크가 영구 저장소가 아니므로 결과 파일은 캐시/다운로드 용도로만 사용합니다.

## 배포 팁

- 기본 이미지 생성 모델이나 텍스트 모델이 바뀌면 `.env` 또는 호스팅 환경변수에서 `TEXT_MODEL`, `IMAGE_MODEL`, `IMAGE_SIZE`를 조정하세요.
- 빌드 오류가 나면 `requirements.txt` 버전 충돌 여부를 먼저 확인하세요.
