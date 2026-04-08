import hashlib
import json
import mimetypes
import os
import random
import re
import shutil
import tempfile
import time
import traceback
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from google import genai
from google.genai import types

# =========================================================
# 환경설정
# =========================================================
API_KEY_ENV_NAME = "GOOGLE_API_KEY"
TEXT_MODEL = os.getenv("TEXT_MODEL", "gemini-2.5-flash")
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gemini-2.5-flash-image")

DEFAULT_STYLE_ZIP_URL = os.getenv(
    "DEFAULT_STYLE_ZIP_URL",
    "https://drive.google.com/uc?export=download&id=1tOFtgXh1M08dHfDVJB_antm8peF61RTj",
)

IMAGE_SIZE = os.getenv("IMAGE_SIZE", "1K")
GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "3"))
GEMINI_BASE_WAIT = float(os.getenv("GEMINI_BASE_WAIT", "2.0"))
MAX_STYLE_IMAGES = int(os.getenv("MAX_STYLE_IMAGES", "6"))
MAX_STYLE_ANALYSIS_IMAGES_FOR_GEMINI = int(
    os.getenv("MAX_STYLE_ANALYSIS_IMAGES_FOR_GEMINI", "4")
)

APP_DIR = Path(__file__).resolve().parent
LOCAL_CACHE_DIR = APP_DIR / ".cache"
STYLE_CACHE_DIR = LOCAL_CACHE_DIR / "style_cache"
OUTPUT_CACHE_DIR = LOCAL_CACHE_DIR / "outputs"


# =========================================================
# 유틸
# =========================================================
def log_step(logs: List[str], message: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    logs.append(f"[{ts}] {message}")


def ensure_dir(path: Path | str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_filename(name: str) -> str:
    name = re.sub(r"[^\w\-.가-힣 ]+", "_", str(name))
    return name.strip().replace(" ", "_") or "output"


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_image_file(path: str) -> bool:
    return Path(path).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def mime_from_path(path: str) -> str:
    return mimetypes.guess_type(path)[0] or "image/png"


def get_client(api_key: Optional[str] = None) -> genai.Client:
    key = (api_key or os.getenv(API_KEY_ENV_NAME, "")).strip()
    if not key:
        raise ValueError(
            f"{API_KEY_ENV_NAME}가 설정되어 있지 않습니다. Streamlit secrets 또는 환경변수를 확인하세요."
        )
    return genai.Client(api_key=key)


def normalize_google_drive_url(url: str) -> str:
    if not url:
        return url

    url = url.strip()
    if "drive.google.com/uc?export=download&id=" in url:
        return url

    m = re.search(r"/file/d/([a-zA-Z0-9_-]+)", url)
    if m:
        return f"https://drive.google.com/uc?export=download&id={m.group(1)}"

    m = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
    if m:
        return f"https://drive.google.com/uc?export=download&id={m.group(1)}"

    return url


def clean_text(value: Any) -> str:
    s = str(value or "").strip()
    return re.sub(r"\s+", " ", s)


def normalize_korean_label_text(value: str) -> str:
    s = clean_text(value).replace("\n", " ").strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"(교수)\s+\1$", r"\1", s)
    s = re.sub(r"(학과|학부)\s+\1$", r"\1", s)
    return s


def dedupe_preserve_order(items: List[Any], max_items: Optional[int] = None) -> List[str]:
    seen = set()
    results: List[str] = []
    for item in items or []:
        s = clean_text(item)
        if not s:
            continue
        key = re.sub(r"\s+", "", s).lower()
        if key in seen:
            continue
        seen.add(key)
        results.append(s)
        if max_items is not None and len(results) >= max_items:
            break
    return results


def normalize_keywords(keywords: List[Any], max_items: int = 3) -> List[str]:
    cleaned: List[str] = []
    for k in keywords or []:
        s = clean_text(k)
        if not s:
            continue
        for part in re.split(r"[/,|]+", s):
            p = clean_text(part)
            if not p:
                continue
            if len(p) > 12:
                continue
            cleaned.append(p)
    return dedupe_preserve_order(cleaned, max_items=max_items)


def normalize_professor_text(dept: str, professor: str) -> str:
    dept = normalize_korean_label_text(dept)
    professor = normalize_korean_label_text(professor)
    professor = re.sub(r"\s*교수\s*교수$", " 교수", professor).strip()
    if professor and not professor.endswith("교수"):
        professor = f"{professor} 교수"

    if dept and professor:
        return f"{dept} {professor}"
    if dept:
        return dept
    if professor:
        return professor
    return ""


def postprocess_analysis_data(data: Dict[str, Any]) -> Dict[str, Any]:
    data = dict(data or {})
    data["technology_name"] = clean_text(data.get("technology_name", ""))
    data["banner_title"] = clean_text(data.get("banner_title", ""))
    data["banner_summary"] = clean_text(data.get("banner_summary", ""))
    data["department"] = normalize_korean_label_text(data.get("department", ""))
    data["professor"] = normalize_korean_label_text(data.get("professor", ""))
    data["field_group"] = clean_text(data.get("field_group", "기타")) or "기타"
    data["field_type"] = clean_text(data.get("field_type", ""))
    data["palette"] = clean_text(data.get("palette", "차분한 블루/그레이 계열"))
    data["mood"] = clean_text(data.get("mood", "깔끔하고 현대적인 기술홍보 배너"))
    data["keywords"] = normalize_keywords(data.get("keywords", []), max_items=3)
    data["application_scene_left"] = dedupe_preserve_order(
        data.get("application_scene_left", []), max_items=4
    )
    data["core_object_center"] = dedupe_preserve_order(
        data.get("core_object_center", []), max_items=3
    )
    data["symbolic_scene_right"] = dedupe_preserve_order(
        data.get("symbolic_scene_right", []), max_items=4
    )
    data["forbidden_elements"] = dedupe_preserve_order(
        data.get("forbidden_elements", []), max_items=7
    )

    if not data["keywords"]:
        fallback: List[str] = []
        if data["field_type"]:
            fallback.append(data["field_type"])
        if data["technology_name"]:
            fallback.append(data["technology_name"])
        data["keywords"] = normalize_keywords(fallback, max_items=3)

    return data


def call_gemini_with_retry(
    api_func,
    logs: Optional[List[str]] = None,
    step_name: str = "Gemini API 호출",
    max_retries: int = GEMINI_MAX_RETRIES,
    base_wait: float = GEMINI_BASE_WAIT,
):
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            if logs is not None:
                log_step(logs, f"{step_name} 시도 {attempt}/{max_retries}")
            return api_func()
        except Exception as e:  # pragma: no cover
            last_error = e
            msg = str(e)
            transient_error = any(
                t in msg
                for t in ["503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500", "502", "504", "INTERNAL"]
            )
            if logs is not None:
                log_step(logs, f"{step_name} 실패 ({attempt}/{max_retries}): {type(e).__name__}: {e}")
            if (not transient_error) or (attempt == max_retries):
                if logs is not None:
                    log_step(logs, f"{step_name} 재시도 종료 - 최종 실패")
                raise
            wait_sec = base_wait * (2 ** (attempt - 1)) + random.uniform(0.0, 0.8)
            if logs is not None:
                log_step(logs, f"{step_name} {wait_sec:.1f}초 후 재시도")
            time.sleep(wait_sec)
    raise last_error


# =========================================================
# 입력 처리
# =========================================================
def extract_pdf_text_and_first_page_png(pdf_path: str, dpi: int = 180) -> Tuple[str, bytes]:
    doc = fitz.open(pdf_path)
    try:
        text_parts = []
        for page in doc:
            txt = page.get_text("text") or ""
            if txt.strip():
                text_parts.append(txt)
        full_text = "\n".join(text_parts).strip()

        page0 = doc.load_page(0)
        zoom = dpi / 72
        pix = page0.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        return full_text, pix.tobytes("png")
    finally:
        doc.close()


def load_image_bytes(image_path: str) -> Tuple[bytes, str]:
    mime_type = mimetypes.guess_type(image_path)[0] or "image/png"
    with open(image_path, "rb") as f:
        return f.read(), mime_type


def prepare_input(path: str) -> Tuple[str, bytes, str, str]:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        text, png_bytes = extract_pdf_text_and_first_page_png(path)
        return text, png_bytes, "image/png", Path(path).name
    if ext in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        img_bytes, mime_type = load_image_bytes(path)
        return "", img_bytes, mime_type, Path(path).name
    raise ValueError("지원하지 않는 파일 형식입니다. PDF 또는 이미지 파일을 업로드하세요.")


# =========================================================
# 스타일 ZIP 처리
# =========================================================
def download_file(url: str, out_path: str) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(out_path, "wb") as f:  # nosec B310
        shutil.copyfileobj(resp, f)


def get_style_zip_path(style_zip_path: Optional[str], logs: List[str]) -> Optional[str]:
    if style_zip_path:
        log_step(logs, f"업로드 스타일 ZIP 사용: {Path(style_zip_path).name}")
        return style_zip_path

    if not DEFAULT_STYLE_ZIP_URL:
        log_step(logs, "기본 스타일 ZIP URL이 없어 스타일 레퍼런스 없이 진행")
        return None

    ensure_dir(STYLE_CACHE_DIR)
    direct_url = normalize_google_drive_url(DEFAULT_STYLE_ZIP_URL)
    url_hash = hashlib.sha256(direct_url.encode("utf-8")).hexdigest()[:16]
    out_path = STYLE_CACHE_DIR / f"default_style_ref_{url_hash}.zip"

    if not out_path.exists():
        log_step(logs, "기본 스타일 ZIP 다운로드 시작")
        download_file(direct_url, str(out_path))
        log_step(logs, f"기본 스타일 ZIP 다운로드 완료: {out_path}")
    else:
        log_step(logs, f"기본 스타일 ZIP 캐시 재사용: {out_path}")
    return str(out_path)


def extract_zip_to_temp(zip_path: str, logs: List[str]) -> str:
    if not zipfile.is_zipfile(zip_path):
        raise ValueError("스타일 ZIP 파일이 올바른 zip 형식이 아닙니다.")
    temp_dir = tempfile.mkdtemp(prefix="style_zip_")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(temp_dir)
    log_step(logs, f"스타일 ZIP 압축 해제 완료: {temp_dir}")
    return temp_dir


def collect_image_paths(root_dir: str, max_count: int = MAX_STYLE_IMAGES) -> List[str]:
    candidates = sorted(str(path) for path in Path(root_dir).rglob("*") if path.is_file() and is_image_file(str(path)))
    if len(candidates) <= max_count:
        return candidates
    step = len(candidates) / max_count
    sampled = []
    for i in range(max_count):
        idx = min(int(round(i * step)), len(candidates) - 1)
        sampled.append(candidates[idx])
    return list(dict.fromkeys(sampled))


def pick_style_images_for_gemini(image_paths: List[str], max_count: int = MAX_STYLE_ANALYSIS_IMAGES_FOR_GEMINI) -> List[str]:
    if len(image_paths) <= max_count:
        return image_paths
    step = len(image_paths) / max_count
    sampled = []
    for i in range(max_count):
        idx = min(int(round(i * step)), len(image_paths) - 1)
        sampled.append(image_paths[idx])
    return list(dict.fromkeys(sampled))


STYLE_ANALYSIS_PROMPT = """
너는 이미지 스타일 분석 전문가다.

입력된 여러 장의 예시 이미지는 부산대학교 Tech Brief 계열 기술소개 배너 예시다.
이 이미지들의 공통 패턴을 분석해서,
향후 새로운 기술 배너를 생성할 때 그대로 모방할 수 있도록
매우 구체적인 JSON 스타일 규격으로 정리하라.

중요:
- \"스타일 참고\"가 아니라 \"공통 템플릿 패턴 추출\"이다.
- 특정 기술 내용은 무시하고, 오직 시각 스타일과 템플릿 규칙만 추출하라.
- 학교명/학과명/교수명 라벨 시스템의 공통 시각 규칙을 자세히 추출하라.
- 학교명은 작은 라운드 박스 안에만 들어가고, 학과/교수명은 박스 아래 바깥에 배치되는지 우선적으로 관찰하라.
- 학과명과 교수명(또는 교수명 포함 전체 구문)의 색상 분리 패턴이 보이면 반드시 반영하라.
- JSON 외 텍스트 절대 금지.

반환 JSON 스키마:
{
  \"style_name\": \"한 줄 설명\",
  \"overall_layout\": {
    \"canvas_ratio\": \"16:9\",
    \"composition_pattern\": \"좌/중앙/우 구도 설명\",
    \"scene_balance\": \"좌측/중앙/우측 비중\",
    \"focal_pattern\": \"시선 집중 패턴\",
    \"thumbnail_behavior\": \"작은 크기에서의 가독성 특징\"
  },
  \"blending_style\": {
    \"transition_type\": \"자연스러운 블렌딩/콜라주/몽타주 설명\",
    \"edge_hardness\": \"하드엣지 여부\",
    \"depth_feel\": \"레이어 깊이감\",
    \"lighting_style\": \"광원/명암 스타일\"
  },
  \"color_style\": {
    \"main_palette\": \"대표 색감\",
    \"contrast_level\": \"대비 수준\",
    \"tone_keywords\": [\"색감 키워드1\", \"키워드2\"]
  },
  \"text_policy\": {
    \"text_amount\": \"적음/중간/많음\",
    \"title_usage\": \"긴 제목 사용 여부\",
    \"summary_usage\": \"요약문 사용 여부\",
    \"keyword_usage\": \"키워드 수와 역할\",
    \"keyword_readability\": \"썸네일 기준 가독성 특징\"
  },
  \"keyword_typography\": {
    \"base_style\": \"키워드 타이포그래피 공통 특징\",
    \"weight\": \"굵기 경향\",
    \"integration\": \"배경과 융합 방식\",
    \"recommended_variation_by_domain\": {
      \"디지털·AI\": \"권장 스타일\",
      \"제조·공정\": \"권장 스타일\",
      \"소재·화학\": \"권장 스타일\",
      \"에너지·환경\": \"권장 스타일\",
      \"바이오·의료\": \"권장 스타일\",
      \"전자·반도체\": \"권장 스타일\",
      \"모빌리티·로봇\": \"권장 스타일\",
      \"건설·도시\": \"권장 스타일\",
      \"농생명·식품\": \"권장 스타일\",
      \"해양·조선\": \"권장 스타일\",
      \"기타\": \"권장 스타일\"
    }
  },
  \"university_label_component\": {
    \"exists\": true,
    \"shape\": \"학교명만 들어가는 작은 rounded rectangle\",
    \"background_style\": \"solid light background\",
    \"padding_feel\": \"compact padding\",
    \"corner_radius_feel\": \"soft rounded corners\",
    \"title_text\": \"부산대학교\",
    \"title_style\": \"학교명 글꼴/굵기/색상 경향\",
    \"subtitle_layout\": \"박스 아래 바깥에 학과명 + 교수명 배치\",
    \"subtitle_department_style\": \"학과명 크기/굵기/색상 경향\",
    \"subtitle_professor_style\": \"교수명(또는 교수명 포함 전체 구문) 크기/굵기/색상 경향\",
    \"subtitle_color_difference\": \"학과명과 교수명 색 분리 여부와 방식\",
    \"spacing\": \"학교명 박스와 서브텍스트 간격\",
    \"alignment\": \"정렬 방식\",
    \"placement_behavior\": \"이미지 상부 영역에 자연스럽게 배치되는 방식\",
    \"consistency_note\": \"얼마나 동일하게 재현해야 하는지\"
  },
  \"do_not_do\": [\"금지사항1\", \"금지사항2\", \"금지사항3\"]
}
""".strip()


ANALYSIS_PROMPT = """
너는 부산대학교 기술소개자료(SMK)를 읽고,
'기술소개 배너 이미지'를 만들기 위한 시각 기획 데이터를 추출하는 전문가다.

중요:
이 배너는 설명형 인포그래픽이 아니라,
부산대학교 Tech Brief 스타일의 가로형 비주얼 배너여야 한다.

목표:
- 업로드된 PDF 또는 기술소개 이미지의 내용을 읽고
- 기술의 핵심 개념과 대표 응용 분야를 파악하고
- 실제 기술과 관련된 이미지 콜라주용 장면 요소를 도출하고
- 최종적으로 배너 생성용 JSON만 반환한다.

반드시 지킬 규칙:
1) 기술과 직접 관련된 요소만 사용한다.
2) 기술과 무관한 사람, 관광지, 감성 풍경, 야경, 도시 거리, 임의의 소품은 넣지 않는다.
3) 설명형 다이어그램, 복잡한 인포그래픽, 박스형 정보구성은 지양한다.
4) 최종 이미지는 텍스트보다 이미지 중심의 배너여야 한다.
5) 키워드는 반드시 서로 다른 짧은 단어 1~3개만 도출한다.
6) 같은 키워드를 반복하거나 유사어를 중복해서 쓰지 않는다.
7) 긴 제목, 긴 문장, 요약문은 배너 본 이미지에 넣지 않는 방향으로 생각한다.
8) field_group은 아래 대분류 중 가장 가까운 것 1개를 선택한다:
   제조·공정 | 소재·화학 | 에너지·환경 | 바이오·의료 | 디지털·AI |
   전자·반도체 | 모빌리티·로봇 | 건설·도시 | 농생명·식품 | 해양·조선 | 기타
9) field_type은 이 기술의 세부 분야를 자연어로 구체적으로 작성한다.
10) 교수명/학과명은 문서에서 자동 추출하되, 없으면 빈 값으로 둔다.
11) 한국어로 작성한다.
12) banner_title은 배너 아래 설명용으로 보여줄 짧은 제목 1줄로 작성한다.
13) banner_summary는 기술 핵심 원리와 활용 가능성을 2~4문장으로 자연스럽게 요약한다.
14) scene/object 리스트 내에서도 같은 의미를 반복하지 않는다.

반환 JSON 스키마:
{
  \"technology_name\": \"기술 전체 명칭\",
  \"banner_title\": \"배너 아래 텍스트로 보여줄 짧은 제목\",
  \"banner_summary\": \"배너 아래 텍스트로 보여줄 2~4문장 요약문\",
  \"department\": \"학과명 또는 학부명\",
  \"professor\": \"교수명\",
  \"field_group\": \"제조·공정|소재·화학|에너지·환경|바이오·의료|디지털·AI|전자·반도체|모빌리티·로봇|건설·도시|농생명·식품|해양·조선|기타\",
  \"field_type\": \"세부 기술분야를 자연어로 구체적으로 작성\",
  \"keywords\": [\"서로 다른 짧은 키워드1\", \"서로 다른 짧은 키워드2\", \"서로 다른 짧은 키워드3\"],
  \"application_scene_left\": [\"좌측에 들어갈 실제 응용/산업 장면 요소 2~4개\"],
  \"core_object_center\": [\"중앙에 들어갈 핵심 기술 오브젝트/구조/소재 요소 1~3개\"],
  \"symbolic_scene_right\": [\"우측에 들어갈 미래지향적 응용/상징 장면 요소 2~4개\"],
  \"forbidden_elements\": [\"넣지 말아야 할 요소 2~5개\"],
  \"palette\": \"권장 색감\",
  \"mood\": \"권장 분위기\"
}

출력은 JSON만 반환하라.
JSON 외 설명문 절대 금지.
""".strip()


def analyze_style_reference(
    client: genai.Client,
    style_image_paths: List[str],
    cache_key: Optional[str],
    logs: List[str],
) -> Dict[str, Any]:
    ensure_dir(STYLE_CACHE_DIR)

    if cache_key:
        cache_path = STYLE_CACHE_DIR / f"style_analysis_{cache_key}.json"
        if cache_path.exists():
            log_step(logs, f"스타일 분석 캐시 재사용: {cache_path}")
            return json.loads(cache_path.read_text(encoding="utf-8"))

    style_image_paths = pick_style_images_for_gemini(style_image_paths, MAX_STYLE_ANALYSIS_IMAGES_FOR_GEMINI)
    if not style_image_paths:
        raise ValueError("스타일 분석용 이미지가 없습니다.")

    parts: List[Any] = [STYLE_ANALYSIS_PROMPT]
    for p in style_image_paths:
        with open(p, "rb") as f:
            parts.append(types.Part.from_bytes(data=f.read(), mime_type=mime_from_path(p)))

    def _call():
        return client.models.generate_content(
            model=TEXT_MODEL,
            contents=parts,
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
            ),
        )

    response = call_gemini_with_retry(_call, logs=logs, step_name="스타일 레퍼런스 분석")
    style_ref = json.loads(response.text)

    if cache_key:
        cache_path = STYLE_CACHE_DIR / f"style_analysis_{cache_key}.json"
        cache_path.write_text(json.dumps(style_ref, ensure_ascii=False, indent=2), encoding="utf-8")
        log_step(logs, f"스타일 분석 캐시 저장: {cache_path}")

    return style_ref


def analyze_with_gemini(
    client: genai.Client,
    text_excerpt: str,
    image_bytes: bytes,
    image_mime_type: str,
    logs: Optional[List[str]] = None,
) -> Dict[str, Any]:
    excerpt = text_excerpt[:18000] if text_excerpt else ""

    def _call():
        return client.models.generate_content(
            model=TEXT_MODEL,
            contents=[
                ANALYSIS_PROMPT,
                "추출 텍스트:\n" + excerpt,
                types.Part.from_bytes(data=image_bytes, mime_type=image_mime_type),
            ],
            config=types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json",
            ),
        )

    response = call_gemini_with_retry(_call, logs=logs, step_name="기술 내용 분석")
    return json.loads(response.text)


def get_field_typography_instruction(field_group: str, style_ref: Dict[str, Any]) -> str:
    base = (((style_ref or {}).get("keyword_typography") or {}).get("recommended_variation_by_domain") or {})
    if field_group in base:
        return str(base[field_group]).strip()
    fallback_map = {
        "디지털·AI": "futuristic, thin, glowing, UI-like Korean typography",
        "제조·공정": "bold, solid, slightly condensed Korean sans-serif typography",
        "소재·화학": "clean modern Korean typography with refined scientific feel",
        "에너지·환경": "balanced modern Korean typography with subtle eco-tech refinement",
        "바이오·의료": "clean, soft, minimal Korean typography",
        "전자·반도체": "precise, sharp, high-tech Korean typography with cool tone",
        "모빌리티·로봇": "bold modern Korean typography with dynamic technological feel",
        "건설·도시": "stable, structured, clean sans-serif Korean typography",
        "농생명·식품": "clean scientific Korean typography with natural and fresh tone",
        "해양·조선": "solid industrial Korean typography with cool marine-tech mood",
        "기타": "clean modern Korean sans-serif typography",
    }
    return fallback_map.get(field_group, fallback_map["기타"])


def join_nonempty(items: List[str], fallback: str) -> str:
    vals = [str(x).strip() for x in items if str(x).strip()]
    return ", ".join(vals) if vals else fallback


def make_display_title(data: Dict[str, Any]) -> str:
    return clean_text(data.get("banner_title", "")) or clean_text(data.get("technology_name", "")) or "기술 홍보 배너"


def make_display_summary(data: Dict[str, Any]) -> str:
    summary = clean_text(data.get("banner_summary", ""))
    if summary:
        return summary
    field_type = clean_text(data.get("field_type", ""))
    keywords = [k.strip() for k in data.get("keywords", []) if str(k).strip()]
    parts = []
    if field_type:
        parts.append(f"본 기술은 {field_type} 분야에 해당합니다.")
    if keywords:
        parts.append(f"핵심 키워드는 {', '.join(keywords[:3])}입니다.")
    parts.append("SMK 내용을 바탕으로 기술 핵심 개념과 활용 가능성을 반영하여 배너를 생성했습니다.")
    return " ".join(parts)


def build_full_banner_prompt(data: Dict[str, Any], style_ref: Dict[str, Any]) -> str:
    dept = clean_text(data.get("department", ""))
    professor = clean_text(data.get("professor", ""))
    field_group = clean_text(data.get("field_group", "기타")) or "기타"
    field_type = clean_text(data.get("field_type", ""))

    keywords = normalize_keywords(data.get("keywords", []), max_items=3)
    left_scene = dedupe_preserve_order(data.get("application_scene_left", []), max_items=4)
    center_scene = dedupe_preserve_order(data.get("core_object_center", []), max_items=3)
    right_scene = dedupe_preserve_order(data.get("symbolic_scene_right", []), max_items=4)
    forbidden = dedupe_preserve_order(data.get("forbidden_elements", []), max_items=7)

    palette = clean_text(data.get("palette", "차분한 블루/그레이 계열"))
    mood = clean_text(data.get("mood", "깔끔하고 현대적인 기술홍보 배너"))
    professor_text = normalize_professor_text(dept, professor)

    overall_layout = style_ref.get("overall_layout", {}) if style_ref else {}
    blending_style = style_ref.get("blending_style", {}) if style_ref else {}
    color_style = style_ref.get("color_style", {}) if style_ref else {}
    text_policy = style_ref.get("text_policy", {}) if style_ref else {}
    keyword_typography = style_ref.get("keyword_typography", {}) if style_ref else {}
    label_component = style_ref.get("university_label_component", {}) if style_ref else {}
    style_dont = style_ref.get("do_not_do", []) if style_ref else []

    typography_instruction = get_field_typography_instruction(field_group, style_ref)

    if len(keywords) == 0:
        text_instruction = "Do not place extra keyword text in the main visual area."
    elif len(keywords) == 1:
        text_instruction = f'Place exactly one short Korean keyword only: "{keywords[0]}"'
    elif len(keywords) == 2:
        text_instruction = f'Place exactly two different short Korean keywords only: "{keywords[0]}" and "{keywords[1]}"'
    else:
        joined = " / ".join([f'"{k}"' for k in keywords])
        text_instruction = f"Place exactly these 3 different short Korean keywords and no others: {joined}"

    style_constraint_block = f"""
[CRITICAL TEMPLATE REPLICATION RULE - MUST FOLLOW STRICTLY]
This is NOT a loose style reference task.
This is a TEMPLATE PATTERN REPLICATION task.
You MUST replicate the same banner design family used in the reference images.

[University Label System - Must Match Reference Pattern]
- Follow the same structure used in the reference images
- Integrate the label system naturally into the composition
- The result must feel like the same original template family

[Top Overlay Box Rule]
- The university box should be positioned as high as possible near the top of the image
- It should sit very close to the upper canvas boundary, but should NOT overlap, cross, or be cropped by the top edge
- Keep a very small top margin only
- The box must remain fully visible inside the canvas
- Do NOT create any top band, strip, ribbon, or header panel
- Do NOT create a full-width top banner
- Do NOT create a colored or white bar across the image
- This rule applies only to the small university rounded box

[Structure Rules]
- Only the text \"부산대학교\" should be inside the rounded light-colored box
- Department text and professor text must NOT be inside that box
- Department text and professor text must be placed below the box, outside the box, as a separate subtitle line

[Box Shape Rules]
- Use a compact rounded rectangle
- The box should be small and tight, not oversized
- Keep horizontal padding compact
- Keep vertical padding compact
- The box width should fit the university text closely
- Do NOT make the box too long or too wide
- Do NOT create a large capsule card
- Do NOT stretch the box sideways

[University Text Rules]
- Inside box: \"부산대학교\" only
- The university name must be bold and strongest
- The university text should be clearly larger than the subtitle line below it

[Subtitle Line Rules]
- Below the box: department text + professor text
- The subtitle line must be significantly smaller than the university name
- The subtitle should look like secondary information
- The subtitle line must be lighter, thinner, and less dominant than the university text
- The subtitle line should stay on one compact horizontal line when possible
- The subtitle line should be approximately 45% to 60% of the visual height of the university text
- Do NOT place the subtitle on a white panel, box, card, or translucent rectangle
- The subtitle must appear directly over the image background as plain overlay text
- The subtitle background must remain transparent
- Do NOT clear, fade out, blur out, wash out, or simplify the background behind the subtitle
- Keep the image background continuous behind the subtitle line

[Department / Professor Color Rule]
- Department text must use a neutral dark gray or muted navy tone
- The full phrase \"OOO 교수\" must use the accent color
- Department text and professor phrase must use different colors
- The full professor phrase including \"교수\" should stay in the accent color

[Placement Rules]
- Place the university box near the upper-right area in the same manner as the reference examples
- The subtitle line should sit directly below the university box
- The subtitle line should align visually with the box above
- Keep the component compact and neat

[Consistency Rules]
- This label system must remain consistent across generations
- Do NOT reinterpret it freely
- Do NOT simplify it into plain text only
- The result must look like it came from the same original template family as the reference images

[Keyword Typography System]
- Keywords are major visual text elements, not explanatory captions
- Use only 1 to 3 short Korean keywords maximum
- Prefer only 1 or 2 keywords for cleaner composition
- Use 3 keywords only when necessary
- Never place long explanations
- Never repeat the same keyword
- Never add extra keyword text beyond the requested set
- Each keyword must appear only once in the entire image
- Never duplicate the same keyword in any position
- Do not render the same word twice, even for emphasis, shadow, echo, or layout balance
- If a keyword is placed once, it must not appear again anywhere else in the image
- Avoid duplicated Korean text caused by stylized text rendering

[Keyword Placement Rules]
- Place keywords near the main focal subject
- Do not scatter them randomly
- Avoid busy backgrounds
- Keep strong hierarchy and immediate readability

[Global Image Rules]
- Do NOT create infographic layout
- Do NOT create diagram board layout
- Do NOT create presentation slide look
- Do NOT use many labels
- Must look like a cinematic blended premium technology banner
""".strip()

    prompt = f"""
{style_constraint_block}

Create a 16:9 wide horizontal Korean university technology banner.

This image must follow the Busan National University Tech Brief-style reference pattern.
It must NOT look like a detailed infographic, poster, brochure, teaching slide, or explanatory diagram.

[Primary Goal]
- Recreate the overall visual template pattern of the reference images
- Replicate the same university label system found in the reference images
- Mimic the composition logic, spacing feel, text amount, and blended collage structure of the reference examples
- The reference style controls the template form, while the uploaded SMK content controls the subject matter

[Reference Template Rules]
- style name: {style_ref.get('style_name', '')}
- composition pattern: {overall_layout.get('composition_pattern', '')}
- scene balance: {overall_layout.get('scene_balance', '')}
- focal pattern: {overall_layout.get('focal_pattern', '')}
- thumbnail behavior: {overall_layout.get('thumbnail_behavior', '')}

[Blending Rules]
- transition type: {blending_style.get('transition_type', '')}
- edge hardness: {blending_style.get('edge_hardness', '')}
- depth feel: {blending_style.get('depth_feel', '')}
- lighting style: {blending_style.get('lighting_style', '')}

[Color Rules]
- reference main palette: {color_style.get('main_palette', '')}
- reference contrast level: {color_style.get('contrast_level', '')}
- reference tone keywords: {', '.join(color_style.get('tone_keywords', []) or [])}
- actual content palette preference: {palette}
- actual content mood preference: {mood}

[Technology Interpretation]
- field group: {field_group}
- field type: {field_type}
- use field group mainly for typography and tone selection
- use field type mainly for scene composition and subject interpretation

[Core Style]
- image-centered
- minimal text
- premium public-institution technology briefing style
- realistic photorealistic collage
- smooth blended montage rather than hard segmented boxes
- clean composition
- no clutter
- no dense captions
- no explanatory paragraphs
- must look effective even at YouTube thumbnail size

[Scene Direction - based on uploaded technology]
- Left side elements: {join_nonempty(left_scene, 'technology-related real-world application scene')}
- Center elements: {join_nonempty(center_scene, 'core technology object')}
- Right side elements: {join_nonempty(right_scene, 'future-oriented related scene')}

[Physical Realism Constraint - CRITICAL]
- All industrial equipment and machinery must be depicted in physically realistic ways
- Do NOT add unrealistic sci-fi effects such as laser beams, energy rays, glowing weapon-like outputs from machines
- Excavators, construction equipment, and industrial tools must operate in a believable real-world manner
- Do NOT make construction machines emit light beams, lasers, or energy effects
- Avoid science-fiction interpretations of real-world engineering equipment
- If visual effects are used, they must represent plausible physical processes (e.g., water flow, cutting stream, dust, sparks)
- The scene must look like a real-world engineering site, not a sci-fi battlefield

[Thumbnail Optimization]
- Design must be clearly visible at small sizes
- Emphasize one dominant central subject
- Avoid clutter and too many elements
- Strong contrast between subject and background
- Background should support the subject, not compete with it

[Text Policy From Reference]
- text amount: {text_policy.get('text_amount', '')}
- title usage: {text_policy.get('title_usage', '')}
- summary usage: {text_policy.get('summary_usage', '')}
- keyword usage: {text_policy.get('keyword_usage', '')}
- keyword readability: {text_policy.get('keyword_readability', '')}

[Keyword Typography Rules]
- reference keyword base style: {keyword_typography.get('base_style', '')}
- reference keyword weight: {keyword_typography.get('weight', '')}
- reference keyword integration: {keyword_typography.get('integration', '')}
- domain-specific typography instruction: {typography_instruction}
- text size must be large and dominant
- avoid default flat caption-style text
- text should feel integrated with lighting or depth

[University Label System - Reference Replication]
- replicate the same university label structure from the reference images
- inside the rounded light box: only \"부산대학교\"
- below the box, outside the box: department text + full professor phrase
- shape family: {label_component.get('shape', 'rounded light box')}
- background style: {label_component.get('background_style', 'solid light background')}
- padding feel: compact and tight
- corner radius feel: {label_component.get('corner_radius_feel', 'soft rounded corners')}
- title style: {label_component.get('title_style', 'bold strong university name')}
- subtitle layout: department and professor on a separate line below the box
- subtitle line size: clearly smaller than the university name
- subtitle line weight: lighter and less dominant than the university name
- department text style: neutral subtitle text
- professor phrase style: accent-colored full phrase including \"교수\"
- subtitle color difference: department and full professor phrase must differ in color
- spacing: tight clean spacing between box and subtitle
- placement behavior: near upper-right area, matching the reference examples
- place the university box very close to the top boundary, but keep it fully inside the canvas
- use only a very small top margin
- do not let the box overlap or cross the top edge
- do not create any top strip, top band, or header area
- the university box should be compact and occupy roughly 14% to 18% of the full image width
- the subtitle line should be approximately 45% to 60% of the visual height of the university text
- keep the entire school label component modest in scale, matching the reference example
- the label component must sit on top of the existing image rather than creating a separate clean background zone
- do not erase, brighten, blur, wash out, or simplify the background behind the school / department / professor text
- keep background details visible behind and around the label component
- do not create a white haze, faded patch, or empty clean area behind the label component
- the subtitle line must be plain transparent overlay text on the image
- do not generate any white or light background behind the department/professor line
- do not place the subtitle inside a box, card, panel, or label container
- Treat the school label component as a small overlay sticker on top of the finished image, not as a layout area that requires background cleanup

[University Label System - Exact Content]
- inside the rounded light box, write exactly: 부산대학교
- below the box, outside the box, write exactly: {professor_text if professor_text else ' '}
- the box must contain only the university name
- department and professor line must not be placed inside the same rounded box
- the university name must be the largest text in this label component
- the subtitle line must be noticeably smaller than the university name
- the subtitle line should be approximately 45% to 60% of the visual height of the university text
- the subtitle line should look compact and secondary
- in the subtitle line, department text must use a neutral color
- the full professor phrase including \"교수\" must use the accent color
- place the university box very close to the top boundary, but keep it fully visible inside the canvas
- do not let the box overlap or cross the top edge
- do not create a top ribbon, header strip, or wide bar
- the subtitle line must be directly overlaid on the image with no white background panel
- do not erase, fade, wash out, or blur the image background behind the school label component
- keep the original background image continuous behind the label component
- the component should mimic the reference examples as closely as possible

[Main keyword rule]
{text_instruction}
- Every keyword must be rendered exactly one time only

[Negative instructions]
- do not add unrelated people unless the technology directly requires them
- do not add tourist scenery
- do not add random city night views unless directly relevant
- do not add random icons, emoji-like graphics, fake dashboards, fake English text, fake numbers
- avoid infographic boxes, arrows, and educational poster composition
- avoid white-background infographic boards
- avoid many explanatory labels across the image
- prefer large blended photographic scenes over diagrammatic explanation
- the image should feel like a premium visual key art banner, not a teaching slide
- use only the requested 1 to 3 short Korean keyword texts
- do not repeat any keyword
- do not add extra keyword text beyond the requested set
- additional reference do-not-do rules: {', '.join(style_dont) if style_dont else 'none'}
- content-specific forbidden elements: {', '.join(forbidden) if forbidden else 'none'}
- do not put department and professor text inside the same rounded box as the university name
- do not create a long horizontally stretched right-extending label card
- do not create an oversized university box with large left and right empty padding
- do not clear, fade out, blur out, wash out, or simplify the background behind the university label component
- keep the original image background fully continuous behind the university box and subtitle line
- do not create an empty clean area for the label
- do not add a white haze, light gradient panel, foggy overlay, or blurred patch behind the school / department / professor text
- the label component must sit on top of the existing image, not replace or erase the background
- background details must remain visible around and behind the label component

The final result must look like a new technology banner generated in the same visual template family as the reference ZIP images, while reflecting the uploaded technology content.
""".strip()

    return prompt


# =========================================================
# 이미지 생성
# =========================================================
def extract_image_bytes_from_response(response) -> bytes:
    if hasattr(response, "candidates") and response.candidates:
        for cand in response.candidates:
            content = getattr(cand, "content", None)
            if not content:
                continue
            for part in getattr(content, "parts", None) or []:
                inline_data = getattr(part, "inline_data", None)
                if inline_data and getattr(inline_data, "data", None):
                    return inline_data.data
    if hasattr(response, "parts"):
        for part in response.parts:
            inline_data = getattr(part, "inline_data", None)
            if inline_data and getattr(inline_data, "data", None):
                return inline_data.data
    raise RuntimeError("응답에서 이미지 바이트를 찾지 못했습니다.")


def generate_final_banner(client: genai.Client, prompt: str, logs: Optional[List[str]] = None) -> bytes:
    def _call():
        return client.models.generate_content(
            model=IMAGE_MODEL,
            contents=[prompt],
            config=types.GenerateContentConfig(
                image_config=types.ImageConfig(
                    aspect_ratio="16:9",
                    image_size=IMAGE_SIZE,
                ),
            ),
        )

    response = call_gemini_with_retry(_call, logs=logs, step_name="최종 배너 생성")
    image_bytes = extract_image_bytes_from_response(response)
    if logs is not None:
        log_step(logs, f"최종 배너 생성 성공: {IMAGE_MODEL} / {IMAGE_SIZE}")
    return image_bytes


# =========================================================
# Streamlit/일반 실행용 헬퍼
# =========================================================
def save_uploaded_file(uploaded_file, suffix: Optional[str] = None) -> str:
    suffix = suffix or Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def process_smk_paths(
    input_path: str,
    style_zip_path: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    logs: List[str] = []
    extracted_style_dir = None

    try:
        log_step(logs, "앱 시작")
        ensure_dir(OUTPUT_CACHE_DIR)
        client = get_client(api_key=api_key)
        log_step(logs, "Gemini 클라이언트 초기화 완료")

        text, image_bytes, image_mime_type, original_name = prepare_input(input_path)
        log_step(logs, f"입력 파일 준비 완료: {original_name}")
        log_step(logs, f"추출 텍스트 길이: {len(text)}자")
        log_step(logs, f"입력 이미지 MIME: {image_mime_type}")

        resolved_style_zip = get_style_zip_path(style_zip_path, logs)

        style_ref: Dict[str, Any] = {}
        style_cache_key = None
        style_image_paths: List[str] = []

        if resolved_style_zip:
            style_cache_key = sha256_of_file(resolved_style_zip)[:24]
            extracted_style_dir = extract_zip_to_temp(resolved_style_zip, logs)
            style_image_paths = collect_image_paths(extracted_style_dir, MAX_STYLE_IMAGES)
            log_step(logs, f"스타일 이미지 수집 완료: {len(style_image_paths)}장")
            if not style_image_paths:
                raise ValueError("스타일 ZIP 안에서 이미지 파일을 찾지 못했습니다.")
            style_ref = analyze_style_reference(
                client=client,
                style_image_paths=style_image_paths,
                cache_key=style_cache_key,
                logs=logs,
            )
            log_step(logs, "스타일 레퍼런스 분석 완료")

        raw_data = analyze_with_gemini(
            client=client,
            text_excerpt=text,
            image_bytes=image_bytes,
            image_mime_type=image_mime_type,
            logs=logs,
        )
        data = postprocess_analysis_data(raw_data)
        log_step(logs, "기술 내용 구조화 분석 완료")
        log_step(logs, f"후처리 키워드: {', '.join(data.get('keywords', []))}")

        final_prompt = build_full_banner_prompt(data, style_ref)
        log_step(logs, "최종 배너 프롬프트 생성 완료")

        final_img_bytes = generate_final_banner(client, final_prompt, logs)
        log_step(logs, f"최종 배너 이미지 바이트 생성 완료: {len(final_img_bytes)} bytes")

        title_part = safe_filename(data.get("technology_name", "banner"))
        out_path = OUTPUT_CACHE_DIR / f"{title_part}.png"
        json_path = OUTPUT_CACHE_DIR / f"{title_part}_analysis.json"
        prompt_path = OUTPUT_CACHE_DIR / f"{title_part}_prompt.txt"
        style_path = OUTPUT_CACHE_DIR / f"{title_part}_style_ref.json"

        out_path.write_bytes(final_img_bytes)
        json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        prompt_path.write_text(final_prompt, encoding="utf-8")
        style_path.write_text(json.dumps(style_ref, ensure_ascii=False, indent=2), encoding="utf-8")

        log_step(logs, f"최종 PNG 저장 완료: {out_path}")
        log_step(logs, f"기술 분석 JSON 저장 완료: {json_path}")
        log_step(logs, f"프롬프트 TXT 저장 완료: {prompt_path}")
        log_step(logs, f"스타일 분석 JSON 저장 완료: {style_path}")

        display_title = make_display_title(data)
        display_summary = make_display_summary(data)
        display_text = f"{display_title}\n\n{display_summary}"
        result_text = (
            f"생성 완료\n\n"
            f"- 원본 파일: {original_name}\n"
            f"- 기술명: {data.get('technology_name', '')}\n"
            f"- 학과: {data.get('department', '')}\n"
            f"- 교수명: {data.get('professor', '')}\n"
            f"- 대분류: {data.get('field_group', '')}\n"
            f"- 세부분야: {data.get('field_type', '')}\n"
            f"- 키워드: {', '.join(data.get('keywords', []))}\n"
            f"- 스타일 이미지 수: {len(style_image_paths)}\n"
            f"- 기본 스타일 ZIP 자동 사용: {'예' if style_zip_path is None else '아니오(업로드 ZIP 사용)'}\n\n"
            f"[실행 로그]\n" + "\n".join(logs)
        )

        return {
            "success": True,
            "image_bytes": final_img_bytes,
            "image_path": str(out_path),
            "display_text": display_text,
            "result_text": result_text,
            "analysis_data": data,
            "style_ref": style_ref,
            "final_prompt": final_prompt,
            "json_path": str(json_path),
            "prompt_path": str(prompt_path),
            "style_path": str(style_path),
            "file_name": f"{title_part}.png",
        }

    except Exception as e:
        tb = traceback.format_exc()
        log_step(logs, f"오류 발생: {type(e).__name__}: {e}")
        log_step(logs, tb)
        friendly_msg = ""
        if "503" in str(e) or "UNAVAILABLE" in str(e):
            friendly_msg = (
                "Gemini 서버가 일시적으로 혼잡한 상태입니다. 코드 문제보다는 외부 API 과부하 가능성이 큽니다.\n\n"
            )
        return {
            "success": False,
            "error": friendly_msg + "실행 중 오류가 발생했습니다.",
            "result_text": friendly_msg + "실행 중 오류가 발생했습니다.\n\n" + "\n".join(logs),
            "logs": logs,
        }
    finally:
        if extracted_style_dir and os.path.exists(extracted_style_dir):
            shutil.rmtree(extracted_style_dir, ignore_errors=True)


def process_smk_streamlit(uploaded_file, style_zip_file=None, api_key: Optional[str] = None) -> Dict[str, Any]:
    input_path = save_uploaded_file(uploaded_file)
    style_path = save_uploaded_file(style_zip_file) if style_zip_file else None
    try:
        return process_smk_paths(input_path=input_path, style_zip_path=style_path, api_key=api_key)
    finally:
        for path in [input_path, style_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
