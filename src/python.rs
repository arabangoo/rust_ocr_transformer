//! PyO3 바인딩 — `feature = "python"` 활성 시 cdylib 으로 빌드되어
//! Python 에서 `import rust_ocr_transformer` 로 사용한다. abi3(stable ABI)라
//! Python 3.9+ 단일 휠로 호환된다(rust_markdown_transformer 와 동일 배포 패턴).
//!
//! ## 노출 범위
//!   - `recognize_image(...)` — 이미지 OCR(검출+인식) → 인식 결과 JSON (`feature = "tract"`)
//!   - `image_ssim(a, b)` — 두 이미지의 구조적 유사도(샘플링 게이트의 판정 근거)
//!   - `segments_to_srt(json)` — 시간 구간 JSON → SRT 자막 문자열
//!
//! 영상 처리(`read_video`)는 영상 디코드(Phase 2)와 함께 추가된다.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::types::Segment;

/// 이미지 OCR — 검출 + 인식을 실행해 인식 결과(JSON 문자열)를 돌려준다.
///
/// 반환: `[{"text": "...", "confidence": 0.97, "bbox": {"x":..,"y":..,"width":..,"height":..}}, ...]`
///
/// 모델·사전 파일은 호출자가 제공한다(README 11장의 PP-OCR ONNX + 전용 사전).
/// 입력 형상 기본값은 PP-OCRv5 권장값(검출 736x1280, 인식 48x320).
///
/// ```python
/// import json, rust_ocr_transformer as roct
/// out = json.loads(roct.recognize_image("page.png", "det.onnx", "rec.onnx", "dict.txt"))
/// for r in out:
///     print(r["confidence"], r["text"])
/// ```
#[cfg(feature = "tract")]
#[pyfunction]
#[pyo3(signature = (
    image_path, det_model, rec_model, dict_path,
    det_height = 736, det_width = 1280, rec_height = 48, rec_width = 320,
))]
#[allow(clippy::too_many_arguments)]
fn recognize_image(
    image_path: &str,
    det_model: &str,
    rec_model: &str,
    dict_path: &str,
    det_height: usize,
    det_width: usize,
    rec_height: usize,
    rec_width: usize,
) -> PyResult<String> {
    use crate::{Frame, OcrEngine, TractTextDetector, TractTextRecognizer};

    let frame = Frame::from_path(image_path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let detector = TractTextDetector::new(det_model, (det_height, det_width))
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let recognizer = TractTextRecognizer::new(rec_model, dict_path, (rec_height, rec_width))
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let engine = OcrEngine::new(detector, recognizer);

    let results = engine.read(&frame).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    serde_json::to_string(&results).map_err(|e| PyRuntimeError::new_err(format!("serialize: {e}")))
}

/// 두 이미지 파일의 전역 SSIM(0.0-1.0). 1.0 에 가까울수록 동일.
/// 샘플링 게이트가 프레임을 버릴지 판정하는 데 쓰는 바로 그 지표.
#[pyfunction]
fn image_ssim(a: &str, b: &str) -> PyResult<f64> {
    let ia = image::open(a).map_err(|e| PyRuntimeError::new_err(format!("open {a}: {e}")))?;
    let ib = image::open(b).map_err(|e| PyRuntimeError::new_err(format!("open {b}: {e}")))?;
    let ga = crate::sampler::downscale_gray(&ia, (64, 64));
    let gb = crate::sampler::downscale_gray(&ib, (64, 64));
    Ok(crate::sampler::ssim(&ga, &gb))
}

/// 시간 구간 JSON 배열(`[{"start":{...},"end":{...},"text":"..."}]`) → SRT 자막.
#[pyfunction]
fn segments_to_srt(segments_json: &str) -> PyResult<String> {
    let segs: Vec<Segment> = serde_json::from_str(segments_json)
        .map_err(|e| PyRuntimeError::new_err(format!("parse segments json: {e}")))?;
    Ok(crate::emit::to_srt(&segs))
}

/// Python 모듈 정의 — 모듈명은 cdylib 이름(`rust_ocr_transformer`)과 일치해야 한다.
#[pymodule]
fn rust_ocr_transformer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    #[cfg(feature = "tract")]
    m.add_function(wrap_pyfunction!(recognize_image, m)?)?;
    m.add_function(wrap_pyfunction!(image_ssim, m)?)?;
    m.add_function(wrap_pyfunction!(segments_to_srt, m)?)?;
    Ok(())
}
