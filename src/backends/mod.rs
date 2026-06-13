//! 인식 백엔드 — [`TextDetector`](crate::TextDetector)/[`TextRecognizer`](crate::TextRecognizer)
//! 의 구현체들. 백엔드는 trait 뒤에서 교체 가능하다(README ADR-3).
//!
//! - [`tract`] — 순수 Rust ONNX 추론(`feature = "tract"`, 기본). C++ FFI 없음.
//! - (예약) ort 백엔드 — 속도·GPU 가 필요할 때의 opt-in FFI 경로. 다음 단계에서 추가.

#[cfg(feature = "tract")]
pub mod tract;
