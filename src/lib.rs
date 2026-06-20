//! # rust_ocr_transformer — 통합 비전 추론·처리 프레임워크 (Rust)
//!
//! 이미지·영상을 **디코드 → 전처리 → 신경망 모델 추론 → 후처리 → 구조화 출력**으로
//! 흘려보내는 순수 Rust 프레임워크. 단일 만능 모델이 아니라, 작업별 모델을
//! [`tasks`] 의 trait 뒤에 꽂아 교체하는 **오케스트레이션 틀**이다.
//!
//! ## 작업(task)과 현재 상태
//! - 텍스트 검출/인식(OCR) — [`TextDetector`]/[`TextRecognizer`], [`OcrEngine`]: 실모델 동작
//!   검증됨(PP-OCRv5 한·영·일 등 실제 화면 사진으로 확인). DB unclip·XY-Cut 읽기순서·자동
//!   방향 보정([`recognize_image_auto`]) 포함
//! - 이미지 분류 — [`Classifier`]: tract 백엔드(컴파일됨, 실모델 정확도 미검증)
//! - 객체 검출 — [`ObjectDetector`]: tract 백엔드(출력 레이아웃 가정, 미검증)
//! - 레이아웃 분석 — [`LayoutAnalyzer`]: trait 정의(레이아웃 라벨을 가진 객체 검출의 특수형)
//! - 세그멘테이션 — [`Segmenter`]: trait 정의(구체 백엔드 미구현)
//! - 영상 시간축 — [`SamplingGate`](SSIM 게이트) + [`TemporalMerger`](중복 병합): 동작·테스트 완료
//!
//! ## 코어는 LLM-free
//! 순수 인식·검출·분류는 소형 특화 모델로 충분하다. "이해(VQA·추론)"가 필요하면 그건
//! 코어 밖의 대형 모델(서버)에 위임하고, 이 프레임워크는 그 입력(구조화된 비전 결과)을
//! 만든다.
//!
//! ## 추론 런타임
//! 기본은 순수 Rust(`tract`, feature `tract`) — C++ FFI 없음, 클린 abi3 휠. 속도·GPU 가
//! 필요하면 ort 백엔드를 opt-in feature 로 추가한다(같은 trait 뒤).
//!
//! ## 모듈
//! - [`types`] — 공통 데이터 타입(Frame, BBox, 작업별 결과, FrameAnalysis)
//! - [`tasks`] — 작업별 모델 trait(plug point)
//! - [`preprocess`] / [`postprocess`] — 재사용 전·후처리(letterbox·정규화 / NMS·softmax·CTC·DB)
//! - [`engine`] — OCR 작업 합성 엔진
//! - [`sampler`] / [`temporal`] / [`emit`] — 영상 시간축 처리와 출력 직렬화
//! - [`backends`] — 구체 추론 백엔드(tract)

pub mod backends;
pub mod emit;
pub mod engine;
pub mod error;
pub mod postprocess;
pub mod preprocess;
pub mod sampler;
pub mod tasks;
pub mod temporal;
pub mod types;

#[cfg(feature = "python")]
mod python;

// ── 자주 쓰는 타입 re-export ──────────────────────────────────
pub use emit::{to_json, to_plain, to_srt};
pub use engine::{crop_regions, OcrEngine};
pub use error::{OcrError, Result, VisionError};
pub use sampler::{ssim, SamplingGate};
pub use tasks::{Classifier, LayoutAnalyzer, ObjectDetector, Segmenter, TextDetector, TextRecognizer};
pub use temporal::TemporalMerger;
pub use types::{
    BBox, Classification, Crop, Detection, Frame, FrameAnalysis, LayoutRegion, Mask, Recognized,
    Segment, TextBox, Timestamp,
};

// 순수 Rust 추론 백엔드(tract) — feature 활성 시 노출.
#[cfg(feature = "tract")]
pub use backends::tract::{
    recognize_image_auto, TractClassifier, TractDocOrientation, TractModel, TractObjectDetector,
    TractTextDetector, TractTextRecognizer,
};
