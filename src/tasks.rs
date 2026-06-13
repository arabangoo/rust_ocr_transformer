//! 작업별 모델 trait — 프레임워크의 "모델을 꽂는 자리(plug point)".
//!
//! 각 trait 은 한 종류의 비전 작업을 추상화한다. 구체 백엔드(tract 순수 Rust, 향후 ort
//! 등)는 이 trait 들의 구현체로 들어오므로, 코어 파이프라인은 어떤 모델을 쓰는지 모른 채
//! 작업 단위로만 조립된다. `Send + Sync` 바운드가 멀티스레드 파이프라인에서 스레드 간
//! 공유를 가능케 한다.
//!
//! 입력은 모두 공통 [`Frame`] 이고, 출력은 작업별 결과 타입이다. 여러 작업의 출력은
//! [`FrameAnalysis`](crate::types::FrameAnalysis) 로 모을 수 있다.

use crate::error::Result;
use crate::types::{Classification, Crop, Detection, Frame, LayoutRegion, Mask, Recognized, TextBox};

/// 텍스트 영역(박스) 검출. (OCR 파이프라인 1단계.)
pub trait TextDetector: Send + Sync {
    fn detect(&self, frame: &Frame) -> Result<Vec<TextBox>>;
}

/// 잘라낸 텍스트 영역들에서 문자열 인식. 배치 입력으로 처리량 극대화. (OCR 2단계.)
pub trait TextRecognizer: Send + Sync {
    fn recognize(&self, crops: &[Crop]) -> Result<Vec<Recognized>>;
}

/// 범용 객체 검출 — 클래스 라벨 + 박스 + 점수. (YOLO 계열 등.)
pub trait ObjectDetector: Send + Sync {
    fn detect_objects(&self, frame: &Frame) -> Result<Vec<Detection>>;
}

/// 이미지 분류 — 전체 프레임에 대한 라벨 + 점수(top-k).
pub trait Classifier: Send + Sync {
    fn classify(&self, frame: &Frame) -> Result<Vec<Classification>>;
}

/// 문서 레이아웃 분석 — 구조 영역(title/text/table/figure 등) 검출.
/// 사실상 라벨이 문서 구조인 객체 검출의 특수형이다.
pub trait LayoutAnalyzer: Send + Sync {
    fn analyze_layout(&self, frame: &Frame) -> Result<Vec<LayoutRegion>>;
}

/// 시맨틱 세그멘테이션 — 픽셀당 클래스 마스크.
pub trait Segmenter: Send + Sync {
    fn segment(&self, frame: &Frame) -> Result<Mask>;
}
