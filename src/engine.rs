//! OCR 작업 파이프라인 — 텍스트 검출 + 인식을 합성한 엔진.
//!
//! 프레임워크의 작업 trait([`TextDetector`]/[`TextRecognizer`], [`crate::tasks`])을 조립한
//! "OCR 작업" 인스턴스다. 검출과 인식을 분리하되 합성 타입으로 묶는다 — 영상에서 자막
//! 위치는 고정이라 검출을 캐시/스킵하고 인식만 매번 돌리는 최적화가 가능하고, 백엔드마다
//! 검출·인식 강점이 달라 믹스앤매치가 필요하기 때문이다. 같은 패턴으로 다른 작업(객체
//! 검출·분류 등)도 각자의 trait 으로 조립한다.

use crate::error::Result;
use crate::tasks::{TextDetector, TextRecognizer};
use crate::types::{Crop, Frame, Recognized, TextBox};

/// 검출기 + 인식기를 합성한 상위 엔진.
///
/// 단일 프레임 처리(이미지 = 1프레임 영상)의 진입점. 영상 경로는 이 `read` 를
/// 샘플링 게이트가 통과시킨 프레임에 대해서만 호출한다.
pub struct OcrEngine<D: TextDetector, R: TextRecognizer> {
    detector: D,
    recognizer: R,
}

impl<D: TextDetector, R: TextRecognizer> OcrEngine<D, R> {
    pub fn new(detector: D, recognizer: R) -> Self {
        Self { detector, recognizer }
    }

    /// 한 프레임을 읽어 인식 결과 목록을 반환한다: 검출 → 크롭 → 인식.
    pub fn read(&self, frame: &Frame) -> Result<Vec<Recognized>> {
        let boxes = self.detector.detect(frame)?;
        let crops = crop_regions(frame, &boxes);
        self.recognizer.recognize(&crops)
    }

    pub fn detector(&self) -> &D {
        &self.detector
    }

    pub fn recognizer(&self) -> &R {
        &self.recognizer
    }
}

/// 검출 박스대로 프레임에서 영역 이미지를 잘라낸다.
/// bbox 를 함께 담아(`Crop`) 인식 결과에 좌표를 되붙일 수 있게 한다.
pub fn crop_regions(frame: &Frame, boxes: &[TextBox]) -> Vec<Crop> {
    boxes
        .iter()
        .map(|tb| {
            let b = tb.bbox;
            // crop_imm 은 원본을 복사해 잘라낸 뷰를 만든다(원본 불변).
            let cropped = frame.image.crop_imm(b.x, b.y, b.width, b.height);
            Crop { image: cropped, bbox: b }
        })
        .collect()
}
