//! 프레임워크 전 구간을 흐르는 핵심 데이터 타입.
//!
//! 설계 원칙: **confidence 와 좌표는 1급 시민이다.** 후단 하이브리드 패턴(직렬 후보정·
//! confidence 라우팅·필드 분담)이 전부 신뢰도를 전제하므로, 모든 결과 타입에 score 를
//! 처음부터 박는다. 작업별 결과(텍스트 인식·객체 검출·분류·레이아웃·세그멘테이션)는
//! 공통 [`Frame`] 입력을 공유하고 [`FrameAnalysis`] 로 한데 모인다.

use serde::{Deserialize, Serialize};

use crate::error::{Result, VisionError};

/// 픽셀 좌표계의 사각 영역. 좌상단 원점.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BBox {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl BBox {
    pub fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self { x, y, width, height }
    }

    /// 면적(px).
    pub fn area(&self) -> u64 {
        self.width as u64 * self.height as u64
    }
}

/// 파이프라인 1단위 = 한 프레임. 이미지는 프레임이 1개인 영상의 퇴화 사례다.
/// 그래서 이미지·영상이 동일 타입을 공유한다.
#[derive(Debug, Clone)]
pub struct Frame {
    /// 디코드된 픽셀 데이터.
    pub image: image::DynamicImage,
    /// 영상 내 프레임 순번(이미지 단건이면 0).
    pub index: u64,
    /// 프레임의 표시 시각(밀리초). 이미지 단건이면 0.
    pub timestamp: Timestamp,
}

impl Frame {
    /// 단일 이미지를 프레임 1개로 감싼다(index=0, timestamp=0).
    pub fn from_image(image: image::DynamicImage) -> Self {
        Self { image, index: 0, timestamp: Timestamp(0) }
    }

    /// 영상 프레임 — 순번과 시각을 명시.
    pub fn new(image: image::DynamicImage, index: u64, timestamp: Timestamp) -> Self {
        Self { image, index, timestamp }
    }

    /// 디코드 입구 — 이미지 파일 경로를 읽어 프레임으로. (영상 디코드는 Phase 2.)
    pub fn from_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let p = path.as_ref();
        let img = image::open(p).map_err(|e| VisionError::decode(format!("open {}: {e}", p.display())))?;
        Ok(Self::from_image(img))
    }
}

// ── 작업별 결과 타입 ──────────────────────────────────────────────

/// 텍스트 검출기가 찾아낸 텍스트 영역. score 는 검출 신뢰도.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TextBox {
    pub bbox: BBox,
    pub confidence: f32,
}

/// 검출 박스로 잘라낸 영역 이미지. 인식기 입력 단위.
#[derive(Debug, Clone)]
pub struct Crop {
    pub image: image::DynamicImage,
    pub bbox: BBox,
}

/// 텍스트 인식 결과. text + confidence + 좌표가 한 묶음.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Recognized {
    pub text: String,
    pub confidence: f32,
    pub bbox: BBox,
}

/// 범용 객체 검출 결과(클래스 라벨 + 점수 + 박스).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Detection {
    pub bbox: BBox,
    pub label: String,
    pub score: f32,
}

/// 이미지 분류 결과(라벨 + 점수).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Classification {
    pub label: String,
    pub score: f32,
}

/// 레이아웃 영역(종류 라벨 + 박스 + 점수). 객체 검출의 특수형 — 라벨이 문서 구조
/// (title/text/table/figure/header/footer 등)를 가리킨다.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayoutRegion {
    pub bbox: BBox,
    pub kind: String,
    pub score: f32,
}

/// 세그멘테이션 마스크 — 픽셀당 클래스 id.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Mask {
    pub width: u32,
    pub height: u32,
    /// row-major, width*height 길이. 값 = 클래스 id.
    pub classes: Vec<u8>,
}

/// 한 프레임의 종합 분석 결과 — 여러 작업의 출력을 모으는 구조화 출력 컨테이너.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FrameAnalysis {
    pub recognized: Vec<Recognized>,
    pub detections: Vec<Detection>,
    pub classifications: Vec<Classification>,
    pub layout: Vec<LayoutRegion>,
}

// ── 시간축(영상) 타입 ─────────────────────────────────────────────

/// 밀리초 단위 타임스탬프. SRT/JSON 출력의 공통 시간 표현.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Timestamp(pub u64);

impl Timestamp {
    pub fn from_millis(ms: u64) -> Self {
        Timestamp(ms)
    }

    pub fn millis(self) -> u64 {
        self.0
    }

    /// SRT 시간 형식 `HH:MM:SS,mmm` 으로 변환.
    pub fn to_srt(self) -> String {
        let ms = self.0 % 1000;
        let total_secs = self.0 / 1000;
        let s = total_secs % 60;
        let m = (total_secs / 60) % 60;
        let h = total_secs / 3600;
        format!("{h:02}:{m:02}:{s:02},{ms:03}")
    }
}

/// temporal 병합의 출력 단위 = 같은 텍스트가 지속된 시간 구간.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Segment {
    pub start: Timestamp,
    pub end: Timestamp,
    pub text: String,
}

impl Segment {
    pub fn new(start: Timestamp, text: impl Into<String>) -> Self {
        Self { start, end: start, text: text.into() }
    }
}
