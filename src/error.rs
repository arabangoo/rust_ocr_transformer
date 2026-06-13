//! 크레이트 전역 에러 타입.
//!
//! 에러 타입은 optional 의존성(`tract` 등)에 직접 의존하지 않는다 — 백엔드 구체 에러는
//! 문자열로 흡수해 [`VisionError::Backend`] 로 변환한다. 그래야 feature 조합과 무관하게
//! 항상 컴파일된다(백엔드 교체 가능 설계와 정합).

use thiserror::Error;

/// 프레임워크 전 구간(디코드·전처리·추론·후처리)에서 발생할 수 있는 에러.
#[derive(Debug, Error)]
pub enum VisionError {
    /// 파일 열기/읽기 실패.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// 이미지/영상 디코드 실패.
    #[error("decode error: {0}")]
    Decode(String),

    /// 추론 백엔드(모델 로드/실행)가 낸 에러.
    /// 구체 백엔드(tract 등)의 에러는 여기로 문자열 변환되어 흡수된다.
    #[error("backend error: {0}")]
    Backend(String),

    /// 아직 연결되지 않은 기능을 호출함.
    /// (예: 백엔드 feature 없이 추론 요청, 미구현 작업 호출 등)
    #[error("not wired yet: {0}")]
    NotWired(&'static str),

    /// 입력이 지원 범위를 벗어남.
    #[error("unsupported: {0}")]
    Unsupported(String),
}

impl VisionError {
    /// 백엔드 에러 헬퍼 — 구체 추론 라이브러리 에러를 문자열로 흡수.
    pub fn backend(detail: impl Into<String>) -> Self {
        VisionError::Backend(detail.into())
    }

    /// 디코드 에러 헬퍼.
    pub fn decode(detail: impl Into<String>) -> Self {
        VisionError::Decode(detail.into())
    }
}

/// 하위 호환 별칭 — 초기 OCR 전용 시절의 명칭. 기존 코드/문서 호환을 위해 유지한다.
pub type OcrError = VisionError;

/// 크레이트 공통 Result 별칭.
pub type Result<T> = std::result::Result<T, VisionError>;
