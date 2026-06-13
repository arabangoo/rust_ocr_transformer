//! 출력 직렬화(README 10.1) — 시간 구간 리스트를 SRT/JSON/평문으로.
//!
//! `(start, end, text)` 구간 스키마를 문서 파서(rust_markdown_transformer)의 출력과
//! 통일하면 텍스트 입구와 비전 입구가 단일 인제스트 인터페이스로 합쳐진다(README 18장).

use crate::error::Result;
use crate::types::Segment;

/// 구간 리스트를 SRT 자막 형식으로 직렬화.
///
/// ```text
/// 1
/// 00:00:01,200 --> 00:00:03,800
/// 첫 자막
///
/// 2
/// 00:00:04,000 --> 00:00:06,500
/// 둘째 자막
/// ```
pub fn to_srt(segments: &[Segment]) -> String {
    let mut out = String::new();
    for (i, seg) in segments.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        out.push_str(&format!("{}\n", i + 1));
        out.push_str(&format!("{} --> {}\n", seg.start.to_srt(), seg.end.to_srt()));
        out.push_str(&seg.text);
        out.push('\n');
    }
    out
}

/// 구간 리스트를 JSON 배열로 직렬화(타임스탬프는 밀리초).
pub fn to_json(segments: &[Segment]) -> Result<String> {
    serde_json::to_string_pretty(segments)
        .map_err(|e| crate::error::OcrError::backend(format!("json serialize: {e}")))
}

/// 구간 텍스트만 줄바꿈으로 이어 붙인 평문.
pub fn to_plain(segments: &[Segment]) -> String {
    segments
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join("\n")
}
