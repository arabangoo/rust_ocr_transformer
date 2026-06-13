//! 해자 2 — temporal 병합(README 10장).
//!
//! 게이트를 통과해도 인접 키프레임의 인식 결과는 거의 같은 텍스트일 수 있다
//! (한 글자 흔들림 등). 정규화 Levenshtein 편집거리로 "같은 자막의 연속"을 하나의
//! 시간 구간으로 합친다. 출력은 `(start, end, text)` 구간 리스트 — SRT/JSON/평문의
//! 공통 원천이다.

use crate::types::{Segment, Timestamp};

/// 인식 결과 스트림을 시간 구간으로 병합한다.
///
/// 프레임마다 `push(timestamp, text)` 를 호출하고, 자막이 바뀌는 순간 직전 구간이
/// 확정되어 반환된다. 스트림이 끝나면 `finish()` 로 마지막 구간을 회수한다.
pub struct TemporalMerger {
    /// 현재 누적 중인 구간.
    open: Option<Segment>,
    /// 정규화 편집거리 유사도 임계값(0.0-1.0). 이 값 이상이면 "같은 자막"으로 본다.
    sim_threshold: f64,
}

impl TemporalMerger {
    pub fn new(sim_threshold: f64) -> Self {
        Self { open: None, sim_threshold }
    }

    /// 한 프레임의 인식 텍스트를 밀어넣는다.
    ///
    /// 직전 구간과 같은 자막이면 구간을 연장하고 `None`. 다른 자막이면 직전 구간을
    /// 확정해 `Some(Segment)` 로 돌려주고 새 구간을 연다.
    pub fn push(&mut self, t: Timestamp, text: &str) -> Option<Segment> {
        match self.open.as_mut() {
            Some(seg) if similarity(&seg.text, text) >= self.sim_threshold => {
                // 같은 자막 → 구간 연장 + 더 완전한 인식본 채택.
                seg.end = t;
                if text.chars().count() > seg.text.chars().count() {
                    seg.text = text.to_string();
                }
                None
            }
            _ => {
                let finished = self.open.take();
                self.open = Some(Segment::new(t, text));
                finished
            }
        }
    }

    /// 스트림 종료 — 아직 열려 있는 마지막 구간을 확정해 반환한다.
    pub fn finish(&mut self) -> Option<Segment> {
        self.open.take()
    }
}

/// 두 문자열의 정규화 Levenshtein 유사도(1.0 = 동일, 0.0 = 완전 상이).
/// 빈 문자열 두 개는 1.0 으로 본다.
pub fn similarity(a: &str, b: &str) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    strsim::normalized_levenshtein(a, b)
}
