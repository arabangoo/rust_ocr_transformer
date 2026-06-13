//! 재사용 후처리 — 모델 출력 텐서를 구조화 결과로 변환하는 순수 함수 모음.
//!
//! 모델 무관 순수 로직(softmax·argmax·IoU·NMS)과 작업별 디코딩(DB 연결요소 박스, CTC
//! 그리디 디코딩)을 함께 둔다. 순수 로직은 단위 테스트로 검증된다(모델 불필요).

use crate::types::BBox;

// ── 모델 무관 순수 로직 ───────────────────────────────────────────

/// 수치 안정 softmax. 입력 logits → 확률 분포(합 1).
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exps: Vec<f32> = logits.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum > 0.0 {
        for e in &mut exps {
            *e /= sum;
        }
    }
    exps
}

/// 최댓값 인덱스와 값(top-1).
pub fn argmax(v: &[f32]) -> (usize, f32) {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best = i;
        }
    }
    (best, best_v)
}

/// 두 박스의 IoU(교집합/합집합). 0.0-1.0.
pub fn iou(a: &BBox, b: &BBox) -> f32 {
    let ax2 = a.x + a.width;
    let ay2 = a.y + a.height;
    let bx2 = b.x + b.width;
    let by2 = b.y + b.height;

    let ix1 = a.x.max(b.x);
    let iy1 = a.y.max(b.y);
    let ix2 = ax2.min(bx2);
    let iy2 = ay2.min(by2);
    if ix2 <= ix1 || iy2 <= iy1 {
        return 0.0;
    }
    let inter = (ix2 - ix1) as f64 * (iy2 - iy1) as f64;
    let union = a.area() as f64 + b.area() as f64 - inter;
    if union <= 0.0 {
        0.0
    } else {
        (inter / union) as f32
    }
}

/// 비최대 억제(Non-Maximum Suppression). 점수 내림차순으로 박스를 유지하되, 이미 채택된
/// 박스와 IoU 가 임계값을 넘으면 버린다. 반환: 유지된 박스의 원본 인덱스(점수 높은 순).
pub fn nms(boxes: &[(BBox, f32)], iou_threshold: f32) -> Vec<usize> {
    let mut order: Vec<usize> = (0..boxes.len()).collect();
    order.sort_by(|&i, &j| boxes[j].1.partial_cmp(&boxes[i].1).unwrap_or(std::cmp::Ordering::Equal));

    let mut keep = Vec::new();
    let mut suppressed = vec![false; boxes.len()];
    for &i in &order {
        if suppressed[i] {
            continue;
        }
        keep.push(i);
        for &j in &order {
            if j != i && !suppressed[j] && iou(&boxes[i].0, &boxes[j].0) > iou_threshold {
                suppressed[j] = true;
            }
        }
    }
    keep
}

// ── 작업별 디코딩 ─────────────────────────────────────────────────

/// CTC 그리디 디코딩 — 타임스텝별 argmax → 반복 축약 → blank(0) 제거 → 문자 매핑.
/// logits 는 [T, C] 행 우선 평탄화, blank=0, 클래스 k(>=1) → `dict[k-1]`.
/// 반환: (text, mean_confidence).
pub fn ctc_greedy_decode(logits: &[f32], t: usize, c: usize, dict: &[String]) -> (String, f32) {
    let mut text = String::new();
    let mut conf_sum = 0f32;
    let mut conf_n = 0usize;
    let mut prev_class = usize::MAX;

    for step in 0..t {
        let row = &logits[step * c..(step + 1) * c];
        let (best, best_v) = argmax(row);
        if best != 0 && best != prev_class {
            if let Some(ch) = dict.get(best - 1) {
                text.push_str(ch);
                conf_sum += best_v;
                conf_n += 1;
            }
        }
        prev_class = best;
    }

    let conf = if conf_n > 0 { conf_sum / conf_n as f32 } else { 0.0 };
    (text, conf)
}

/// DB(검출) 확률맵에서 연결 요소(4-이웃 BFS)별 축 정렬 bbox + 평균 확률을 추출한다.
/// 반환: (x, y, w, h, score) — 검출공간 픽셀 좌표.
pub fn connected_boxes(
    prob: &[f32],
    w: usize,
    h: usize,
    threshold: f32,
    min_area: usize,
) -> Vec<(usize, usize, usize, usize, f32)> {
    let mut visited = vec![false; w * h];
    let mut boxes = Vec::new();
    let mut stack: Vec<usize> = Vec::new();

    for start in 0..w * h {
        if visited[start] || prob[start] < threshold {
            continue;
        }
        let (mut min_x, mut min_y, mut max_x, mut max_y) = (usize::MAX, usize::MAX, 0usize, 0usize);
        let mut sum = 0f32;
        let mut count = 0usize;
        stack.clear();
        stack.push(start);
        visited[start] = true;

        while let Some(idx) = stack.pop() {
            let (px, py) = (idx % w, idx / w);
            min_x = min_x.min(px);
            min_y = min_y.min(py);
            max_x = max_x.max(px);
            max_y = max_y.max(py);
            sum += prob[idx];
            count += 1;

            let push = |nx: isize, ny: isize, stack: &mut Vec<usize>, visited: &mut [bool]| {
                if nx >= 0 && ny >= 0 && (nx as usize) < w && (ny as usize) < h {
                    let nidx = ny as usize * w + nx as usize;
                    if !visited[nidx] && prob[nidx] >= threshold {
                        visited[nidx] = true;
                        stack.push(nidx);
                    }
                }
            };
            push(px as isize - 1, py as isize, &mut stack, &mut visited);
            push(px as isize + 1, py as isize, &mut stack, &mut visited);
            push(px as isize, py as isize - 1, &mut stack, &mut visited);
            push(px as isize, py as isize + 1, &mut stack, &mut visited);
        }

        let bw = max_x - min_x + 1;
        let bh = max_y - min_y + 1;
        if bw * bh >= min_area {
            boxes.push((min_x, min_y, bw, bh, sum / count as f32));
        }
    }
    boxes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_sums_to_one_and_monotonic() {
        let p = softmax(&[1.0, 2.0, 3.0]);
        let sum: f32 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(p[2] > p[1] && p[1] > p[0], "큰 logit 일수록 큰 확률");
    }

    #[test]
    fn argmax_picks_max() {
        let (i, v) = argmax(&[0.1, 0.9, 0.3]);
        assert_eq!(i, 1);
        assert!((v - 0.9).abs() < 1e-6);
    }

    #[test]
    fn iou_identical_and_disjoint() {
        let a = BBox::new(0, 0, 10, 10);
        assert!((iou(&a, &a) - 1.0).abs() < 1e-5, "동일 박스 IoU=1");
        let b = BBox::new(100, 100, 10, 10);
        assert_eq!(iou(&a, &b), 0.0, "떨어진 박스 IoU=0");
        let c = BBox::new(5, 0, 10, 10); // 50% 가로 겹침
        let v = iou(&a, &c);
        assert!(v > 0.3 && v < 0.4, "절반 겹침 IoU≈0.33, got {v}");
    }

    #[test]
    fn nms_suppresses_overlap_keeps_separate() {
        let boxes = vec![
            (BBox::new(0, 0, 10, 10), 0.9),   // 0: 최고 점수
            (BBox::new(1, 1, 10, 10), 0.8),   // 1: 0과 크게 겹침 → 억제
            (BBox::new(100, 100, 10, 10), 0.7), // 2: 떨어짐 → 유지
        ];
        let keep = nms(&boxes, 0.5);
        assert_eq!(keep, vec![0, 2], "겹친 저점수는 억제, 떨어진 건 유지");
    }

    #[test]
    fn ctc_collapses_and_drops_blank() {
        let dict = vec!["a".to_string(), "b".to_string()]; // class1=a, class2=b
        // T=4, C=3 (blank=0). 시퀀스: a a blank b → "ab"
        let logits = vec![
            0.0, 1.0, 0.0, // a
            0.0, 1.0, 0.0, // a (반복 축약)
            1.0, 0.0, 0.0, // blank
            0.0, 0.0, 1.0, // b
        ];
        let (text, conf) = ctc_greedy_decode(&logits, 4, 3, &dict);
        assert_eq!(text, "ab");
        assert!(conf > 0.0);
    }
}
