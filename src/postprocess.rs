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

/// 검출 박스를 읽기 순서로 정렬한 인덱스를 돌려준다: 위→아래 줄, 한 줄 안에서는 왼→오른쪽.
///
/// 연결요소 추출([`connected_boxes`])은 확률맵 래스터 스캔 순서로 박스를 내놓아, 같은 줄의
/// 단어들이 좌우 뒤섞여 나올 수 있다(예: 줄 끝 단어가 먼저). 자막·문서 라인 같은 축 정렬
/// 가로 텍스트를 가정하고, 세로 중심으로 줄을 묶은 뒤 줄 안에서 x 로 정렬한다.
///
/// 같은 줄 판정: 박스의 세로 중심이 현재 줄 기준에서 그 박스 높이의 절반을 넘게 벗어나면
/// 새 줄로 본다(글자 높이 정도의 흔들림은 같은 줄로 흡수).
pub fn reading_order(boxes: &[BBox]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..boxes.len()).collect();
    if idx.len() < 2 {
        return idx;
    }
    let cy = |b: &BBox| b.y as f32 + b.height as f32 / 2.0;
    // 1) 세로 중심 오름차순.
    idx.sort_by(|&i, &j| cy(&boxes[i]).partial_cmp(&cy(&boxes[j])).unwrap_or(std::cmp::Ordering::Equal));
    // 2) 줄 단위로 묶는다(현재 줄의 첫 박스 세로중심을 기준으로 비교).
    let mut lines: Vec<Vec<usize>> = Vec::new();
    for &i in &idx {
        let ci = cy(&boxes[i]);
        let same_line = lines.last().is_some_and(|line| {
            let first = *line.first().unwrap();
            (ci - cy(&boxes[first])).abs() <= boxes[i].height as f32 * 0.5
        });
        if same_line {
            lines.last_mut().unwrap().push(i);
        } else {
            lines.push(vec![i]);
        }
    }
    // 3) 각 줄 안에서 x 오름차순으로 펼친다.
    let mut out = Vec::with_capacity(boxes.len());
    for mut line in lines {
        line.sort_by_key(|&i| boxes[i].x);
        out.extend(line);
    }
    out
}

// ── XY-Cut 읽기순서 (재귀 투영 분할) ─────────────────────────────────
//
// 박스들을 Y 축으로 투영해 가로 띠(행)로 가르고, 각 띠를 다시 X 축으로 투영해 세로 단(열)
// 으로 가른다. 단이 더 안 갈리면 그 묶음을 읽기순서로 확정하고, 갈리면 각 단을 재귀한다.
// [`reading_order`] 의 단순 줄 묶기와 달리 다단·표·나란한 패널(표 + 설명) 같은 복잡한
// 레이아웃의 읽기순서를 올바로 잡는다(왼쪽 단을 끝까지 읽고 다음 단으로).

type Rect = (u32, u32, u32, u32); // (left, top, right, bottom)

/// boxes 의 [start,end) 구간을 axis(0=x, 1=y)로 투영한 1D 카운트 히스토그램.
fn xy_projection(boxes: &[Rect], axis: usize) -> Vec<u32> {
    let max = boxes.iter().map(|b| if axis == 0 { b.2 } else { b.3 }).max().unwrap_or(0) as usize;
    let mut hist = vec![0u32; max];
    for b in boxes {
        let (s, e) = if axis == 0 { (b.0 as usize, b.2 as usize) } else { (b.1 as usize, b.3 as usize) };
        for h in hist.iter_mut().take(e.min(max)).skip(s) {
            *h += 1;
        }
    }
    hist
}

/// 투영 히스토그램에서 값이 min_value 초과인 연속 구간들의 (start, end[exclusive]).
/// 비영 위치의 인덱스 간격이 min_gap 초과면 다른 구간으로 가른다.
fn xy_split(hist: &[u32], min_value: u32, min_gap: usize) -> Vec<(usize, usize)> {
    let idx: Vec<usize> = hist.iter().enumerate().filter(|(_, &v)| v > min_value).map(|(i, _)| i).collect();
    if idx.is_empty() {
        return Vec::new();
    }
    let mut starts = vec![idx[0]];
    let mut ends = Vec::new();
    for w in idx.windows(2) {
        if w[1] - w[0] > min_gap {
            ends.push(w[0]);
            starts.push(w[1]);
        }
    }
    ends.push(*idx.last().unwrap());
    starts.into_iter().zip(ends).map(|(s, e)| (s, e + 1)).collect()
}

fn xy_cut_rec(boxes: &[Rect], indices: &[usize], res: &mut Vec<usize>, min_gap: usize) {
    if boxes.is_empty() {
        return;
    }
    // Y 정렬 후 가로 띠로 분할.
    let mut yo: Vec<usize> = (0..boxes.len()).collect();
    yo.sort_by_key(|&i| boxes[i].1);
    let yb: Vec<Rect> = yo.iter().map(|&i| boxes[i]).collect();
    let yi: Vec<usize> = yo.iter().map(|&i| indices[i]).collect();
    for (r0, r1) in xy_split(&xy_projection(&yb, 1), 0, min_gap) {
        let sel: Vec<usize> = (0..yb.len()).filter(|&i| (yb[i].1 as usize) >= r0 && (yb[i].1 as usize) < r1).collect();
        // 띠 안에서 X 정렬 후 세로 단으로 분할.
        let bb: Vec<Rect> = sel.iter().map(|&i| yb[i]).collect();
        let bi: Vec<usize> = sel.iter().map(|&i| yi[i]).collect();
        let mut xo: Vec<usize> = (0..bb.len()).collect();
        xo.sort_by_key(|&i| bb[i].0);
        let xb: Vec<Rect> = xo.iter().map(|&i| bb[i]).collect();
        let xi: Vec<usize> = xo.iter().map(|&i| bi[i]).collect();
        let cols = xy_split(&xy_projection(&xb, 0), 0, min_gap);
        match cols.len() {
            0 => continue,
            1 => res.extend(&xi), // 단일 단 → x 순서로 확정
            _ => {
                for (c0, c1) in cols {
                    let csel: Vec<usize> =
                        (0..xb.len()).filter(|&i| (xb[i].0 as usize) >= c0 && (xb[i].0 as usize) < c1).collect();
                    let cb: Vec<Rect> = csel.iter().map(|&i| xb[i]).collect();
                    let ci: Vec<usize> = csel.iter().map(|&i| xi[i]).collect();
                    xy_cut_rec(&cb, &ci, res, min_gap);
                }
            }
        }
    }
}

/// 재귀 XY-Cut 으로 검출 박스의 읽기순서 인덱스를 구한다. 다단·표·나란한 패널 레이아웃을
/// [`reading_order`] 의 단순 줄 묶기보다 정확히 처리한다(왼쪽 단을 끝까지 읽고 다음 단으로).
/// `min_gap` 은 단/행을 가르는 최소 빈 픽셀(권장 8 — 자간보다 크고 단 간격보다 작게).
pub fn xy_cut_order(boxes: &[BBox], min_gap: usize) -> Vec<usize> {
    if boxes.len() < 2 {
        return (0..boxes.len()).collect();
    }
    let rects: Vec<Rect> = boxes
        .iter()
        .map(|b| (b.x, b.y, b.x + b.width.max(1), b.y + b.height.max(1)))
        .collect();
    let indices: Vec<usize> = (0..rects.len()).collect();
    let mut res = Vec::with_capacity(rects.len());
    xy_cut_rec(&rects, &indices, &mut res, min_gap);
    // 투영이 0이라 빠진 퇴화 박스는 끝에 원래 순서로 덧붙여 모든 인덱스를 보존.
    if res.len() < boxes.len() {
        let seen: std::collections::HashSet<usize> = res.iter().copied().collect();
        res.extend((0..boxes.len()).filter(|i| !seen.contains(i)));
    }
    res
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

    #[test]
    fn reading_order_groups_lines_then_sorts_x() {
        // 1줄(y≈50): 오른쪽 단어(x=150)가 왼쪽(x=40)보다 먼저 들어옴(래스터 스캔 흉내).
        // 2줄(y≈120): 가운데 한 단어. 같은 줄 흔들림(y 50 vs 52)은 한 줄로 흡수돼야 한다.
        let boxes = vec![
            BBox::new(150, 50, 30, 20), // 0: 1줄 오른쪽
            BBox::new(40, 52, 30, 20),  // 1: 1줄 왼쪽
            BBox::new(200, 120, 30, 20),// 2: 2줄
        ];
        // 기대: 1줄을 왼→오른(1, 0) 정렬 후 2줄(2).
        assert_eq!(reading_order(&boxes), vec![1, 0, 2]);
    }

    #[test]
    fn reading_order_handles_trivial() {
        assert_eq!(reading_order(&[]), Vec::<usize>::new());
        assert_eq!(reading_order(&[BBox::new(0, 0, 5, 5)]), vec![0]);
    }

    #[test]
    fn xy_cut_reads_columns_top_to_bottom() {
        // 왼쪽 단(세로로 긴 한 박스) + 오른쪽 단(위·아래 두 박스). 가운데 세로 공백(gutter)
        // 으로 나뉘는 다단 레이아웃 → 왼 단을 먼저, 그다음 오른 단(위→아래).
        let boxes = vec![
            BBox::new(0, 0, 40, 90),    // 0: 왼쪽 단
            BBox::new(100, 0, 40, 30),  // 1: 오른쪽 단 위
            BBox::new(100, 50, 40, 30), // 2: 오른쪽 단 아래
        ];
        assert_eq!(xy_cut_order(&boxes, 8), vec![0, 1, 2]);
    }

    #[test]
    fn xy_cut_reads_rows_when_horizontally_separated() {
        // 행 사이 가로 공백이 있는 2x2 격자 → 윗행(좌→우), 아랫행(좌→우).
        let boxes = vec![
            BBox::new(0, 0, 40, 30),    // 0: 좌상
            BBox::new(100, 0, 40, 30),  // 1: 우상
            BBox::new(0, 50, 40, 30),   // 2: 좌하
            BBox::new(100, 50, 40, 30), // 3: 우하
        ];
        assert_eq!(xy_cut_order(&boxes, 8), vec![0, 1, 2, 3]);
    }

    #[test]
    fn xy_cut_trivial() {
        assert_eq!(xy_cut_order(&[], 8), Vec::<usize>::new());
        assert_eq!(xy_cut_order(&[BBox::new(0, 0, 5, 5)], 8), vec![0]);
    }
}
