//! 순수 Rust 추론 백엔드 — ONNX 모델을 [`tract`](https://github.com/sonos/tract)로 실행한다.
//! C++ ONNX Runtime 없이 단일 바이너리·클린 abi3 휠을 유지한다.
//!
//! [`TractModel`]이 고정 입력 형상의 ONNX를 적재해 1회 추론을 돌리는 공용 러너이고, 그 위에
//! 작업별 백엔드(텍스트 검출/인식, 분류, 객체 검출)가 [`crate::tasks`]의 trait 을 구현한다.
//! 전처리는 [`crate::preprocess`], 후처리는 [`crate::postprocess`]의 재사용 함수를 쓴다.
//!
//! ## 검증 경계 (정직한 고지)
//! 텍스트 검출/인식(OCR)은 **실제 PP-OCRv5 모델(한·영·일 등)로 동작 검증됨** — 실 화면 사진에서
//! 검출·인식·DB unclip·XY-Cut 읽기순서·자동 방향 보정([`recognize_image_auto`])까지 확인했다.
//! 분류([`TractClassifier`])와 객체 검출([`TractObjectDetector`])은 컴파일·구조는 갖췄으나 **실모델
//! 정확도는 미검증**이다 — 특히 객체 검출은 **출력 레이아웃을 `[N,6]`으로 가정**해 모델에 따라
//! 디코더 교체가 필요할 수 있다. 검출 박스는 축 정렬이다(개별 줄의 회전은 미적용).

use std::path::Path;

use tract_onnx::pb;
use tract_onnx::prelude::*;

use crate::engine::OcrEngine;
use crate::error::{Result, VisionError};
use crate::preprocess;
use crate::postprocess;
use crate::tasks::{Classifier, ObjectDetector, TextDetector, TextRecognizer};
use crate::types::{BBox, Classification, Crop, Detection, Frame, Recognized, TextBox};

/// 빌드된 실행 가능 모델(고정 입력 형상).
type Runnable = TypedRunnableModel<TypedModel>;

/// 공용 ONNX 러너 — 고정 입력 [1,3,h,w]로 모델을 적재하고 1회 추론한다.
/// 모든 작업 백엔드가 이 위에 올라가므로, 새 작업은 전/후처리만 추가하면 된다.
pub struct TractModel {
    model: Runnable,
    in_h: usize,
    in_w: usize,
}

impl TractModel {
    /// 고정 입력 형상으로 ONNX 모델 적재 → 최적화 → runnable.
    ///
    /// PaddleOCR 같은 실모델은 입력이 동적 차원(N·H·W 가변)이라 tract 가 그대로는 파싱하지
    /// 못한다(심볼릭 차원 거부). 그래서 원시 proto 를 받아 입력 0 의 차원을 정적
    /// [1,3,in_h,in_w] 으로 못 박은 뒤 파싱한다.
    pub fn load(model_path: impl AsRef<Path>, in_h: usize, in_w: usize) -> Result<Self> {
        let p = model_path.as_ref();
        // 출력/중간 텐서의 심볼릭 형상은 무시하고 tract 가 직접 추론하게 한다
        // (PaddleOCR 의 동적 차원이 출력에도 남아 있어 파싱이 깨지므로).
        let onnx = tract_onnx::onnx()
            .with_ignore_output_shapes(true)
            .with_ignore_output_types(true);
        let mut proto = onnx
            .proto_model_for_path(p)
            .map_err(|e| VisionError::backend(format!("load {}: {e}", p.display())))?;
        fix_input_shape(&mut proto, in_h, in_w);
        let model = onnx
            .parse(&proto, None)
            .map_err(|e| VisionError::backend(format!("parse {}: {e}", p.display())))?
            .model
            .into_optimized()
            .map_err(|e| VisionError::backend(format!("optimize: {e}")))?
            .into_runnable()
            .map_err(|e| VisionError::backend(format!("runnable: {e}")))?;
        Ok(Self { model, in_h, in_w })
    }

    /// 입력 (높이, 너비).
    pub fn dims(&self) -> (usize, usize) {
        (self.in_h, self.in_w)
    }

    /// [1,3,h,w] f32 텐서로 1회 추론 → (출력 슬라이스, 출력 형상).
    pub fn run(&self, data: Vec<f32>) -> Result<(Vec<f32>, Vec<usize>)> {
        let input = Tensor::from_shape(&[1, 3, self.in_h, self.in_w], &data)
            .map_err(|e| VisionError::backend(format!("tensor from_shape: {e}")))?;
        let out = self
            .model
            .run(tvec!(input.into()))
            .map_err(|e| VisionError::backend(format!("inference: {e}")))?;
        let t = &out[0];
        let shape = t.shape().to_vec();
        let slice = t
            .as_slice::<f32>()
            .map_err(|e| VisionError::backend(format!("output as f32: {e}")))?
            .to_vec();
        Ok((slice, shape))
    }
}

// ─────────────────────────────────────────────────────────────────
// 텍스트 검출 (DBNet 계열)
// ─────────────────────────────────────────────────────────────────

pub struct TractTextDetector {
    model: TractModel,
    bin_threshold: f32,
    min_area: usize,
    /// DB unclip 비율 — 수축된 검출 박스를 글자 전체가 들어오도록 바깥으로 팽창.
    /// PaddleOCR 의 det_db_unclip_ratio 에 해당(1.0 = 미적용, 권장 1.5).
    unclip_ratio: f32,
    /// XY-Cut 읽기순서의 최소 분할 간격(px). 단/행을 가르는 빈 픽셀 임계값.
    xycut_gap: usize,
}

impl TractTextDetector {
    /// 검출 모델 적재. 입력은 letterbox 로 (h,w)에 맞춘다(예: 736x1280).
    pub fn new(model_path: impl AsRef<Path>, (h, w): (usize, usize)) -> Result<Self> {
        Ok(Self {
            model: TractModel::load(model_path, h, w)?,
            bin_threshold: 0.3,
            min_area: 16,
            unclip_ratio: 1.5,
            xycut_gap: 8,
        })
    }

    pub fn with_threshold(mut self, bin_threshold: f32) -> Self {
        self.bin_threshold = bin_threshold;
        self
    }

    /// DB unclip 비율 설정(1.0 = 팽창 없음). DB 검출은 글자보다 수축된 영역을 예측하므로,
    /// 크롭 전에 박스를 되팽창시켜야 글자 가장자리(특히 첫 글자)가 잘리지 않는다.
    pub fn with_unclip(mut self, ratio: f32) -> Self {
        self.unclip_ratio = ratio.max(1.0);
        self
    }

    /// XY-Cut 읽기순서의 최소 분할 간격(px) 설정. 작을수록 잘게 가른다(자간보다 크게).
    pub fn with_xycut_gap(mut self, gap: usize) -> Self {
        self.xycut_gap = gap.max(1);
        self
    }
}

impl TextDetector for TractTextDetector {
    fn detect(&self, frame: &Frame) -> Result<Vec<TextBox>> {
        let (h, w) = self.model.dims();
        let (data, scale) =
            preprocess::letterbox_chw(&frame.image, h, w, preprocess::IMAGENET_MEAN, preprocess::IMAGENET_STD);
        let (prob, shape) = self.model.run(data)?;

        let (ph, pw) = match shape.len() {
            n if n >= 2 => (shape[n - 2], shape[n - 1]),
            _ => return Err(VisionError::backend(format!("unexpected det output shape {shape:?}"))),
        };
        if prob.len() < ph * pw {
            return Err(VisionError::backend("det output smaller than HxW"));
        }

        let boxes = postprocess::connected_boxes(&prob[..ph * pw], pw, ph, self.bin_threshold, self.min_area);
        let map_x = self.model.dims().1 as f32 / pw as f32;
        let map_y = self.model.dims().0 as f32 / ph as f32;
        let (iw, ih) = (frame.image.width() as f32, frame.image.height() as f32);
        let ratio = self.unclip_ratio;
        // 각 박스를 (타이트 원본박스, unclip 팽창 TextBox) 쌍으로. 읽기순서는 타이트 박스로,
        // 크롭·출력은 팽창 박스로 한다(팽창 박스는 줄끼리 Y 로 겹쳐 XY-Cut 투영 분할을 망침).
        let mapped: Vec<(BBox, TextBox)> = boxes
            .into_iter()
            .map(|(bx, by, bw, bh, score)| {
                let x = bx as f32 * map_x / scale;
                let y = by as f32 * map_y / scale;
                let w = (bw as f32 * map_x / scale).max(1.0);
                let h = (bh as f32 * map_y / scale).max(1.0);
                let tight = BBox::new(x as u32, y as u32, w as u32, h as u32);
                // DB unclip(DBNet): 수축된 박스를 오프셋 d = area*ratio/perimeter 만큼 바깥으로
                // 팽창(PaddleOCR 와 동일). 축 정렬 박스에 근사 적용하고 이미지 경계로 자른다.
                let d = if ratio > 1.0 { (w * h) * ratio / (2.0 * (w + h)) } else { 0.0 };
                let x0 = (x - d).max(0.0);
                let y0 = (y - d).max(0.0);
                let x1 = (x + w + d).min(iw);
                let y1 = (y + h + d).min(ih);
                let unclipped =
                    BBox::new(x0 as u32, y0 as u32, (x1 - x0).max(1.0) as u32, (y1 - y0).max(1.0) as u32);
                (tight, TextBox { bbox: unclipped, confidence: score })
            })
            .collect();
        // 연결요소 추출은 래스터 스캔 순서라 단어가 뒤섞인다. XY-Cut 으로 읽기순서(다단·표·
        // 나란한 패널까지)를 잡아 인식 결과가 페이지 순서대로 나오게 한다.
        let tights: Vec<BBox> = mapped.iter().map(|(t, _)| *t).collect();
        Ok(postprocess::xy_cut_order(&tights, self.xycut_gap)
            .into_iter()
            .map(|i| mapped[i].1)
            .collect())
    }
}

// ─────────────────────────────────────────────────────────────────
// 텍스트 인식 (CRNN/SVTR + CTC)
// ─────────────────────────────────────────────────────────────────

pub struct TractTextRecognizer {
    model: TractModel,
    dict: Vec<String>,
}

impl TractTextRecognizer {
    /// 인식 모델 + 문자 사전 적재. 입력 높이 48, 폭 320 권장(PP-OCRv5).
    pub fn new(
        model_path: impl AsRef<Path>,
        dict_path: impl AsRef<Path>,
        (h, w): (usize, usize),
    ) -> Result<Self> {
        let dict = load_lines(dict_path.as_ref())?;
        Ok(Self { model: TractModel::load(model_path, h, w)?, dict })
    }
}

impl TextRecognizer for TractTextRecognizer {
    fn recognize(&self, crops: &[Crop]) -> Result<Vec<Recognized>> {
        let (h, w) = self.model.dims();
        let mut out = Vec::with_capacity(crops.len());
        for crop in crops {
            let data = preprocess::fixed_height_chw(&crop.image, h, w);
            let (logits, shape) = self.model.run(data)?;
            let (t, c) = match shape.len() {
                n if n >= 2 => (shape[n - 2], shape[n - 1]),
                _ => return Err(VisionError::backend(format!("unexpected rec output shape {shape:?}"))),
            };
            let (text, conf) = postprocess::ctc_greedy_decode(&logits, t, c, &self.dict);
            out.push(Recognized { text, confidence: conf, bbox: crop.bbox });
        }
        Ok(out)
    }
}

// ─────────────────────────────────────────────────────────────────
// 이미지 분류 (출력 [1, C] → softmax → top-k)
// ─────────────────────────────────────────────────────────────────

pub struct TractClassifier {
    model: TractModel,
    labels: Vec<String>,
    top_k: usize,
}

impl TractClassifier {
    /// 분류 모델 + 라벨 목록 적재(라벨 1줄 = 클래스 1개, 인덱스 순서).
    pub fn new(
        model_path: impl AsRef<Path>,
        labels_path: impl AsRef<Path>,
        (h, w): (usize, usize),
    ) -> Result<Self> {
        let labels = load_lines(labels_path.as_ref())?;
        Ok(Self { model: TractModel::load(model_path, h, w)?, labels, top_k: 5 })
    }

    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = k.max(1);
        self
    }
}

impl Classifier for TractClassifier {
    fn classify(&self, frame: &Frame) -> Result<Vec<Classification>> {
        let (h, w) = self.model.dims();
        let (data, _) =
            preprocess::letterbox_chw(&frame.image, h, w, preprocess::IMAGENET_MEAN, preprocess::IMAGENET_STD);
        let (logits, _) = self.model.run(data)?;
        let probs = postprocess::softmax(&logits);

        let mut idx: Vec<usize> = (0..probs.len()).collect();
        idx.sort_by(|&i, &j| probs[j].partial_cmp(&probs[i]).unwrap_or(std::cmp::Ordering::Equal));
        Ok(idx
            .into_iter()
            .take(self.top_k)
            .map(|i| Classification {
                label: self.labels.get(i).cloned().unwrap_or_else(|| format!("class_{i}")),
                score: probs[i],
            })
            .collect())
    }
}

// ─────────────────────────────────────────────────────────────────
// 문서 방향 분류 (PP-LCNet doc-ori → 0/90/180/270 보정)
// ─────────────────────────────────────────────────────────────────

/// 문서 이미지 방향(0/90/180/270°) 분류 — 회전된 사진/스캔을 OCR 전에 바로 세운다.
/// PP-LCNet doc-ori 모델(출력 [1,4], 클래스 순서 0/90/180/270). 입력 224x224 권장.
/// 폰을 옆으로 들고 찍은 화면 사진처럼 통째로 회전된 입력을 정상화한다.
pub struct TractDocOrientation {
    model: TractModel,
}

impl TractDocOrientation {
    /// 방향 분류 모델 적재(입력 224x224 권장).
    pub fn new(model_path: impl AsRef<Path>, (h, w): (usize, usize)) -> Result<Self> {
        Ok(Self { model: TractModel::load(model_path, h, w)? })
    }

    /// 추정 회전 각도(0/90/180/270) — 이미지가 시계방향으로 그만큼 돌아가 있다는 의미.
    pub fn predict(&self, frame: &Frame) -> Result<u32> {
        Ok(self.predict_conf(frame)?.0)
    }

    /// 추정 각도 + 분류기 신뢰도(softmax 최대 확률, 0.0-1.0). 화면 사진처럼 학습 분포 밖
    /// 입력에선 오분류가 잦아, 신뢰도로 게이팅해 저신뢰면 회전을 보류하는 데 쓴다.
    pub fn predict_conf(&self, frame: &Frame) -> Result<(u32, f32)> {
        let (h, w) = self.model.dims();
        let data =
            preprocess::resize_chw(&frame.image, h, w, preprocess::IMAGENET_MEAN, preprocess::IMAGENET_STD);
        let (logits, _) = self.model.run(data)?;
        let probs = postprocess::softmax(&logits);
        let (idx, conf) = postprocess::argmax(&probs);
        Ok(([0u32, 90, 180, 270][idx.min(3)], conf))
    }

    /// 방향을 보정한 새 Frame(똑바로 세움). 예측 0°면 원본 그대로.
    /// 시계방향 deg 만큼 돌아간 것을 되돌리므로 반대 방향으로 회전한다.
    /// `min_conf` 이상일 때만 회전 — 저신뢰 오분류로 멀쩡한 입력을 뒤집는 것을 막는다.
    pub fn correct(&self, frame: &Frame, min_conf: f32) -> Result<Frame> {
        let (deg, conf) = self.predict_conf(frame)?;
        let deg = if conf >= min_conf { deg } else { 0 };
        let img = match deg {
            90 => frame.image.rotate270(),
            180 => frame.image.rotate180(),
            270 => frame.image.rotate90(),
            _ => frame.image.clone(),
        };
        Ok(Frame::new(img, frame.index, frame.timestamp))
    }
}

// ─────────────────────────────────────────────────────────────────
// 객체 검출 (출력 레이아웃 가정 + NMS)
// ─────────────────────────────────────────────────────────────────

/// 출력 텐서가 [N, 6] = (x1, y1, x2, y2, score, class_id) 평탄화라고 **가정**하는 범용
/// 객체 검출 백엔드. (YOLO 변형마다 출력 레이아웃이 달라, 이 가정은 실제 모델로 검증해야
/// 한다 — 다른 레이아웃이면 이 디코더를 그 모델에 맞춰 교체한다.)
pub struct TractObjectDetector {
    model: TractModel,
    labels: Vec<String>,
    score_threshold: f32,
    iou_threshold: f32,
}

impl TractObjectDetector {
    pub fn new(
        model_path: impl AsRef<Path>,
        labels_path: impl AsRef<Path>,
        (h, w): (usize, usize),
    ) -> Result<Self> {
        let labels = load_lines(labels_path.as_ref())?;
        Ok(Self {
            model: TractModel::load(model_path, h, w)?,
            labels,
            score_threshold: 0.25,
            iou_threshold: 0.45,
        })
    }

    pub fn with_thresholds(mut self, score: f32, iou: f32) -> Self {
        self.score_threshold = score;
        self.iou_threshold = iou;
        self
    }
}

impl ObjectDetector for TractObjectDetector {
    fn detect_objects(&self, frame: &Frame) -> Result<Vec<Detection>> {
        let (h, w) = self.model.dims();
        let (data, scale) =
            preprocess::letterbox_chw(&frame.image, h, w, preprocess::IMAGENET_MEAN, preprocess::IMAGENET_STD);
        let (out, shape) = self.model.run(data)?;

        // [N, 6] 가정: 마지막 축이 6(x1,y1,x2,y2,score,cls).
        let stride = *shape.last().unwrap_or(&0);
        if stride < 6 || out.is_empty() {
            return Err(VisionError::backend(format!(
                "object detector expects [N,>=6] output, got shape {shape:?}"
            )));
        }
        let n = out.len() / stride;

        let mut candidates: Vec<(BBox, f32, usize)> = Vec::new();
        for i in 0..n {
            let row = &out[i * stride..i * stride + stride];
            let score = row[4];
            if score < self.score_threshold {
                continue;
            }
            // letterbox 역매핑(좌상단 배치 → scale 만 보정).
            let x1 = (row[0] / scale).max(0.0);
            let y1 = (row[1] / scale).max(0.0);
            let x2 = (row[2] / scale).max(0.0);
            let y2 = (row[3] / scale).max(0.0);
            let bbox = BBox::new(
                x1 as u32,
                y1 as u32,
                (x2 - x1).max(1.0) as u32,
                (y2 - y1).max(1.0) as u32,
            );
            candidates.push((bbox, score, row[5] as usize));
        }

        let boxed: Vec<(BBox, f32)> = candidates.iter().map(|(b, s, _)| (*b, *s)).collect();
        let keep = postprocess::nms(&boxed, self.iou_threshold);
        Ok(keep
            .into_iter()
            .map(|i| {
                let (bbox, score, cls) = &candidates[i];
                Detection {
                    bbox: *bbox,
                    label: self.labels.get(*cls).cloned().unwrap_or_else(|| format!("class_{cls}")),
                    score: *score,
                }
            })
            .collect())
    }
}

// ── 공용 로더 ─────────────────────────────────────────────────────

/// 입력 0 의 (동적) 형상을 정적 [1,3,in_h,in_w] 으로 덮어쓴다. tract 가 PaddleOCR 의
/// 심볼릭 차원(DynamicDimension)을 거부하므로, 파싱 전에 차원을 못 박는다.
fn fix_input_shape(proto: &mut pb::ModelProto, in_h: usize, in_w: usize) {
    use pb::tensor_shape_proto::dimension::Value as DimVal;
    use pb::type_proto::Value as TyVal;
    let dims = [1i64, 3, in_h as i64, in_w as i64];
    let debug = std::env::var("ROCT_DEBUG_ONNX").is_ok();

    if let Some(graph) = proto.graph.as_mut() {
        // 모든 입력 중 "동적 차원을 가진 rank-4" 를 이미지 입력으로 보고 [1,3,h,w] 로 고정.
        for inp in graph.input.iter_mut() {
            if let Some(TyVal::TensorType(t)) = inp.r#type.as_mut().and_then(|ty| ty.value.as_mut()) {
                if let Some(shape) = t.shape.as_mut() {
                    let dynamic = shape
                        .dim
                        .iter()
                        .any(|d| !matches!(d.value, Some(DimVal::DimValue(_))));
                    if debug {
                        eprintln!("[onnx] input '{}' rank={} dynamic={}", inp.name, shape.dim.len(), dynamic);
                    }
                    if shape.dim.len() == 4 && dynamic {
                        for (i, d) in shape.dim.iter_mut().enumerate() {
                            if let Some(v) = dims.get(i) {
                                d.value = Some(DimVal::DimValue(*v));
                            }
                        }
                    }
                }
            }
        }
        // 출력·중간 텐서의 (심볼릭) 형상 제거 → tract 가 직접 추론.
        for vi in graph.output.iter_mut().chain(graph.value_info.iter_mut()) {
            if let Some(TyVal::TensorType(t)) = vi.r#type.as_mut().and_then(|ty| ty.value.as_mut()) {
                t.shape = None;
            }
        }
    }
}

/// 사전/라벨 파일 로드 — 한 줄당 한 항목. 인덱스 보존을 위해 빈 줄도 유지하되 마지막
/// 개행만 제거한다.
fn load_lines(path: &Path) -> Result<Vec<String>> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| VisionError::backend(format!("read {}: {e}", path.display())))?;
    let mut lines: Vec<String> = raw.split('\n').map(|s| s.trim_end_matches('\r').to_string()).collect();
    if lines.last().map(|s| s.is_empty()).unwrap_or(false) {
        lines.pop();
    }
    Ok(lines)
}

// ── 고수준 이미지 OCR 파이프라인 (CLI·Python 바인딩 공용) ────────────

fn round32(n: usize) -> usize {
    (((n + 16) / 32) * 32).max(32)
}

/// 프레임 크기에 비례한 검출 입력 (h, w) — 긴 변을 det_long 에 맞추고 32 배수로(DBNet 1/32).
fn det_shape(frame: &Frame, det_long: usize) -> (usize, usize) {
    let (iw, ih) = (frame.image.width() as f32, frame.image.height() as f32);
    let long = iw.max(ih).max(1.0);
    let s = det_long as f32 / long;
    (round32((ih * s) as usize), round32((iw * s) as usize))
}

/// 인식 결과 중 비어있지 않고 고신뢰(>=0.8)인 영역 수 — 방향 판정 척도(거꾸로/옆이면 바닥).
fn orient_score(rs: &[Recognized]) -> usize {
    rs.iter().filter(|r| !r.text.trim().is_empty() && r.confidence >= 0.8).count()
}

/// 고수준 이미지 OCR — 검출·인식 모델·사전 경로와 옵션으로 인식 결과를 돌려준다(검출 + 인식
/// 합성). DB unclip 과 XY-Cut 읽기순서는 기본 적용된다.
///
/// `auto_rotate` 가 참이면 0/90/180/270° 중 OCR 신뢰도(고신뢰 영역 수)가 가장 높은 방향을 자동
/// 선택한다 — 폰을 옆으로 들고 찍은 화면 사진처럼 통째 회전된 입력에 견고하다(회전 분류기는
/// 방향을 자주 틀려, 분류기 대신 인식 점수를 척도로 쓴다). 똑바로 선 입력은 0° 한 번만 보고
/// 빠르게 통과하고, 0° 가 잘 안 읽힐 때만 네 방향을 모두 시도한다.
///
/// `det_long` 은 검출 입력의 긴 변 목표 px(입력 크기에 비례, 32 배수). 고해상 사진을 작은
/// 고정 입력에 욱여넣어 글자가 뭉개지는 것을 막는다. 검출·인식 모델은 호출마다 새로 적재한다.
pub fn recognize_image_auto(
    frame: &Frame,
    det_model: impl AsRef<Path>,
    rec_model: impl AsRef<Path>,
    dict: impl AsRef<Path>,
    auto_rotate: bool,
    det_long: usize,
) -> Result<Vec<Recognized>> {
    let (det_model, rec_model, dict) = (det_model.as_ref(), rec_model.as_ref(), dict.as_ref());
    let build = |f: &Frame| -> Result<OcrEngine<TractTextDetector, TractTextRecognizer>> {
        let (h, w) = det_shape(f, det_long);
        Ok(OcrEngine::new(
            TractTextDetector::new(det_model, (h, w))?,
            TractTextRecognizer::new(rec_model, dict, (48, 320))?,
        ))
    };

    let eng = build(frame)?; // 0°/180° 형상
    let r0 = eng.read(frame)?;
    // 방향 보정 안 함, 또는 0° 가 충분히 잘 읽히면(흔한 경우) 그대로 채택.
    if !auto_rotate || orient_score(&r0) >= 8 {
        return Ok(r0);
    }
    // 0° 가 부실하면 90/180/270 까지 인식해 가장 잘 읽히는 방향을 고른다.
    let rot = |d: u32| -> Frame {
        let img = match d {
            90 => frame.image.rotate90(),
            180 => frame.image.rotate180(),
            270 => frame.image.rotate270(),
            _ => frame.image.clone(),
        };
        Frame::new(img, frame.index, frame.timestamp)
    };
    let f180 = rot(180);
    let r180 = eng.read(&f180)?; // 0/180 은 같은 형상 → 같은 엔진 재사용
    let f90 = rot(90);
    let f270 = rot(270);
    let eng_p = build(&f90)?; // 90/270 은 회전 형상 → 새 엔진(1 회만 적재)
    let r90 = eng_p.read(&f90)?;
    let r270 = eng_p.read(&f270)?;
    // 0° 우선(동점 시 유지), 나머지 중 고신뢰 영역이 가장 많은 방향 선택.
    let mut best = (orient_score(&r0), r0);
    for r in [r90, r180, r270] {
        let s = orient_score(&r);
        if s > best.0 {
            best = (s, r);
        }
    }
    Ok(best.1)
}
