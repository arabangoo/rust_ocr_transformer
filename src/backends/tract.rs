//! 순수 Rust 추론 백엔드 — ONNX 모델을 [`tract`](https://github.com/sonos/tract)로 실행한다.
//! C++ ONNX Runtime 없이 단일 바이너리·클린 abi3 휠을 유지한다.
//!
//! [`TractModel`]이 고정 입력 형상의 ONNX를 적재해 1회 추론을 돌리는 공용 러너이고, 그 위에
//! 작업별 백엔드(텍스트 검출/인식, 분류, 객체 검출)가 [`crate::tasks`]의 trait 을 구현한다.
//! 전처리는 [`crate::preprocess`], 후처리는 [`crate::postprocess`]의 재사용 함수를 쓴다.
//!
//! ## 검증 경계 (정직한 고지)
//! 이 모듈은 **컴파일까지 검증**됐고, 전처리·후처리 로직은 단위 테스트로 확인됐다. 다만
//! **실제 모델 파일로 추론 정확도를 확인하지는 못했다.** 첫 실모델 실행 후 출력 텐서 축
//! 순서, 사전/라벨 인덱스 오프셋, 임계값, 그리고 특히 객체 검출의 **출력 레이아웃 가정**이
//! 조정될 수 있다. 검출 박스는 축 정렬(회전·unclip 미적용)이다.

use std::path::Path;

use tract_onnx::pb;
use tract_onnx::prelude::*;

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
}

impl TractTextDetector {
    /// 검출 모델 적재. 입력은 letterbox 로 (h,w)에 맞춘다(예: 736x1280).
    pub fn new(model_path: impl AsRef<Path>, (h, w): (usize, usize)) -> Result<Self> {
        Ok(Self { model: TractModel::load(model_path, h, w)?, bin_threshold: 0.3, min_area: 16 })
    }

    pub fn with_threshold(mut self, bin_threshold: f32) -> Self {
        self.bin_threshold = bin_threshold;
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
        Ok(boxes
            .into_iter()
            .map(|(bx, by, bw, bh, score)| {
                let ox = (bx as f32 * map_x / scale) as u32;
                let oy = (by as f32 * map_y / scale) as u32;
                let ow = (bw as f32 * map_x / scale).max(1.0) as u32;
                let oh = (bh as f32 * map_y / scale).max(1.0) as u32;
                TextBox { bbox: BBox::new(ox, oy, ow, oh), confidence: score }
            })
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
