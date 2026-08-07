# rust_ocr_transformer

> **A unified vision inference and processing framework, written in Rust.**
>
> It streams images and video through `decode -> preprocess -> neural-network inference -> postprocess -> structured output`.
> It is an orchestration layer where per-task models (text detection/recognition, classification, object detection, layout, segmentation) plug in behind traits and swap out freely. The default inference runs on **pure Rust (`tract`), zero FFI, CPU**.

This document is the **developer manual** for the library. It covers the design principles, the public API, per-task behavior and maturity, CLI and Python usage, model preparation, service integration, how to add a new task or backend, and the build and test workflow.

**Key references**

1. DBNet: Real-time Scene Text Detection with Differentiable Binarization (text detection + DB post-processing) - https://arxiv.org/abs/1911.08947
2. CRNN: An End-to-End Trainable Neural Network for Image-based Sequence Recognition (CNN+RNN+CTC sequence recognition) - https://arxiv.org/abs/1507.05717
3. SVTR: Scene Text Recognition with a Single Visual Model (PP-OCRv5 recognition backbone) - https://arxiv.org/abs/2205.00159
4. PP-OCR: A Practical Ultra Lightweight OCR System (the lightweight OCR model family embedded here) - https://arxiv.org/abs/2009.09941
5. Image Quality Assessment: From Error Visibility to Structural Similarity (SSIM, the video frame-sampling gate) - https://ece.uwaterloo.ca/~z70wang/research/ssim/

---

## Table of Contents

1. [Key Features](#1-key-features)
2. [Quick Start](#2-quick-start)
3. [Installation and Cargo Features](#3-installation-and-cargo-features)
4. [Architecture](#4-architecture)
5. [Common Type Reference](#5-common-type-reference)
6. [Public API Reference](#6-public-api-reference)
7. [Per-Task Behavior and Maturity](#7-per-task-behavior-and-maturity)
8. [Video Timeline Processing](#8-video-timeline-processing)
9. [CLI Tool (`roct`)](#9-cli-tool-roct)
10. [Python Bindings (PyO3)](#10-python-bindings-pyo3)
11. [Model Preparation (ONNX Models and Dictionaries)](#11-model-preparation-onnx-models-and-dictionaries)
12. [Embedding into a Service Pipeline](#12-embedding-into-a-service-pipeline)
13. [Adding a New Task or Backend](#13-adding-a-new-task-or-backend)
14. [Build, Feature Combinations, and Testing](#14-build-feature-combinations-and-testing)
15. [Directory Layout](#15-directory-layout)
16. [License](#16-license)

---

## 1. Key Features

The underrated part of a vision pipeline is **pre/post-processing and orchestration**. No matter how good the model is, results collapse when input normalization, detection post-processing, or decoding conventions are off. This library is not a single do-everything model; it aims to be the **framework** that owns the systems engineering above and below the model.

| Principle | What it means |
|---|---|
| **Plug the model in, don't bake it in** | The "intelligence" belongs to the model behind the trait. The framework handles decode, preprocessing, post-processing, and assembly. Same input, same model, same result (deterministic). |
| **Task-level abstraction** | Text detection/recognition, classification, object detection, layout, and segmentation are each separated behind their own trait. A new model plugs in by implementing a single trait. |
| **Pure Rust by default, zero FFI** | The default inference engine is `tract` (a pure-Rust ONNX runtime). No C++ ONNX Runtime, CUDA, or subprocess. A CPU single binary and a clean abi3 wheel. When you need speed or GPU, add `ort` as an opt-in. |
| **Video as a first-class citizen** | A still image is the degenerate case of a video with one frame. The same pipeline takes both, and video adds SSIM sampling and timeline merging. |

### "Understanding" lives outside the core

What this framework produces is **structured vision results**, up to "what is where" (text, boxes, classes, coordinates). **Understanding and reasoning**, such as "interpret this receipt as JSON" or "assess the damaged area in this accident photo", belong to large VLMs/LLMs. That work is delegated outside the core (to a server, for example), and this framework produces its input. For pure character recognition, small specialized models are more accurate, deterministic, and cheaper than frontier LLMs, so the core is kept LLM-free.

---

## 2. Quick Start

### Rust library (OCR)

```rust
use rust_ocr_transformer::{Frame, OcrEngine, TractTextDetector, TractTextRecognizer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load a single image (for video, feed a frame stream into the same engine)
    let frame = Frame::from_path("page.png")?;

    // detector (input 736x1280) + recognizer (input 48x320) + character dictionary
    let detector = TractTextDetector::new("models/det.onnx", (736, 1280))?;
    let recognizer = TractTextRecognizer::new("models/rec.onnx", "models/dict.txt", (48, 320))?;
    let engine = OcrEngine::new(detector, recognizer);

    // detect -> crop -> recognize
    for r in engine.read(&frame)? {
        println!("[{:.2}] ({},{}) {}", r.confidence, r.bbox.x, r.bbox.y, r.text);
    }
    Ok(())
}
```

See [section 11](#11-model-preparation-onnx-models-and-dictionaries) for how to obtain the model and dictionary files.

### CLI

```bash
cargo build --release --features cli
./target/release/roct image page.png \
  --det-model models/det.onnx --rec-model models/rec.onnx --dict models/dict.txt
```

### Python

```python
import json
import rust_ocr_transformer as roct

# Image OCR: pass model/dict paths, get recognition results as JSON
out = roct.recognize_image("page.png", "models/det.onnx", "models/rec.onnx", "models/dict.txt")
for r in json.loads(out):
    print(r["confidence"], r["text"])
```

---

## 3. Installation and Cargo Features

`Cargo.toml`:

```toml
[dependencies]
rust_ocr_transformer = { git = "https://github.com/arabangoo/rust_ocr_transformer", tag = "v0.2.2" }
```

### Feature list

| Feature | Enables | Notes |
|---|---|---|
| **`tract`** | pure-Rust ONNX inference backend | on by default. No C++ FFI. `tract-onnx` |
| `video` | video decode (Phase 2) | `ffmpeg-next`. Placeholder only for now, not yet implemented |
| **`cli`** | the `roct` binary | `clap`. An opt-in that does not leak to library consumers. |
| **`python`** | PyO3 cdylib bindings | `pyo3` (abi3) |

```toml
# default = ["tract"]  # pure-Rust recognition backend, zero FFI

# core IP only (types, traits, sampling, merging, pre/post-processing); inject the backend yourself
rust_ocr_transformer = { version = "0.1", default-features = false }
```

> The default (`tract`) build requires no external `.so/.dll` and no subprocess. If speed or GPU becomes the bottleneck, the plan is to add an `ort` (ONNX Runtime, C++ FFI) backend behind the same trait as an opt-in feature (not yet implemented; we do not create dead features up front).

---

## 4. Architecture

```text
decode -> preprocess -> task model (trait) -> postprocess -> structured output
 Frame    preprocess    tasks::*             postprocess    FrameAnalysis
                        (model plug point)                  + SRT/JSON/plaintext
```

The heart of it is the **task-trait layer**. Each task (detection, recognition, classification, object detection, layout, segmentation) takes a common [`Frame`](#5-common-type-reference) as input and produces a per-task result type. The model enters as an implementation of that trait, so the core pipeline assembles purely at the task level without knowing which model is used.

- **Decode**: an image/video becomes a [`Frame`](#5-common-type-reference). An image is a one-frame video.
- **Preprocess**: [`preprocess`](#63-preprocessing-preprocess): letterbox resize, normalization, NCHW tensorization. Reused across tasks.
- **Inference**: the model behind a [`tasks`](#61-task-traits) trait ([`backends::tract`](#133-the-tract-backend-reference)).
- **Postprocess**: [`postprocess`](#64-postprocessing-postprocess): non-maximum suppression (NMS), softmax, connectionist temporal classification (CTC) decoding, DB boxes.
- **Output**: [`FrameAnalysis`](#5-common-type-reference) gathers results from multiple tasks, and video is emitted as [SRT/JSON](#8-video-timeline-processing).

---

## 5. Common Type Reference

The `types` module. Result types implement `serde::{Serialize, Deserialize}`, so they can be dumped straight to JSON.

```rust
/// One pipeline unit. An image is the degenerate case of a video with a single frame.
pub struct Frame {
    pub image:     image::DynamicImage,
    pub index:     u64,        // frame index within the video (0 for an image)
    pub timestamp: Timestamp,  // presentation time in ms (0 for an image)
}
// Frame::from_path(p) / Frame::from_image(img) / Frame::new(img, index, ts)

/// Rectangular region in pixel coordinates (origin at top-left).
pub struct BBox { pub x: u32, pub y: u32, pub width: u32, pub height: u32 }
```

### Per-task result types

```rust
pub struct TextBox     { pub bbox: BBox, pub confidence: f32 }                  // detection
pub struct Crop        { pub image: image::DynamicImage, pub bbox: BBox }       // between detection and recognition
pub struct Recognized  { pub text: String, pub confidence: f32, pub bbox: BBox }// recognition
pub struct Detection   { pub bbox: BBox, pub label: String, pub score: f32 }    // object detection
pub struct Classification { pub label: String, pub score: f32 }                 // classification
pub struct LayoutRegion { pub bbox: BBox, pub kind: String, pub score: f32 }    // layout
pub struct Mask        { pub width: u32, pub height: u32, pub classes: Vec<u8> }// segmentation

/// Combined analysis of one frame: a structured-output container that gathers multiple task outputs.
pub struct FrameAnalysis {
    pub recognized:      Vec<Recognized>,
    pub detections:      Vec<Detection>,
    pub classifications: Vec<Classification>,
    pub layout:          Vec<LayoutRegion>,
}
```

### Video timeline types

```rust
pub struct Timestamp(pub u64);   // milliseconds. to_srt() -> "HH:MM:SS,mmm"
pub struct Segment { pub start: Timestamp, pub end: Timestamp, pub text: String }
```

---

## 6. Public API Reference

### 6.1 Task Traits

The `tasks` module. This is where models plug in. All are `Send + Sync` (shareable across threads).

```rust
pub trait TextDetector:   Send + Sync { fn detect(&self, frame: &Frame) -> Result<Vec<TextBox>>; }
pub trait TextRecognizer: Send + Sync { fn recognize(&self, crops: &[Crop]) -> Result<Vec<Recognized>>; }
pub trait ObjectDetector: Send + Sync { fn detect_objects(&self, frame: &Frame) -> Result<Vec<Detection>>; }
pub trait Classifier:     Send + Sync { fn classify(&self, frame: &Frame) -> Result<Vec<Classification>>; }
pub trait LayoutAnalyzer: Send + Sync { fn analyze_layout(&self, frame: &Frame) -> Result<Vec<LayoutRegion>>; }
pub trait Segmenter:      Send + Sync { fn segment(&self, frame: &Frame) -> Result<Mask>; }
```

### 6.2 OcrEngine: composing the OCR task

An OCR pipeline that composes a detector and a recognizer. Other tasks are assembled directly through their own traits.

```rust
OcrEngine::new(detector: D, recognizer: R) -> OcrEngine<D, R>   // D: TextDetector, R: TextRecognizer
fn read(&self, frame: &Frame) -> Result<Vec<Recognized>>        // detect -> crop -> recognize
fn detector(&self) -> &D
fn recognizer(&self) -> &R

// helper: crop the frame by detection boxes into a list of Crops
crop_regions(frame: &Frame, boxes: &[TextBox]) -> Vec<Crop>
```

### 6.3 Preprocessing (`preprocess`)

Task-agnostic reusable functions that turn an image into a model input tensor (NCHW f32).

```rust
pub const IMAGENET_MEAN: [f32; 3];  // [0.485, 0.456, 0.406]
pub const IMAGENET_STD:  [f32; 3];  // [0.229, 0.224, 0.225]

// letterbox (keep aspect ratio + pad) + per-channel normalization -> (data[3*h*w], scale). scale maps boxes back.
fn letterbox_chw(img: &DynamicImage, in_h: usize, in_w: usize, mean: [f32;3], std: [f32;3]) -> (Vec<f32>, f32)

// fixed height, keep aspect ratio, right-pad width with 0, normalize to [-1,1] -> NCHW (for text recognition)
fn fixed_height_chw(img: &DynamicImage, in_h: usize, in_w: usize) -> Vec<f32>

// force square/fixed-size resize (ignore aspect ratio) + per-channel normalization -> NCHW (for global-structure models like classification / orientation)
fn resize_chw(img: &DynamicImage, in_h: usize, in_w: usize, mean: [f32;3], std: [f32;3]) -> Vec<f32>
```

### 6.4 Postprocessing (`postprocess`)

The model-agnostic pure logic is covered by unit tests.

```rust
fn softmax(logits: &[f32]) -> Vec<f32>            // numerically stable softmax
fn argmax(v: &[f32]) -> (usize, f32)              // top-1 (index, value)
fn iou(a: &BBox, b: &BBox) -> f32                 // intersection over union
fn nms(boxes: &[(BBox, f32)], iou_threshold: f32) -> Vec<usize>   // non-maximum suppression -> kept indices
fn ctc_greedy_decode(logits: &[f32], t: usize, c: usize, dict: &[String]) -> (String, f32)  // CTC
fn connected_boxes(prob: &[f32], w: usize, h: usize, threshold: f32, min_area: usize)
    -> Vec<(usize, usize, usize, usize, f32)>     // DB detection post-processing (connected-component boxes)
fn reading_order(boxes: &[BBox]) -> Vec<usize>   // indices sorting detection boxes into reading order (top-to-bottom rows, left-to-right within a row)
fn xy_cut_order(boxes: &[BBox], min_gap: usize) -> Vec<usize>  // recursive XY-Cut reading order (handles multi-column, tables, side-by-side panels)
```

### 6.5 Error Types

```rust
pub enum VisionError {
    Io(std::io::Error),
    Decode(String),        // image/video decode failure
    Backend(String),       // model load/inference error (absorbs concrete backend errors into a string)
    NotWired(&'static str),// call to a not-yet-wired feature
    Unsupported(String),
}
pub type OcrError = VisionError;          // backward-compatible alias for the old name
pub type Result<T> = std::result::Result<T, VisionError>;
```

> The error types do not depend on optional dependencies (`tract`, etc.). Concrete backend errors are absorbed into strings, so the crate always compiles under any feature combination.

---

## 7. Per-Task Behavior and Maturity

| Task | trait / backend | Status |
|---|---|---|
| Text detection + recognition (OCR) | `TextDetector`/`TextRecognizer`, `TractTextDetector`/`TractTextRecognizer`, `OcrEngine` | **Verified working.** End-to-end verified on real screen photos (phone-shot 4000x3000) with a PP-OCRv5 Korean model. Includes DB unclip, reading-order sorting, adaptive detection resolution, and automatic orientation correction (see [Screen Capture and Photo OCR](#94-screen-capture-and-photo-ocr)). |
| Image classification | `Classifier`, `TractClassifier` | Compiles and structurally complete, **accuracy not verified with a real model** |
| Object detection | `ObjectDetector`, `TractObjectDetector` | Compiles. **Assumes an output layout of `[N,6]` = (x1,y1,x2,y2,score,cls)**, which varies by model, so verify and adapt |
| Layout analysis | `LayoutAnalyzer` | **Trait definition only** (a specialization of object detection with layout labels) |
| Segmentation | `Segmenter`, `Mask` | **Trait definition only** (no concrete backend yet) |
| Video timeline | `SamplingGate`, `TemporalMerger` | Working and tested ([section 8](#8-video-timeline-processing)). But video **decode is not implemented** (Phase 2) |

Verification and limitation notes:

- **OCR accuracy is sensitive to dictionary alignment.** The recognition model's class indices and the character dictionary must correspond 1:1. You must use the dedicated dictionary shipped with the model; a mismatched dictionary shifts every character (feeding a Chinese dictionary to a Korean model turns Hangul into Chinese characters, for example). See [section 11](#11-model-preparation-onnx-models-and-dictionaries).
- **DB unclip is applied.** DB detection predicts a region shrunk relative to the glyphs, so before cropping, boxes are re-expanded the same way as PaddleOCR (offset `d = area×ratio/perimeter`, default ratio 1.5) to prevent glyph edges (especially the first character) from being clipped. Tune it with `TractTextDetector::with_unclip(ratio)`. On clean Korean, accuracy dropped noticeably from glyph clipping without unclip and was measured to normalize once applied.
- **Reading-order sorting (XY-Cut).** Connected-component extraction follows raster-scan order, so words come out scrambled. The detector recovers reading order with a recursive XY-Cut (`postprocess::xy_cut_order`): it projects boxes onto Y to cut horizontal bands (rows), then projects each band onto X to cut vertical columns, recursively, so even complex layouts (multi-column, tables, side-by-side panels such as a table plus its description) read the left column fully before moving to the next (the effect is confirmed on the panel structure of screen photos straightened with `--auto-rotate`). Sorting uses the tight boxes before unclip, because expanded boxes overlap between lines and interfere with the projection split. Adjust the split gap with `with_xycut_gap`; a simple line-grouping version (`reading_order`) also remains in the public API.
- **Detection boxes are axis-aligned** (no rotated unclip). This is sufficient for horizontal subtitles and on-screen text, but skewed individual lines are a target for future work.
- **Adaptive detection resolution.** The detection input is sized in proportion to the input image (CLI `--det-long`, the target long-side px, a multiple of 32). This prevents glyphs from being smeared when a high-resolution photo is forced into a small fixed input (for example 736x1280). The recognition input is fixed at 48x320 (very wide lines may be clipped).
- **Automatic orientation correction.** A screen photo shot with the phone held sideways is entirely rotated. CLI `--auto-rotate` picks, among 0/90/180/270 degrees, the orientation with the highest OCR confidence and straightens it. The orientation classifier (PP-LCNet doc-ori) frequently gets the direction wrong (especially 90 vs 270) on out-of-distribution inputs like screen photos, so instead of the classifier we use the recognition score directly as the metric (exploiting the fact that recognition confidence bottoms out when the input is upside down). A straight input is checked at 0 degrees once and passes quickly; all four orientations are tried only when it does not read well.
- **Remaining limitation: small, dense text.** Small form text loses accuracy at the resolution limit. Raising `--det-long` (slower) or super-resolution preprocessing is future work. Large and medium text (titles, descriptions, paragraphs) is confirmed to be recognized accurately on real screen photos.
- **Speed.** The CLI currently loads and optimizes the model on every call (loading the 88MB server det alone takes tens of seconds). For large batches, use the lightweight mobile det, or a batch mode (future) that loads the model once and processes many images.
- **Dynamic-input ONNX handling.** PaddleOCR ONNX has dynamic input dimensions, so the backend statically fixes the raw proto's input dimensions at load time before parsing. tract was confirmed able to run both detection (DB) and recognition (CRNN/SVTR).

---

## 8. Video Timeline Processing

What makes text extraction from video different: instead of recognizing every frame, only frames that changed are recognized, and duplicates across adjacent frames are merged into time segments.

### 8.1 SSIM sampling gate

If the structural similarity (SSIM) to the previous keyframe is high enough, the frame is dropped. In a 30fps video, subtitles change only a few frames per second, so the pass rate can be lowered to single-digit percent, cutting the recognition calls themselves.

```rust
let mut gate = SamplingGate::new(0.98);        // threshold (higher drops fewer). with_scale adjusts the comparison size
if gate.admit(&frame) {                        // true = admit (new frame), false = skip (same as previous)
    // OCR only this frame
}
ssim(&gray_a, &gray_b) -> f64                  // global SSIM of two grayscale images
```

### 8.2 Temporal merging

Recognition results from adjacent frames that pass the gate may be nearly the same text. Using normalized Levenshtein similarity, "a continuous run of the same subtitle" is merged into a single `(start, end, text)` segment.

```rust
let mut merger = TemporalMerger::new(0.8);     // similarity threshold
if let Some(seg) = merger.push(timestamp, text) { /* subtitle changed -> finalize the previous segment */ }
let last = merger.finish();                     // end of stream -> flush the last segment
```

### 8.3 Output serialization (`emit`)

```rust
emit::to_srt(&segments)  -> String          // SRT subtitles
emit::to_json(&segments) -> Result<String>  // JSON array (timestamps in ms)
emit::to_plain(&segments)-> String          // text only, joined by newlines
```

---

## 9. CLI Tool (`roct`)

Built with `--features cli`.

```bash
cargo build --release --features cli
```

| Subcommand | Arguments | Behavior |
|---|---|---|
| `image` | `<path>` `--det-model` `--rec-model` `--dict` `[--auto-rotate]` `[--det-long N]` | image OCR (detection + recognition). `--auto-rotate` for automatic orientation correction, `--det-long` for detection resolution (long-side px, default 1600). Prints results as `[score] (x,y,w,h) text`. |
| `classify` | `<path>` `--model` `--labels` | image classification (top-k) |
| `ssim` | `<a> <b>` | prints the structural similarity of two images (the sampling-gate metric) |
| `srt` | `<segments.json>` | time-segment JSON -> SRT subtitles (stdout) |
| `video` | `<path>` | video OCR. Phase 2, currently returns a `NotWired` error |

```bash
# Image OCR (a straight input like a scan/screenshot)
roct image page.png --det-model models/det.onnx --rec-model models/rec.onnx --dict models/dict.txt

# Image classification (model + label list)
roct classify cat.jpg --model models/cls.onnx --labels models/imagenet.txt

# segment JSON -> SRT
roct srt segments.json > out.srt
```

### 9.4 Screen Capture and Photo OCR

For a phone-shot screen photo or a high-resolution capture, two things are decisive: orientation and detection resolution. Use `--auto-rotate` to straighten the orientation automatically, and `--det-long` to raise the detection resolution to match the input size.

```bash
# Screen-photo OCR (automatic orientation correction + input-proportional resolution)
roct image photo.jpg \
  --det-model models/det.onnx --rec-model models/rec.onnx --dict models/dict.txt \
  --auto-rotate --det-long 1600
```

Behavior and options:

- **`--auto-rotate`**: picks, among 0/90/180/270 degrees, the orientation with the highest OCR confidence (the number of high-confidence regions). A straight input is checked at 0 degrees once and passes quickly; all four orientations are tried only when 0 degrees does not read well.
- **`--det-long N`**: the target long-side px of the detection input (rounded to a multiple of 32, default 1600). Raise it when small text is not detected (slower).
- **Fine tuning**: the DB unclip ratio (default 1.5) and the XY-Cut split gap (default 8) are adjusted through the Rust API builders (`TractTextDetector::with_unclip` / `with_xycut_gap`).
- **Speed**: the large server det is slow to load. For large batches, using the lightweight mobile det to quickly find the orientation is practical. See [section 11](#11-model-preparation-onnx-models-and-dictionaries) for the detection/recognition models and dictionaries.

---

## 10. Python Bindings (PyO3)

Built with **abi3 (stable ABI)**, so a single wheel is compatible with Python 3.9+ (a pure-Rust wheel with no C++ runtime dependency).

From Python you call image OCR (`recognize_image`) and the core utilities. The caller provides the model and dictionary files (see [section 11](#11-model-preparation-onnx-models-and-dictionaries)). Video processing (`read_video`) will be added together with video decode (Phase 2).

### Installation

```bash
# After PyPI publication: no Rust toolchain needed
pip install rust_ocr_transformer

# From source (latest main / before publication): requires a Rust toolchain on the install machine
pip install "git+https://github.com/arabangoo/rust_ocr_transformer"
```

### API

```python
import json
import rust_ocr_transformer as roct

roct.__version__                              # "0.2.2"

# Image OCR: detection + recognition results as a JSON string (DB unclip and XY-Cut reading order applied by default)
out = roct.recognize_image("page.png", "models/det.onnx", "models/rec.onnx", "models/dict.txt")

# A phone-shot screen photo: automatic orientation correction + proportional detection resolution
out = roct.recognize_image("photo.jpg", "models/det.onnx", "models/rec.onnx", "models/dict.txt",
                           auto_rotate=True, det_long=1600)
for r in json.loads(out):
    print(r["confidence"], r["text"], r["bbox"])   # {"x":..,"y":..,"width":..,"height":..}

# Core utilities
roct.image_ssim("a.png", "b.png")             # structural similarity of two images (0.0-1.0)
roct.segments_to_srt(segments_json)           # time-segment JSON string -> SRT subtitle string
```

Optional arguments of `recognize_image`: `auto_rotate` (default `False`, 0/90/180/270-degree automatic correction) and `det_long` (default 1600, the detection input long-side px). DB unclip and XY-Cut reading order are always applied.

### Building (developers and publishers)

The root `pyproject.toml` (maturin backend) provides the build metadata. Thanks to `[tool.maturin] features = ["python"]`, you can omit `--features python`.

```bash
pip install maturin
maturin develop --release          # install into the current venv
maturin build --release            # build a wheel into target/wheels/
```

---

## 11. Model Preparation (ONNX Models and Dictionaries)

This framework does not bundle models. Inference needs an ONNX model file and (for recognition) a character dictionary. Model weights are not committed to the repo (`.gitignore` excludes `*.onnx` and `models/`).

### PaddleOCR ONNX models (source used for verification)

You can download pre-converted PP-OCR ONNX models and dictionaries directly (PaddlePaddle, Apache-2.0).

```bash
mkdir -p models
base="https://github.com/GreatV/oar-ocr/releases/download/v0.3.0"
# detection (language-agnostic)
curl -sL -o models/det.onnx       "$base/pp-ocrv5_mobile_det.onnx"
# recognition (per-language), e.g. Korean PP-OCRv5
curl -sL -o models/rec.onnx       "$base/korean_pp-ocrv5_mobile_rec.onnx"
# character dictionary: must match the recognition model
curl -sL -o models/dict.txt       "$base/ppocrv5_korean_dict.txt"
```

Recognition models for other languages (English, Latin, Japanese, and so on) and their matching dictionaries are in the same release.

### HuggingFace source (used for screen-photo verification)

A source where you can get per-language recognition models and dictionaries along with the orientation model for rotation correction (PP-LCNet doc-orientation), all in one place (PaddlePaddle conversion, Apache-2.0). This set was used for the real Korean screen-photo verification.

```bash
mkdir -p models
base="https://huggingface.co/monkt/paddleocr-onnx/resolve/main"
# detection (server, high accuracy)
curl -sL -o models/det.onnx   "$base/detection/v5/det.onnx"
# recognition (Korean) + dictionary
curl -sL -o models/rec.onnx   "$base/languages/korean/rec.onnx"
curl -sL -o models/dict.txt   "$base/languages/korean/dict.txt"
# (optional) lightweight mobile detection, for large batches / orientation search
curl -sL -o models/det_mobile.onnx "$base/detection/v3/det.onnx"
```

> The Korean dictionary starts with decomposed jamo (the U+1100 block), but the recognition model outputs precomposed syllables directly, so no separate NFC normalization is needed (verified). For detection, server (v5, about 88MB) is accurate but slow to load, while mobile (v3, about 2.4MB) is fast but has somewhat weaker word splitting and accuracy.

### Multilingual recognition (per-language models and dictionaries)

The detection (det) and orientation (doc-ori) models are language-agnostic and shared; only the recognition (rec) and dictionary (dict) change per language. You can download the major-language models and dictionaries in one go from the same source.

```bash
base="https://huggingface.co/monkt/paddleocr-onnx/resolve/main"
for L in arabic chinese english eslav greek hindi korean latin tamil telugu thai; do
  mkdir -p models/langs/$L
  curl -sL -o models/langs/$L/rec.onnx  "$base/languages/$L/rec.onnx"
  curl -sL -o models/langs/$L/dict.txt  "$base/languages/$L/dict.txt"
done
```

| Language | Folder | Notes |
|---|---|---|
| Chinese/Japanese (CJK) | `chinese` | includes Han characters (about 15,500) and kana (hiragana/katakana). Japanese is recognized by this model too (verified: こんにちは世界, 東京タワー, 日本語認識テスト accurate). Server-class (about 80MB). |
| English | `english` | Latin alphanumerics and symbols |
| Korean | `korean` | outputs precomposed Hangul |
| Latin (European) | `latin` | Latin-script languages such as French, German, Spanish |
| Others | `arabic` · `eslav` (Cyrillic) · `greek` · `hindi` · `tamil` · `telugu` · `thai` | each script |

To switch languages, change only the recognition model and dictionary paths (the detection model stays the same):

```bash
# Japanese (CJK model: Han + kana)
roct image jp.png --det-model models/det.onnx \
  --rec-model models/langs/chinese/rec.onnx --dict models/langs/chinese/dict.txt --auto-rotate

# English
roct image en.png --det-model models/det.onnx \
  --rec-model models/langs/english/rec.onnx --dict models/langs/english/dict.txt
```

> In PP-OCRv5, Japanese is not a separate model; the general CJK (`chinese`) model handles Han characters and kana together.

> **The dictionary must match the model exactly.** The recognition model's number of output classes and the dictionary's length and character order must correspond 1:1 for correct recognition. Even for the same language, the dictionary differs by model version (v3/v4/v5), so use the dedicated dictionary distributed with the model. A mismatched dictionary maps every character incorrectly.

### Input shapes

Specify the input shape when constructing a backend (PP-OCRv5 recommended values): detection `(736, 1280)`, recognition `(48, 320)`, classification `(224, 224)`. Even if the model has dynamic inputs, the backend fixes it to this shape at load time.

---

## 12. Embedding into a Service Pipeline

This library is not a standalone app; it is a core dependency you embed at the vision input stage. The extracted structured results (text, boxes, segments) are useful on their own, or you can feed them into the vision entry point of a retrieval-augmented-generation (RAG) ingest.

### 12.1 Embed into a Rust service

Inference is CPU-bound, so wrap it in `spawn_blocking` inside an async server (axum/actix). Load the model once and reuse it (share the engine via `Arc`, since the traits are `Send + Sync`).

```rust
use std::sync::Arc;
use rust_ocr_transformer::{Frame, OcrEngine, TractTextDetector, TractTextRecognizer};

// load once at startup, then share
let engine = Arc::new(OcrEngine::new(
    TractTextDetector::new("models/det.onnx", (736, 1280))?,
    TractTextRecognizer::new("models/rec.onnx", "models/dict.txt", (48, 320))?,
));

// handler
let eng = engine.clone();
let results = tokio::task::spawn_blocking(move || {
    let frame = Frame::from_path("upload.png")?;
    eng.read(&frame)
}).await??;
```

### 12.2 Video to subtitles (SRT) pipeline

The SSIM gate thins frames, only passing frames are OCR'd, and temporal merging forms subtitle segments.

```rust
use rust_ocr_transformer::{SamplingGate, TemporalMerger, emit};

let mut gate = SamplingGate::new(0.98);
let mut merger = TemporalMerger::new(0.85);
let mut segments = Vec::new();

for frame in frames {                       // video decode is on the caller side (until Phase 2)
    if !gate.admit(&frame) { continue; }    // skip unchanged frames
    let text = engine.read(&frame)?
        .iter().map(|r| r.text.as_str()).collect::<Vec<_>>().join(" ");
    if let Some(seg) = merger.push(frame.timestamp, &text) { segments.push(seg); }
}
if let Some(seg) = merger.finish() { segments.push(seg); }
std::fs::write("out.srt", emit::to_srt(&segments))?;
```

### 12.3 Other languages / batch: wrap the CLI

From non-Python, non-Rust stacks or from batch jobs, call the `roct` binary as a subprocess. It is a single static binary, so you only need to drop `roct` into your container (no runtime dependency).

```bash
roct image /data/page.png --det-model det.onnx --rec-model rec.onnx --dict dict.txt
```

---

## 13. Adding a New Task or Backend

You can plug in a new model or a new task without touching the core.

### 13.1 New backend: implement an existing task trait

For example, wire your own classification model as a `Classifier`. Using the shared [`TractModel`](#133-the-tract-backend-reference) runner plus the reusable pre/post-processing keeps it short.

```rust
use rust_ocr_transformer::{Classifier, Classification, Frame, Result, TractModel};
use rust_ocr_transformer::{preprocess, postprocess};

struct MyClassifier { model: TractModel, labels: Vec<String> }

impl Classifier for MyClassifier {
    fn classify(&self, frame: &Frame) -> Result<Vec<Classification>> {
        let (h, w) = self.model.dims();
        let (data, _) = preprocess::letterbox_chw(&frame.image, h, w,
            preprocess::IMAGENET_MEAN, preprocess::IMAGENET_STD);
        let (logits, _) = self.model.run(data)?;
        let probs = postprocess::softmax(&logits);
        let (i, score) = postprocess::argmax(&probs);
        Ok(vec![Classification { label: self.labels[i].clone(), score }])
    }
}
```

### 13.2 New task: a new trait

For a task the existing six do not express (for example keypoints or depth estimation), define a new `Send + Sync` trait following the `tasks` pattern and have it take a `Frame` as input. For pre/post-processing, reuse the functions in `preprocess`/`postprocess`.

### 13.3 The tract Backend (reference)

What `backends::tract` provides:

```rust
TractModel::load(path, in_h, in_w) -> Result<TractModel>   // loads even dynamic inputs after fixing them statically
fn dims(&self) -> (usize, usize)
fn run(&self, data: Vec<f32>) -> Result<(Vec<f32>, Vec<usize>)>  // (output slice, shape)

TractTextDetector::new(model, (h, w))            // .with_threshold(f32) . with_unclip(ratio) re-expands DB boxes
TractTextRecognizer::new(model, dict, (h, w))
TractClassifier::new(model, labels, (h, w))      // .with_top_k(usize)
TractObjectDetector::new(model, labels, (h, w))  // .with_thresholds(score, iou)
TractDocOrientation::new(model, (h, w))          // orientation classification 0/90/180/270 -> .predict_conf / .correct(frame, min_conf)
```

---

## 14. Build, Feature Combinations, and Testing

If you clone this repository, you need to build it once with a **Rust toolchain (stable, 1.74 or newer recommended)** before using it.

| Use case | Build command | Result |
|---|---|---|
| CLI tool | `cargo build --release --features cli` | the single `target/release/roct` binary |
| Python module | `pip install maturin && maturin develop --release` | `import rust_ocr_transformer` in the current venv |
| Rust library | a `path`/`git` dependency in `Cargo.toml` | links into another Rust project |

```bash
# Default: includes the pure-Rust (tract) backend, zero FFI
cargo build --release

# Core IP only (inject the backend yourself)
cargo build --release --no-default-features

# CLI / Python
cargo build --release --features cli
maturin develop --release

# Test / lint
cargo test                  # postprocessing pure logic (softmax, argmax, IoU, NMS, CTC) + pipeline integration
cargo clippy --all-targets
```

The tests verify what works without external models: with synthetic images they check the SSIM gate, temporal merging, SRT output, and trait-composition engine wiring, and with synthetic tensors they deterministically check NMS, softmax, CTC, and IoU postprocessing (`tests/pipeline.rs`, unit tests in `src/postprocess.rs`).

---

## 15. Directory Layout

```text
rust_ocr_transformer/
  Cargo.toml
  README.md              # this document
  LICENSE                # Apache-2.0
  src/
    lib.rs               # crate root, re-exports
    types.rs             # common types (Frame/BBox/result types/FrameAnalysis/Segment)
    tasks.rs             # task traits (TextDetector/Recognizer/ObjectDetector/Classifier/...)
    engine.rs            # OcrEngine (detection + recognition composition), crop_regions
    preprocess.rs        # letterbox, normalization, CHW tensorization
    postprocess.rs       # NMS, softmax, argmax, IoU, CTC, DB connected-component boxes (+ unit tests)
    sampler.rs           # SSIM sampling gate (video)
    temporal.rs          # temporal merging (video)
    emit.rs              # SRT / JSON / plaintext serialization
    error.rs             # VisionError / Result
    python.rs            # PyO3 bindings (feature = "python")
    bin/
      roct.rs            # CLI binary (feature = "cli")
    backends/
      mod.rs             # feature gates
      tract.rs           # pure-Rust inference backend (feature = "tract")
  tests/
    pipeline.rs          # synthetic-fixture integration tests
```

---

## 16. License

Apache-2.0
