# rust_ocr_transformer

> **Rust 기반 통합 비전 추론·처리 프레임워크**
>
> 이미지·영상을 `디코드 → 전처리 → 신경망 모델 추론 → 후처리 → 구조화 출력`으로 흘려보낸다.
> 작업별 모델(텍스트 검출·인식, 분류, 객체 검출, 레이아웃, 세그멘테이션)을 trait 뒤에 꽂아
> 교체하는 오케스트레이션 틀이며, 기본 추론은 **순수 Rust(`tract`) / zero FFI / CPU** 로 동작한다.

이 문서는 라이브러리의 **개발자 매뉴얼**이다. 설계 원칙, 공개 API, 작업별 동작과 성숙도,
CLI/Python 사용법, 모델 준비, 서비스 통합, 새 작업/백엔드 추가법, 빌드·테스트 절차를 담는다.

[주요 참고 논문]

1. DBNet: Real-time Scene Text Detection with Differentiable Binarization (텍스트 검출 + DB 후처리) - https://arxiv.org/abs/1911.08947
2. CRNN: An End-to-End Trainable Neural Network for Image-based Sequence Recognition (CNN+RNN+CTC 시퀀스 인식) - https://arxiv.org/abs/1507.05717
3. SVTR: Scene Text Recognition with a Single Visual Model (PP-OCRv5 인식 백본) - https://arxiv.org/abs/2205.00159
4. PP-OCR: A Practical Ultra Lightweight OCR System (임베드하는 경량 OCR 모델군) - https://arxiv.org/abs/2009.09941
5. Image Quality Assessment: From Error Visibility to Structural Similarity (SSIM — 영상 프레임 샘플링 게이트) - https://ece.uwaterloo.ca/~z70wang/research/ssim/

---

## 목차

1. [핵심 특징](#1-핵심-특징)
2. [빠른 시작](#2-빠른-시작)
3. [설치와 Cargo Feature](#3-설치와-cargo-feature)
4. [아키텍처](#4-아키텍처)
5. [공통 타입 레퍼런스](#5-공통-타입-레퍼런스)
6. [공개 API 레퍼런스](#6-공개-api-레퍼런스)
7. [작업별 동작과 성숙도](#7-작업별-동작과-성숙도)
8. [영상 시간축 처리](#8-영상-시간축-처리)
9. [CLI 도구 (`roct`)](#9-cli-도구-roct)
10. [Python 바인딩 (PyO3)](#10-python-바인딩-pyo3)
11. [모델 준비 (ONNX 모델·사전)](#11-모델-준비-onnx-모델사전)
12. [서비스 파이프라인에 붙이기](#12-서비스-파이프라인에-붙이기)
13. [새 작업·백엔드 추가하기](#13-새-작업백엔드-추가하기)
14. [빌드 · Feature 조합 · 테스트](#14-빌드--feature-조합--테스트)
15. [디렉토리 구조](#15-디렉토리-구조)
16. [라이선스](#16-라이선스)

---

## 1. 핵심 특징

비전 파이프라인에서 과소평가되는 영역이 **전·후처리와 오케스트레이션**이다. 모델이 아무리 좋아도
입력 정규화·검출 후처리·디코딩 규약이 어긋나면 결과가 무너진다. 이 라이브러리는 단일 만능 모델이
아니라, 그 위아래의 시스템 엔지니어링을 책임지는 **틀**을 지향한다.

| 원칙 | 의미 |
|---|---|
| **모델은 담지 않고 꽂는다** | "지능"은 trait 뒤의 모델 몫이다. 프레임워크는 디코드·전처리·후처리·조립을 맡는다. 같은 입력이면 같은 모델로 같은 결과(결정적). |
| **작업 단위 추상화** | 텍스트 검출/인식, 분류, 객체 검출, 레이아웃, 세그멘테이션을 각각의 trait 으로 분리. 새 모델은 trait 하나만 구현하면 끼워진다. |
| **순수 Rust 기본 / zero FFI** | 기본 추론 엔진은 `tract`(순수 Rust ONNX 런타임). C++ ONNX Runtime·CUDA·subprocess 불필요. CPU 단일 바이너리·클린 abi3 휠. 속도/GPU 가 필요하면 `ort` 를 opt-in 으로 더한다. |
| **영상 1급 시민** | 정지 이미지는 프레임 1개인 영상의 퇴화 사례. 같은 파이프라인이 둘 다 받고, 영상에는 SSIM 샘플링·시간축 병합이 더해진다. |

### "이해"는 코어 밖이다

이 프레임워크가 만드는 것은 "무엇이 어디에 있다"(텍스트·박스·클래스·좌표)까지의 **구조화된 비전
결과**다. "이 영수증을 JSON 으로 해석", "사고 사진의 손상 부위 판독" 같은 **이해·추론**은 대형
VLM/LLM 의 몫이며, 그건 코어 밖(서버 등)에 위임하고 이 프레임워크는 그 입력을 만든다. 순수 문자
인식은 소형 특화 모델이 프론티어 LLM 보다 정확하고 결정적·저비용이므로, 코어는 LLM-free 로 둔다.

---

## 2. 빠른 시작

### Rust 라이브러리 (OCR)

```rust
use rust_ocr_transformer::{Frame, OcrEngine, TractTextDetector, TractTextRecognizer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 이미지 1장 로드 (영상은 프레임 스트림으로 같은 엔진에 흘린다)
    let frame = Frame::from_path("page.png")?;

    // 검출 모델(입력 736x1280) + 인식 모델(입력 48x320) + 문자 사전
    let detector = TractTextDetector::new("models/det.onnx", (736, 1280))?;
    let recognizer = TractTextRecognizer::new("models/rec.onnx", "models/dict.txt", (48, 320))?;
    let engine = OcrEngine::new(detector, recognizer);

    // 검출 → 크롭 → 인식
    for r in engine.read(&frame)? {
        println!("[{:.2}] ({},{}) {}", r.confidence, r.bbox.x, r.bbox.y, r.text);
    }
    Ok(())
}
```

모델·사전 파일을 받는 방법은 [11장](#11-모델-준비-onnx-모델사전)에 있다.

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

# 이미지 OCR — 모델·사전 경로를 주면 인식 결과 JSON 반환
out = roct.recognize_image("page.png", "models/det.onnx", "models/rec.onnx", "models/dict.txt")
for r in json.loads(out):
    print(r["confidence"], r["text"])
```

---

## 3. 설치와 Cargo Feature

`Cargo.toml`:

```toml
[dependencies]
rust_ocr_transformer = { git = "https://github.com/arabangoo/rust_ocr_transformer", tag = "v0.2.1" }
```

### Feature 목록

| Feature | 활성화 대상 | 비고 |
|---|---|---|
| **`tract`** | 순수 Rust ONNX 추론 백엔드 | 기본 활성. C++ FFI 없음. `tract-onnx` |
| `video` | 영상 디코드(Phase 2) | `ffmpeg-next` — 현재 자리만 잡힘, 미구현 |
| **`cli`** | `roct` 실행 바이너리 | `clap` — 라이브러리 소비자에겐 새지 않는 opt-in |
| **`python`** | PyO3 cdylib 바인딩 | `pyo3`(abi3) |

```toml
# default = ["tract"]   ← 순수 Rust 인식 백엔드 포함, zero FFI

# 코어 IP(타입·trait·샘플링·병합·전후처리)만, 백엔드는 직접 주입하고 싶을 때
rust_ocr_transformer = { version = "0.1", default-features = false }
```

> 기본(`tract`) 빌드는 외부 `.so/.dll` 이나 subprocess 를 요구하지 않는다. 속도·GPU 가 병목이면
> 같은 trait 뒤에 들어오는 `ort`(ONNX Runtime, C++ FFI) 백엔드를 opt-in feature 로 추가할 계획이다
> (현재 미구현 — 데드 feature 를 미리 만들지 않는다).

---

## 4. 아키텍처

```text
디코드 → 전처리 → 작업 모델(trait) → 후처리 → 구조화 출력
 Frame   preprocess   tasks::*        postprocess   FrameAnalysis
                       (모델 plug point)              + SRT/JSON/평문
```

핵심은 **작업 trait 레이어**다. 각 작업(검출·인식·분류·객체검출·레이아웃·세그멘테이션)은
입력으로 공통 [`Frame`](#5-공통-타입-레퍼런스) 을 받고 작업별 결과 타입을 낸다. 모델은 그 trait 의
구현체로 들어오므로, 코어 파이프라인은 어떤 모델을 쓰는지 모른 채 작업 단위로만 조립된다.

- **디코드** — 이미지/영상을 [`Frame`](#5-공통-타입-레퍼런스) 으로. 이미지 = 프레임 1개 영상.
- **전처리** — [`preprocess`](#63-전처리-preprocess): letterbox 리사이즈·정규화·NCHW 텐서화. 작업 무관 재사용.
- **추론** — [`tasks`](#61-작업-trait) trait 뒤의 모델([`backends::tract`](#133-tract-백엔드-참고)).
- **후처리** — [`postprocess`](#64-후처리-postprocess): 비최대 억제(NMS)·softmax·연결성 시계열 분류(CTC) 디코딩·DB 박스.
- **출력** — [`FrameAnalysis`](#5-공통-타입-레퍼런스) 로 여러 작업 결과를 모으고, 영상은 [SRT/JSON](#8-영상-시간축-처리) 으로.

---

## 5. 공통 타입 레퍼런스

`types` 모듈. 결과 타입은 `serde::{Serialize, Deserialize}` 를 구현해 그대로 JSON 으로 떨어뜨릴 수 있다.

```rust
/// 파이프라인 1단위. 이미지는 프레임이 1개인 영상의 퇴화 사례다.
pub struct Frame {
    pub image:     image::DynamicImage,
    pub index:     u64,        // 영상 내 프레임 순번 (이미지면 0)
    pub timestamp: Timestamp,  // 표시 시각(ms) (이미지면 0)
}
// Frame::from_path(p) / Frame::from_image(img) / Frame::new(img, index, ts)

/// 픽셀 좌표계 사각 영역(좌상단 원점).
pub struct BBox { pub x: u32, pub y: u32, pub width: u32, pub height: u32 }
```

### 작업별 결과 타입

```rust
pub struct TextBox     { pub bbox: BBox, pub confidence: f32 }                  // 검출
pub struct Crop        { pub image: image::DynamicImage, pub bbox: BBox }       // 검출→인식 사이
pub struct Recognized  { pub text: String, pub confidence: f32, pub bbox: BBox }// 인식
pub struct Detection   { pub bbox: BBox, pub label: String, pub score: f32 }    // 객체 검출
pub struct Classification { pub label: String, pub score: f32 }                 // 분류
pub struct LayoutRegion { pub bbox: BBox, pub kind: String, pub score: f32 }    // 레이아웃
pub struct Mask        { pub width: u32, pub height: u32, pub classes: Vec<u8> }// 세그멘테이션

/// 한 프레임의 종합 분석 결과 — 여러 작업 출력을 모으는 구조화 출력 컨테이너.
pub struct FrameAnalysis {
    pub recognized:      Vec<Recognized>,
    pub detections:      Vec<Detection>,
    pub classifications: Vec<Classification>,
    pub layout:          Vec<LayoutRegion>,
}
```

### 영상 시간축 타입

```rust
pub struct Timestamp(pub u64);   // 밀리초. to_srt() → "HH:MM:SS,mmm"
pub struct Segment { pub start: Timestamp, pub end: Timestamp, pub text: String }
```

---

## 6. 공개 API 레퍼런스

### 6.1 작업 trait

`tasks` 모듈. 모델을 꽂는 자리다. 모두 `Send + Sync`(멀티스레드 공유 가능).

```rust
pub trait TextDetector:   Send + Sync { fn detect(&self, frame: &Frame) -> Result<Vec<TextBox>>; }
pub trait TextRecognizer: Send + Sync { fn recognize(&self, crops: &[Crop]) -> Result<Vec<Recognized>>; }
pub trait ObjectDetector: Send + Sync { fn detect_objects(&self, frame: &Frame) -> Result<Vec<Detection>>; }
pub trait Classifier:     Send + Sync { fn classify(&self, frame: &Frame) -> Result<Vec<Classification>>; }
pub trait LayoutAnalyzer: Send + Sync { fn analyze_layout(&self, frame: &Frame) -> Result<Vec<LayoutRegion>>; }
pub trait Segmenter:      Send + Sync { fn segment(&self, frame: &Frame) -> Result<Mask>; }
```

### 6.2 OcrEngine — OCR 작업 합성

검출기 + 인식기를 합성한 OCR 파이프라인. 다른 작업은 각자의 trait 으로 직접 조립한다.

```rust
OcrEngine::new(detector: D, recognizer: R) -> OcrEngine<D, R>   // D: TextDetector, R: TextRecognizer
fn read(&self, frame: &Frame) -> Result<Vec<Recognized>>        // 검출 → 크롭 → 인식
fn detector(&self) -> &D
fn recognizer(&self) -> &R

// 헬퍼: 검출 박스대로 프레임을 잘라 Crop 목록 생성
crop_regions(frame: &Frame, boxes: &[TextBox]) -> Vec<Crop>
```

### 6.3 전처리 (`preprocess`)

작업 무관 재사용 함수. 이미지를 모델 입력 텐서(NCHW f32)로.

```rust
pub const IMAGENET_MEAN: [f32; 3];  // [0.485, 0.456, 0.406]
pub const IMAGENET_STD:  [f32; 3];  // [0.229, 0.224, 0.225]

// letterbox(비율 유지+패딩) + 채널별 정규화 → (data[3*h*w], scale). scale 은 박스 역매핑용.
fn letterbox_chw(img: &DynamicImage, in_h: usize, in_w: usize, mean: [f32;3], std: [f32;3]) -> (Vec<f32>, f32)

// 높이 고정·비율 유지, 폭 우측 0-pad, [-1,1] 정규화 → NCHW (텍스트 인식용)
fn fixed_height_chw(img: &DynamicImage, in_h: usize, in_w: usize) -> Vec<f32>

// 정사각/고정 크기 강제 리사이즈(비율 무시) + 채널별 정규화 → NCHW (분류·방향 추정 등 전역구조 모델용)
fn resize_chw(img: &DynamicImage, in_h: usize, in_w: usize, mean: [f32;3], std: [f32;3]) -> Vec<f32>
```

### 6.4 후처리 (`postprocess`)

모델 무관 순수 로직은 단위 테스트로 검증돼 있다.

```rust
fn softmax(logits: &[f32]) -> Vec<f32>            // 수치 안정 softmax
fn argmax(v: &[f32]) -> (usize, f32)              // top-1 (인덱스, 값)
fn iou(a: &BBox, b: &BBox) -> f32                 // 교집합/합집합
fn nms(boxes: &[(BBox, f32)], iou_threshold: f32) -> Vec<usize>   // 비최대 억제 → 유지 인덱스
fn ctc_greedy_decode(logits: &[f32], t: usize, c: usize, dict: &[String]) -> (String, f32)  // CTC
fn connected_boxes(prob: &[f32], w: usize, h: usize, threshold: f32, min_area: usize)
    -> Vec<(usize, usize, usize, usize, f32)>     // DB 검출 후처리(연결요소 박스)
fn reading_order(boxes: &[BBox]) -> Vec<usize>   // 검출 박스를 읽기 순서(위→아래 줄, 줄 안 왼→오른)로 정렬한 인덱스
fn xy_cut_order(boxes: &[BBox], min_gap: usize) -> Vec<usize>  // 재귀 XY-Cut 읽기순서(다단·표·나란한 패널 처리)
```

### 6.5 에러 타입

```rust
pub enum VisionError {
    Io(std::io::Error),
    Decode(String),       // 이미지/영상 디코드 실패
    Backend(String),      // 모델 로드/추론 에러 (구체 백엔드 에러를 문자열로 흡수)
    NotWired(&'static str),// 아직 연결 안 된 기능 호출
    Unsupported(String),
}
pub type OcrError = VisionError;          // 구 명칭 호환 별칭
pub type Result<T> = std::result::Result<T, VisionError>;
```

> 에러 타입은 optional 의존성(`tract` 등)에 의존하지 않는다. 백엔드 구체 에러는 문자열로 흡수하므로
> 어떤 feature 조합에서도 항상 컴파일된다.

---

## 7. 작업별 동작과 성숙도

| 작업 | trait / 백엔드 | 상태 |
|---|---|---|
| 텍스트 검출 + 인식(OCR) | `TextDetector`/`TextRecognizer`, `TractTextDetector`/`TractTextRecognizer`, `OcrEngine` | **동작 검증됨** — PP-OCRv5 한국어 모델로 실제 화면 사진(폰 촬영 4000x3000) end-to-end 검증. DB unclip·읽기순서 정렬·적응형 검출 해상도·자동 방향보정 포함([화면 캡처/사진 OCR](#94-화면-캡처사진-ocr) 참고) |
| 이미지 분류 | `Classifier`, `TractClassifier` | 컴파일·구조 완성, **실모델 정확도 미검증** |
| 객체 검출 | `ObjectDetector`, `TractObjectDetector` | 컴파일됨. **출력 레이아웃을 `[N,6]`=(x1,y1,x2,y2,score,cls)로 가정** — 모델마다 다르니 검증·교체 필요 |
| 레이아웃 분석 | `LayoutAnalyzer` | **trait 정의만**(레이아웃 라벨을 가진 객체 검출의 특수형) |
| 세그멘테이션 | `Segmenter`, `Mask` | **trait 정의만**(구체 백엔드 미구현) |
| 영상 시간축 | `SamplingGate`, `TemporalMerger` | 동작·테스트 완료 ([8장](#8-영상-시간축-처리)). 단 영상 **디코드는 미구현**(Phase 2) |

검증·한계 메모:

- **OCR 정확도는 사전(dict) 정합에 민감하다.** 인식 모델의 클래스 인덱스와 문자 사전이 1:1로 맞아야
  한다. 모델에 딸린 전용 사전을 써야 하며, 다른 사전을 물리면 글자가 통째로 어긋난다(중국어 사전을
  한국어 모델에 물리면 한글이 한자로 나오는 식). [11장](#11-모델-준비-onnx-모델사전) 참고.
- **DB unclip 적용.** DB 검출은 글자보다 수축된 영역을 예측하므로, 크롭 전에 박스를 PaddleOCR 와
  동일한 방식(오프셋 `d = area×ratio/perimeter`, 기본 비율 1.5)으로 되팽창시켜 글자 가장자리(특히 첫
  글자)가 잘리는 것을 막는다. `TractTextDetector::with_unclip(ratio)` 로 조정한다. 깨끗한 한글 기준
  unclip 미적용 시 글자 잘림으로 정확도가 크게 떨어지던 것이 적용 후 정상화됨을 실측 확인했다.
- **읽기순서 정렬 (XY-Cut).** 연결요소 추출은 래스터 스캔 순서라 단어가 뒤섞여 나온다. 검출기는
  재귀 XY-Cut(`postprocess::xy_cut_order`)으로 읽기순서를 잡는다 — 박스를 Y 로 투영해 가로 띠(행)로,
  각 띠를 X 로 투영해 세로 단(열)으로 가르는 것을 재귀해, 다단·표·나란한 패널(표 + 설명) 같은 복잡한
  레이아웃도 왼쪽 단을 끝까지 읽고 다음 단으로 넘어간다(`--auto-rotate` 로 세운 화면 사진의 패널
  구조에서 효과 확인). 정렬은 unclip 전 타이트 박스로 한다 — 팽창 박스는 줄끼리 겹쳐 투영 분할을
  방해하기 때문이다. `with_xycut_gap` 으로 분할 간격을 조정하고, 단순 줄 묶기
  버전(`reading_order`)도 공개 API 로 남아 있다.
- **검출 박스는 축 정렬**이다(회전 unclip 미적용). 수평 자막·화면텍스트엔 충분하나 개별 줄이 기울어진
  텍스트는 후속 고도화 대상이다.
- **적응형 검출 해상도.** 검출 입력을 입력 이미지 크기에 비례해 정한다(CLI `--det-long`, 긴 변 목표
  px, 32 배수). 고해상 사진을 작은 고정 입력(예: 736x1280)에 욱여넣어 글자가 뭉개지는 것을 막는다.
  인식 입력은 48x320 고정(매우 넓은 줄은 잘릴 수 있음).
- **회전 자동 보정.** 폰을 옆으로 들고 찍은 화면 사진은 내용이 통째 회전돼 있다. CLI `--auto-rotate`
  가 0/90/180/270° 중 OCR 신뢰도가 가장 높은 방향을 골라 세운다. 회전 분류 모델(PP-LCNet doc-ori)은
  화면 사진처럼 학습 분포 밖 입력에선 방향(특히 90 vs 270)을 자주 틀려, 분류기 대신 인식 점수를 직접
  척도로 쓴다(거꾸로면 인식 신뢰도가 바닥나는 성질을 이용). 똑바로 선 입력은 0° 한 번만 보고 빠르게
  통과하고, 잘 안 읽힐 때만 네 방향을 모두 시도한다.
- **남은 한계 — 작고 빽빽한 텍스트.** 작은 폼 글자는 해상도 한계로 정확도가 떨어진다. `--det-long`
  을 올리거나(느려짐) 초해상 전처리가 후속 과제다. 큰/중간 텍스트(제목·설명·문단)는 실제 화면 사진에서
  정확하게 인식됨을 확인했다.
- **속도.** 현재 CLI 는 호출당 모델을 새로 적재·최적화한다(server det 88MB 는 적재만 수십 초). 대량
  배치는 가벼운 mobile det 를 쓰거나, 모델을 한 번만 적재해 여러 장을 처리하는 배치 모드(후속)가 낫다.
- **동적 입력 ONNX 처리.** PaddleOCR ONNX 는 입력이 동적 차원이라, 백엔드가 로드 시 원시 proto 의
  입력 차원을 정적으로 고정한 뒤 파싱한다. tract 는 검출(DB)·인식(CRNN/SVTR) 모두 실행 가능함을 확인했다.

---

## 8. 영상 시간축 처리

영상에서 텍스트를 뽑을 때의 차별점 — 모든 프레임을 인식하지 않고, 변화가 있는 프레임만 인식한 뒤
인접 프레임의 중복을 시간 구간으로 병합한다.

### 8.1 SSIM 샘플링 게이트

직전 키프레임과 구조적 유사도(SSIM)가 충분히 높으면 프레임을 버린다. 30fps 영상에서 자막은 초당 수
프레임만 바뀌므로, 통과율을 한 자릿수 %까지 낮춰 인식 호출 자체를 줄인다.

```rust
let mut gate = SamplingGate::new(0.98);        // 임계값(높을수록 덜 버림). with_scale 로 비교 크기 조정
if gate.admit(&frame) {                        // true=통과(새 프레임), false=스킵(직전과 동일)
    // 이 프레임만 OCR
}
ssim(&gray_a, &gray_b) -> f64                  // 두 그레이스케일 이미지의 전역 SSIM
```

### 8.2 temporal 병합

게이트를 통과한 인접 프레임의 인식 결과는 거의 같은 텍스트일 수 있다. 정규화 Levenshtein 유사도로
"같은 자막의 연속"을 하나의 `(start, end, text)` 구간으로 합친다.

```rust
let mut merger = TemporalMerger::new(0.8);     // 유사도 임계값
if let Some(seg) = merger.push(timestamp, text) { /* 자막이 바뀜 → 직전 구간 확정 */ }
let last = merger.finish();                     // 스트림 끝 → 마지막 구간 회수
```

### 8.3 출력 직렬화 (`emit`)

```rust
emit::to_srt(&segments)  -> String          // SRT 자막
emit::to_json(&segments) -> Result<String>  // JSON 배열(타임스탬프 ms)
emit::to_plain(&segments)-> String          // 텍스트만 줄바꿈 연결
```

---

## 9. CLI 도구 (`roct`)

`--features cli` 로 빌드된다.

```bash
cargo build --release --features cli
```

| 서브커맨드 | 인자 | 동작 |
|---|---|---|
| `image` | `<path>` `--det-model` `--rec-model` `--dict` `[--auto-rotate]` `[--det-long N]` | 이미지 OCR(검출+인식). `--auto-rotate` 는 회전 자동보정, `--det-long` 은 검출 해상도(긴 변 px, 기본 1600). 결과를 `[score] (x,y,w,h) text` 로 출력 |
| `classify` | `<path>` `--model` `--labels` | 이미지 분류(top-k) |
| `ssim` | `<a> <b>` | 두 이미지의 구조적 유사도 출력(샘플링 게이트 지표) |
| `srt` | `<segments.json>` | 시간 구간 JSON → SRT 자막(stdout) |
| `video` | `<path>` | 영상 OCR — Phase 2, 현재 `NotWired` 에러 반환 |

```bash
# 이미지 OCR (스캔/스크린샷처럼 똑바로 선 입력)
roct image page.png --det-model models/det.onnx --rec-model models/rec.onnx --dict models/dict.txt

# 이미지 분류 (모델 + 라벨 목록)
roct classify cat.jpg --model models/cls.onnx --labels models/imagenet.txt

# 구간 JSON → SRT
roct srt segments.json > out.srt
```

### 9.4 화면 캡처/사진 OCR

폰으로 찍은 화면 사진이나 고해상 캡처는 두 가지가 결정적이다 — 회전과 검출 해상도. `--auto-rotate`
로 방향을 자동으로 세우고, `--det-long` 으로 검출 해상도를 입력 크기에 맞춰 올린다.

```bash
# 화면 사진 OCR (회전 자동보정 + 입력 비례 해상도)
roct image photo.jpg \
  --det-model models/det.onnx --rec-model models/rec.onnx --dict models/dict.txt \
  --auto-rotate --det-long 1600
```

동작과 옵션:

- **`--auto-rotate`** — 0/90/180/270° 중 OCR 신뢰도(고신뢰 영역 수)가 가장 높은 방향을 고른다.
  똑바로 선 입력은 0° 한 번만 보고 빠르게 통과하고, 0° 가 잘 안 읽힐 때만 네 방향을 모두 시도한다.
- **`--det-long N`** — 검출 입력의 긴 변 목표 px(32 배수로 반올림, 기본 1600). 작은 글자가 안 잡히면
  올린다(느려진다).
- **세밀 튜닝** — DB unclip 비율(기본 1.5)·XY-Cut 분할 간격(기본 8)은 Rust API 의 빌더
  (`TractTextDetector::with_unclip` / `with_xycut_gap`)로 조정한다.
- **속도** — 큰 server det 는 적재가 느리다. 대량 배치는 가벼운 mobile det 로 방향을 빠르게 잡는 식이
  실용적이다. 검출·인식 모델과 사전 출처는 [11장](#11-모델-준비-onnx-모델사전) 참고.

---

## 10. Python 바인딩 (PyO3)

**abi3(stable ABI)** 로 빌드되어 Python 3.9+ 단일 휠로 호환된다(C++ 런타임 의존 없는 순수 Rust 휠).

Python 에서 이미지 OCR(`recognize_image`)과 코어 유틸리티를 호출한다. 모델·사전 파일은 호출자가
제공한다([11장](#11-모델-준비-onnx-모델사전)). 영상 처리(`read_video`)는 영상 디코드(Phase 2)와 함께 추가된다.

### 설치

```bash
# PyPI 게시 후 — Rust 툴체인 불필요
pip install rust_ocr_transformer

# 소스에서(최신 main / 게시 전) — 설치 머신에 Rust 툴체인 필요
pip install "git+https://github.com/arabangoo/rust_ocr_transformer"
```

### API

```python
import json
import rust_ocr_transformer as roct

roct.__version__                              # "0.2.1"

# 이미지 OCR — 검출+인식 결과를 JSON 문자열로 (DB unclip·XY-Cut 읽기순서 기본 적용)
out = roct.recognize_image("page.png", "models/det.onnx", "models/rec.onnx", "models/dict.txt")

# 폰으로 찍은 화면 사진 — 회전 자동보정 + 검출 해상도 비례
out = roct.recognize_image("photo.jpg", "models/det.onnx", "models/rec.onnx", "models/dict.txt",
                           auto_rotate=True, det_long=1600)
for r in json.loads(out):
    print(r["confidence"], r["text"], r["bbox"])   # {"x":..,"y":..,"width":..,"height":..}

# 코어 유틸리티
roct.image_ssim("a.png", "b.png")             # 두 이미지의 구조적 유사도(0.0-1.0)
roct.segments_to_srt(segments_json)           # 시간 구간 JSON 문자열 → SRT 자막 문자열
```

`recognize_image` 의 선택 인자 — `auto_rotate`(기본 `False`, 0/90/180/270° 자동 보정) ·
`det_long`(기본 1600, 검출 입력 긴 변 px). DB unclip 과 XY-Cut 읽기순서는 항상 적용된다.

### 빌드 (개발자 · 게시자)

루트 `pyproject.toml`(maturin 백엔드)이 빌드 메타데이터를 제공한다. `[tool.maturin] features = ["python"]`
덕분에 `--features python` 을 생략해도 된다.

```bash
pip install maturin
maturin develop --release          # 현재 venv 에 설치
maturin build --release            # target/wheels/ 에 휠 빌드
```

---

## 11. 모델 준비 (ONNX 모델·사전)

이 프레임워크는 모델을 담지 않는다 — 추론에는 ONNX 모델 파일과(인식의 경우) 문자 사전이 필요하다.
모델 가중치는 레포에 커밋하지 않는다(`.gitignore` 가 `*.onnx` 와 `models/` 를 제외).

### PaddleOCR ONNX 모델 (검증에 사용한 출처)

사전 변환된 PP-OCR ONNX 모델과 사전을 직접 받을 수 있다(PaddlePaddle, Apache-2.0).

```bash
mkdir -p models
base="https://github.com/GreatV/oar-ocr/releases/download/v0.3.0"
# 검출(언어 무관)
curl -sL -o models/det.onnx       "$base/pp-ocrv5_mobile_det.onnx"
# 인식(언어별) — 예: 한국어 PP-OCRv5
curl -sL -o models/rec.onnx       "$base/korean_pp-ocrv5_mobile_rec.onnx"
# 문자 사전 — 반드시 인식 모델과 짝이 맞는 것
curl -sL -o models/dict.txt       "$base/ppocrv5_korean_dict.txt"
```

영어·라틴·일본어 등 다른 언어 인식 모델과 그에 맞는 사전도 같은 릴리스에 있다.

### HuggingFace 출처 (화면 사진 검증에 사용)

언어별 인식 모델·사전과 함께 회전 보정용 방향 모델(PP-LCNet doc-orientation)까지 한 곳에서 받을 수
있는 출처다(PaddlePaddle 변환, Apache-2.0). 실제 한국어 화면 사진 검증에 이 세트를 사용했다.

```bash
mkdir -p models
base="https://huggingface.co/monkt/paddleocr-onnx/resolve/main"
# 검출(server, 고정확)
curl -sL -o models/det.onnx   "$base/detection/v5/det.onnx"
# 인식(한국어) + 사전
curl -sL -o models/rec.onnx   "$base/languages/korean/rec.onnx"
curl -sL -o models/dict.txt   "$base/languages/korean/dict.txt"
# (선택) 가벼운 mobile 검출 — 대량 배치/방향 탐색용
curl -sL -o models/det_mobile.onnx "$base/detection/v3/det.onnx"
```

> 한국어 사전은 조합형 자모(U+1100 블록)로 시작하지만, 인식 모델은 완성형 음절을 직접 출력하므로
> 별도 NFC 정규화가 필요 없다(실측 확인). 검출은 server(v5, 약 88MB)가 정확하나 적재가 느리고,
> mobile(v3, 약 2.4MB)은 빠르나 단어 분할·정확도가 다소 낮다.

### 다국어 인식 (언어별 모델·사전)

검출(det)·방향(doc-ori) 모델은 언어 무관 공용이고, 인식(rec)·사전(dict)만 언어별로 바꾼다. 같은
출처에서 주요 언어 모델·사전을 한꺼번에 받을 수 있다.

```bash
base="https://huggingface.co/monkt/paddleocr-onnx/resolve/main"
for L in arabic chinese english eslav greek hindi korean latin tamil telugu thai; do
  mkdir -p models/langs/$L
  curl -sL -o models/langs/$L/rec.onnx  "$base/languages/$L/rec.onnx"
  curl -sL -o models/langs/$L/dict.txt  "$base/languages/$L/dict.txt"
done
```

| 언어 | 폴더 | 비고 |
|---|---|---|
| 중국어·일본어(CJK) | `chinese` | 한자(약 15,500) + 가나(히라가나·가타카나) 포함 — 일본어도 이 모델로 인식(실측: こんにちは世界·東京タワー·日本語認識テスト 정확). server급(약 80MB) |
| 영어 | `english` | 라틴 영숫자·기호 |
| 한국어 | `korean` | 완성형 한글 출력 |
| 라틴(유럽어) | `latin` | 프랑스·독일·스페인 등 라틴 문자권 |
| 그 외 | `arabic` · `eslav`(키릴) · `greek` · `hindi` · `tamil` · `telugu` · `thai` | 각 문자권 |

언어 전환은 인식 모델·사전 경로만 바꾸면 된다(검출 모델은 그대로):

```bash
# 일본어 (CJK 모델 — 한자 + 가나)
roct image jp.png --det-model models/det.onnx \
  --rec-model models/langs/chinese/rec.onnx --dict models/langs/chinese/dict.txt --auto-rotate

# 영어
roct image en.png --det-model models/det.onnx \
  --rec-model models/langs/english/rec.onnx --dict models/langs/english/dict.txt
```

> 일본어는 PP-OCRv5 에서 별도 모델이 아니라 CJK 범용(`chinese`) 모델이 한자·가나를 함께 처리한다.

> **사전은 모델과 정확히 짝이 맞아야 한다.** 인식 모델의 출력 클래스 수와 사전 길이·문자 순서가 1:1로
> 대응해야 정상 인식된다. 같은 언어라도 모델 버전(v3/v4/v5)에 따라 사전이 다르므로, 모델과 함께 배포된
> 전용 사전을 쓴다. 사전이 어긋나면 글자가 통째로 잘못 매핑된다.

### 입력 형상

백엔드 생성 시 입력 형상을 지정한다(PP-OCRv5 권장값): 검출 `(736, 1280)`, 인식 `(48, 320)`, 분류 `(224, 224)`.
모델은 동적 입력이어도 백엔드가 이 형상으로 고정해 적재한다.

---

## 12. 서비스 파이프라인에 붙이기

이 라이브러리는 단독 앱이 아니라 비전 입력 단에 박아 넣는 코어 의존성이다. 추출된 구조화 결과(텍스트·
박스·구간)는 그 자체로 쓰거나, 검색 증강 생성(RAG) 인제스트의 비전 입구로 흘려보낸다.

### 12.1 Rust 서비스에 임베드

추론은 CPU 바운드이므로 async 서버(axum/actix)에서는 `spawn_blocking` 으로 감싼다. 모델은 한 번
로드해 재사용한다(엔진을 `Arc` 로 공유 — trait 이 `Send + Sync`).

```rust
use std::sync::Arc;
use rust_ocr_transformer::{Frame, OcrEngine, TractTextDetector, TractTextRecognizer};

// 기동 시 1회 로드 후 공유
let engine = Arc::new(OcrEngine::new(
    TractTextDetector::new("models/det.onnx", (736, 1280))?,
    TractTextRecognizer::new("models/rec.onnx", "models/dict.txt", (48, 320))?,
));

// 핸들러
let eng = engine.clone();
let results = tokio::task::spawn_blocking(move || {
    let frame = Frame::from_path("upload.png")?;
    eng.read(&frame)
}).await??;
```

### 12.2 영상 → 자막(SRT) 파이프라인

SSIM 게이트로 프레임을 솎고, 통과 프레임만 OCR 한 뒤 temporal 병합으로 자막 구간을 만든다.

```rust
use rust_ocr_transformer::{SamplingGate, TemporalMerger, emit};

let mut gate = SamplingGate::new(0.98);
let mut merger = TemporalMerger::new(0.85);
let mut segments = Vec::new();

for frame in frames {                       // 영상 디코드는 호출자 측(Phase 2 전까지)
    if !gate.admit(&frame) { continue; }    // 변화 없는 프레임 스킵
    let text = engine.read(&frame)?
        .iter().map(|r| r.text.as_str()).collect::<Vec<_>>().join(" ");
    if let Some(seg) = merger.push(frame.timestamp, &text) { segments.push(seg); }
}
if let Some(seg) = merger.finish() { segments.push(seg); }
std::fs::write("out.srt", emit::to_srt(&segments))?;
```

### 12.3 타 언어 / 배치 — CLI 래핑

Python·Rust 가 아닌 스택이나 배치 잡에서는 `roct` 바이너리를 subprocess 로 호출한다. 단일 정적
바이너리라 컨테이너에 `roct` 하나만 넣으면 된다(런타임 의존 없음).

```bash
roct image /data/page.png --det-model det.onnx --rec-model rec.onnx --dict dict.txt
```

---

## 13. 새 작업·백엔드 추가하기

코어를 건드리지 않고 새 모델·새 작업을 끼울 수 있다.

### 13.1 새 백엔드 — 기존 작업 trait 구현

예: 자체 분류 모델을 `Classifier` 로. 공용 [`TractModel`](#64-tract-백엔드) 러너 + 재사용 전·후처리를 쓰면 짧다.

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

### 13.2 새 작업 — 새 trait

기존 6개로 표현 안 되는 작업(예: 키포인트·깊이 추정)은 `tasks` 패턴대로 `Send + Sync` trait 을 새로
정의하고 `Frame` 을 입력으로 받게 만든다. 전·후처리는 `preprocess`/`postprocess` 의 재사용 함수를 쓴다.

### 13.3 tract 백엔드 (참고)

`backends::tract` 가 제공하는 것:

```rust
TractModel::load(path, in_h, in_w) -> Result<TractModel>   // 동적 입력도 정적 고정 후 적재
fn dims(&self) -> (usize, usize)
fn run(&self, data: Vec<f32>) -> Result<(Vec<f32>, Vec<usize>)>  // (출력 슬라이스, 형상)

TractTextDetector::new(model, (h, w))            // .with_threshold(f32) · .with_unclip(ratio) — DB 박스 되팽창
TractTextRecognizer::new(model, dict, (h, w))
TractClassifier::new(model, labels, (h, w))      // .with_top_k(usize)
TractObjectDetector::new(model, labels, (h, w))  // .with_thresholds(score, iou)
TractDocOrientation::new(model, (h, w))          // 방향 분류 0/90/180/270 → .predict_conf / .correct(frame, min_conf)
```

---

## 14. 빌드 · Feature 조합 · 테스트

이 저장소를 clone 한 경우, 쓰려면 **Rust 툴체인(stable, 1.74 이상 권장)** 으로 한 번 빌드해야 한다.

| 쓰는 방식 | 빌드 명령 | 결과물 |
|---|---|---|
| CLI 도구 | `cargo build --release --features cli` | `target/release/roct` 단일 바이너리 |
| Python 모듈 | `pip install maturin && maturin develop --release` | 현재 venv 에 `import rust_ocr_transformer` |
| Rust 라이브러리 | `Cargo.toml` 에 `path`/`git` 의존성 | 다른 Rust 프로젝트에 링크 |

```bash
# 기본: 순수 Rust(tract) 백엔드 포함, zero FFI
cargo build --release

# 코어 IP 만 (백엔드 직접 주입)
cargo build --release --no-default-features

# CLI / Python
cargo build --release --features cli
maturin develop --release

# 테스트 / 린트
cargo test                  # 후처리 순수 로직(softmax·argmax·IoU·NMS·CTC) + 파이프라인 통합
cargo clippy --all-targets
```

테스트는 외부 모델 없이 동작하는 범위를 검증한다 — 합성 이미지로 SSIM 게이트·temporal 병합·SRT
출력·trait 합성 엔진 결선을, 합성 텐서로 NMS·softmax·CTC·IoU 후처리를 결정적으로 확인한다
(`tests/pipeline.rs`, `src/postprocess.rs` 단위 테스트).

---

## 15. 디렉토리 구조

```text
rust_ocr_transformer/
  Cargo.toml
  README.md              # 이 문서
  LICENSE                # Apache-2.0
  src/
    lib.rs               # 크레이트 루트 · re-export
    types.rs             # 공통 타입(Frame/BBox/결과 타입/FrameAnalysis/Segment)
    tasks.rs             # 작업 trait(TextDetector/Recognizer/ObjectDetector/Classifier/...)
    engine.rs            # OcrEngine(검출+인식 합성) · crop_regions
    preprocess.rs        # letterbox · 정규화 · CHW 텐서화
    postprocess.rs       # NMS · softmax · argmax · IoU · CTC · DB 연결요소 박스 (+ 단위 테스트)
    sampler.rs           # SSIM 샘플링 게이트(영상)
    temporal.rs          # temporal 병합(영상)
    emit.rs              # SRT / JSON / 평문 직렬화
    error.rs             # VisionError / Result
    python.rs            # PyO3 바인딩 (feature = "python")
    bin/
      roct.rs            # CLI 바이너리 (feature = "cli")
    backends/
      mod.rs             # feature 게이트
      tract.rs           # 순수 Rust 추론 백엔드 (feature = "tract")
  tests/
    pipeline.rs          # 합성 픽스처 기반 통합 테스트
```

---

## 16. 라이선스

Apache-2.0
