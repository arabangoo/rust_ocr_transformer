//! roct — rust_ocr_transformer CLI (`--features cli`).
//!
//! 현재 동작하는 명령은 백엔드 없이 쓰는 코어 유틸리티(`ssim`, `srt`)다.
//! `image` / `video` 는 인식 백엔드(ONNX)가 연결되면 동작한다 — 지금은 정직하게
//! "미연결" 에러를 내고 종료한다(가짜 출력 금지).

use clap::{Parser, Subcommand};

use rust_ocr_transformer::{emit, sampler, types::Segment, OcrError, Result};

#[derive(Parser)]
#[command(name = "roct", version, about = "Video-first Rust OCR pipeline")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// 두 이미지의 구조적 유사도(SSIM) 출력 — 샘플링 게이트 판정 지표.
    Ssim { a: String, b: String },
    /// 시간 구간 JSON 파일 → SRT 자막(stdout).
    Srt { path: String },
    /// 이미지 OCR — 순수 Rust(tract) 백엔드로 검출+인식.
    /// 모델/사전 파일 경로가 필요하다(레포에 미동봉, .gitignore 처리).
    Image {
        /// 입력 이미지 경로.
        path: String,
        /// 검출 모델(.onnx) 경로.
        #[arg(long)]
        det_model: String,
        /// 인식 모델(.onnx) 경로.
        #[arg(long)]
        rec_model: String,
        /// 문자 사전(.txt) 경로.
        #[arg(long)]
        dict: String,
        /// 자동 방향 보정 — 0/90/180/270° 중 OCR 신뢰도가 가장 높은 방향을 선택한다.
        /// 폰을 옆으로 들고 찍은 화면 사진처럼 회전된 입력에 사용(회전 분류기보다 견고).
        #[arg(long)]
        auto_rotate: bool,
        /// 검출 입력의 긴 변 목표 px(입력 크기에 비례해 검출 해상도 결정, 32 배수 반올림).
        #[arg(long, default_value_t = 1600)]
        det_long: usize,
    },
    /// 이미지 분류 — 순수 Rust(tract) 백엔드. 모델/라벨 파일 경로 필요.
    Classify {
        /// 입력 이미지 경로.
        path: String,
        /// 분류 모델(.onnx) 경로.
        #[arg(long)]
        model: String,
        /// 라벨 목록(.txt, 한 줄당 클래스 1개) 경로.
        #[arg(long)]
        labels: String,
    },
    /// 영상 OCR (인식 백엔드 + ffmpeg 필요 — Phase 2, 현재 미연결).
    Video { path: String },
}

/// 환경변수를 f32 로 파싱(없거나 실패면 None). DB unclip 등 튜닝 노브용.
#[cfg(feature = "tract")]
fn env_f32(key: &str) -> Option<f32> {
    std::env::var(key).ok().and_then(|s| s.parse().ok())
}

#[cfg(feature = "tract")]
fn env_usize(key: &str) -> Option<usize> {
    std::env::var(key).ok().and_then(|s| s.parse().ok())
}

/// n 을 가장 가까운 32 의 배수로(최소 32). DBNet 백본이 1/32 다운샘플이라 입력은 32 배수.
#[cfg(feature = "tract")]
fn round32(n: usize) -> usize {
    (((n + 16) / 32) * 32).max(32)
}

/// 검출 입력 (h, w) 산정 — 이미지 긴 변을 det_long 에 맞춰 비례(32 배수). 고해상 사진을
/// 작은 고정 입력에 욱여넣어 글자가 뭉개지는 것을 막는다. ROCT_DET_H/W 가 둘 다 있으면 우선.
#[cfg(feature = "tract")]
fn det_input_shape(frame: &rust_ocr_transformer::Frame, det_long: usize) -> (usize, usize) {
    if let (Some(h), Some(w)) = (env_usize("ROCT_DET_H"), env_usize("ROCT_DET_W")) {
        return (h, w);
    }
    let (iw, ih) = (frame.image.width() as f32, frame.image.height() as f32);
    let long = iw.max(ih).max(1.0);
    let scale = det_long as f32 / long;
    (round32((ih * scale) as usize), round32((iw * scale) as usize))
}

/// 방향 채점 — 인식 결과 중 비어 있지 않고 고신뢰(>=0.8)인 영역 수. 똑바로 선 화면은
/// 고신뢰 영역이 수십 개, 거꾸로/옆으로면 한두 개라 방향 판정의 견고한 척도가 된다.
#[cfg(feature = "tract")]
fn orient_score(rs: &[rust_ocr_transformer::Recognized]) -> usize {
    rs.iter().filter(|r| !r.text.trim().is_empty() && r.confidence >= 0.8).count()
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Ssim { a, b } => {
            let ia = image::open(&a).map_err(|e| OcrError::decode(format!("open {a}: {e}")))?;
            let ib = image::open(&b).map_err(|e| OcrError::decode(format!("open {b}: {e}")))?;
            let ga = sampler::downscale_gray(&ia, (64, 64));
            let gb = sampler::downscale_gray(&ib, (64, 64));
            println!("{:.4}", sampler::ssim(&ga, &gb));
        }
        Cmd::Srt { path } => {
            let raw = std::fs::read_to_string(&path)?;
            let segs: Vec<Segment> = serde_json::from_str(&raw)
                .map_err(|e| OcrError::backend(format!("parse segments json: {e}")))?;
            print!("{}", emit::to_srt(&segs));
        }
        Cmd::Image { path, det_model, rec_model, dict, auto_rotate, det_long } => {
            #[cfg(feature = "tract")]
            {
                use rust_ocr_transformer::{
                    Frame, OcrEngine, Recognized, TractTextDetector, TractTextRecognizer,
                };

                let base = Frame::from_path(&path)?;

                // 주어진 프레임 크기에 맞춰 검출+인식 엔진을 구성(검출 해상도는 입력 비례).
                let build_engine = |frame: &Frame| -> Result<OcrEngine<TractTextDetector, TractTextRecognizer>> {
                    let (dh, dw) = det_input_shape(frame, det_long);
                    let mut det = TractTextDetector::new(&det_model, (dh, dw))?;
                    if let Some(rv) = env_f32("ROCT_UNCLIP") {
                        det = det.with_unclip(rv);
                    }
                    if let Some(g) = env_usize("ROCT_XYCUT_GAP") {
                        det = det.with_xycut_gap(g);
                    }
                    let rec = TractTextRecognizer::new(&rec_model, &dict, (48, 320))?;
                    Ok(OcrEngine::new(det, rec))
                };

                // 방향 결정 + OCR. 방향 모델이 화면사진에서 약하고 신뢰도로 못 거르므로(오판과
                // 정답이 같은 ~0.44), 예측이 0°가 아니면 회전본과 원본을 실제 인식해 고신뢰
                // 영역이 많은 쪽을 택한다(거꾸로면 인식이 바닥 → 결정이 견고, 추가 모델적재 없음).
                let (frame, results): (Frame, Vec<Recognized>) = if auto_rotate {
                    // 자동 방향 보정 — 회전 분류기는 화면사진에서 방향(특히 90 vs 270)을 자주
                    // 틀리므로, OCR 신뢰도를 직접 척도로 쓴다. 먼저 0°를 보고 충분히 잘 읽히면
                    // (흔한 경우) 그대로 채택, 아니면 90/180/270 까지 돌려 가장 잘 읽히는 방향 선택.
                    let rot = |img: &image::DynamicImage, d: u32| match d {
                        90 => img.rotate90(),
                        180 => img.rotate180(),
                        270 => img.rotate270(),
                        _ => img.clone(),
                    };
                    let eng_land = build_engine(&base)?; // 0°/180° (원본 형상)
                    let r0 = eng_land.read(&base)?;
                    let ok = env_usize("ROCT_ORIENT_OK").unwrap_or(8);
                    if orient_score(&r0) >= ok {
                        eprintln!("auto-rotate: 0° score={} (>= {ok}) → keep", orient_score(&r0));
                        (base, r0)
                    } else {
                        let f180 = Frame::new(rot(&base.image, 180), base.index, base.timestamp);
                        let f90 = Frame::new(rot(&base.image, 90), base.index, base.timestamp);
                        let f270 = Frame::new(rot(&base.image, 270), base.index, base.timestamp);
                        let r180 = eng_land.read(&f180)?;
                        let eng_port = build_engine(&f90)?; // 90°/270° (회전 형상)
                        let r90 = eng_port.read(&f90)?;
                        let r270 = eng_port.read(&f270)?;
                        let mut cands = vec![
                            (0u32, base, r0),
                            (180u32, f180, r180),
                            (90u32, f90, r90),
                            (270u32, f270, r270),
                        ];
                        for (d, _, r) in &cands {
                            eprintln!("  {d}°: score={}", orient_score(r));
                        }
                        let best = (0..cands.len()).max_by_key(|&i| orient_score(&cands[i].2)).unwrap();
                        let (deg, f, r) = cands.swap_remove(best);
                        eprintln!("auto-rotate: chosen {deg}°");
                        (f, r)
                    }
                } else {
                    let eng = build_engine(&base)?;
                    let r = eng.read(&base)?;
                    (base, r)
                };

                let (det_h, det_w) = det_input_shape(&frame, det_long);
                eprintln!("detection input: {det_h}x{det_w}");
                for r in &results {
                    println!(
                        "[{:.2}] ({},{},{},{})\t{}",
                        r.confidence, r.bbox.x, r.bbox.y, r.bbox.width, r.bbox.height, r.text
                    );
                }
                eprintln!("{} region(s) recognized", results.len());
            }
            #[cfg(not(feature = "tract"))]
            {
                let _ = (&path, &det_model, &rec_model, &dict, auto_rotate, det_long);
                return Err(OcrError::NotWired(
                    "image OCR needs a recognition backend — build with --features tract",
                ));
            }
        }
        Cmd::Classify { path, model, labels } => {
            #[cfg(feature = "tract")]
            {
                use rust_ocr_transformer::{Classifier, Frame, TractClassifier};

                let frame = Frame::from_path(&path)?;
                let classifier = TractClassifier::new(&model, &labels, (224, 224))?;
                let results = classifier.classify(&frame)?;
                for c in &results {
                    println!("[{:.4}] {}", c.score, c.label);
                }
                eprintln!("{} class(es)", results.len());
            }
            #[cfg(not(feature = "tract"))]
            {
                let _ = (&path, &model, &labels);
                return Err(OcrError::NotWired(
                    "classify needs a backend — build with --features tract",
                ));
            }
        }
        Cmd::Video { .. } => {
            return Err(OcrError::NotWired(
                "video OCR is Phase 2 — decode + multithread pipeline not yet implemented",
            ));
        }
    }
    Ok(())
}
