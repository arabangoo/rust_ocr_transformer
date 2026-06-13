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
        Cmd::Image { path, det_model, rec_model, dict } => {
            #[cfg(feature = "tract")]
            {
                use rust_ocr_transformer::{Frame, OcrEngine, TractTextDetector, TractTextRecognizer};

                let frame = Frame::from_path(&path)?;
                // 검출 입력 736x1280, 인식 입력 48x320 (PP-OCRv5 권장).
                let detector = TractTextDetector::new(&det_model, (736, 1280))?;
                let recognizer = TractTextRecognizer::new(&rec_model, &dict, (48, 320))?;
                let engine = OcrEngine::new(detector, recognizer);

                let results = engine.read(&frame)?;
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
                let _ = (&path, &det_model, &rec_model, &dict);
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
