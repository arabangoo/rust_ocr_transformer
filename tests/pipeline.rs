//! 코어 파이프라인 통합 테스트 — 백엔드 없이 동작하는 IP 전 구간 검증.
//! (샘플링 게이트 · temporal 병합 · SRT 출력 · trait 합성 엔진 wiring)

use image::{DynamicImage, RgbImage};

use rust_ocr_transformer::{
    emit, BBox, Crop, Frame, OcrEngine, Recognized, Result, SamplingGate, Segment, TextBox,
    TextDetector, TextRecognizer, TemporalMerger, Timestamp,
};

/// 단색 이미지 합성 헬퍼.
fn solid(w: u32, h: u32, rgb: [u8; 3]) -> DynamicImage {
    DynamicImage::ImageRgb8(RgbImage::from_pixel(w, h, image::Rgb(rgb)))
}

#[test]
fn ssim_gate_skips_identical_admits_different() {
    let mut gate = SamplingGate::new(0.98);
    let red = Frame::from_image(solid(100, 40, [200, 30, 30]));

    // 첫 프레임은 비교 대상이 없어 항상 통과.
    assert!(gate.admit(&red), "첫 프레임은 통과해야 한다");
    // 동일 프레임은 SSIM=1.0 → 스킵.
    assert!(!gate.admit(&red), "동일 프레임은 버려야 한다");

    // 전혀 다른 프레임은 통과.
    let white = Frame::from_image(solid(100, 40, [255, 255, 255]));
    assert!(gate.admit(&white), "다른 프레임은 통과해야 한다");
}

#[test]
fn temporal_merger_groups_same_text() {
    let mut m = TemporalMerger::new(0.8);

    assert!(m.push(Timestamp(0), "hi there").is_none(), "첫 텍스트는 구간을 연다");
    // 부분 인식(짧음) → 완전 인식(긺)으로 흔들림. 같은 자막으로 병합되고,
    // 더 완전한(긴) 인식본이 채택되어야 한다.
    assert!(
        m.push(Timestamp(100), "hi there!").is_none(),
        "유사 텍스트는 같은 구간으로 병합"
    );

    // 완전히 다른 자막 → 직전 구간 확정.
    let finished = m.push(Timestamp(200), "goodbye").expect("자막 전환 시 직전 구간 확정");
    assert_eq!(finished.text, "hi there!", "더 완전한(긴) 인식본을 유지");
    assert_eq!(finished.start, Timestamp(0));
    assert_eq!(finished.end, Timestamp(100));

    let last = m.finish().expect("스트림 종료 시 마지막 구간 회수");
    assert_eq!(last.text, "goodbye");
}

#[test]
fn srt_output_format() {
    let segs = vec![
        Segment { start: Timestamp(1200), end: Timestamp(3800), text: "첫 자막".into() },
        Segment { start: Timestamp(4000), end: Timestamp(6500), text: "둘째 자막".into() },
    ];
    let srt = emit::to_srt(&segs);

    assert!(srt.starts_with("1\n"), "인덱스는 1부터");
    assert!(srt.contains("00:00:01,200 --> 00:00:03,800"), "SRT 시간 형식");
    assert!(srt.contains("첫 자막"));
    assert!(srt.contains("2\n00:00:04,000 --> 00:00:06,500"));
}

// ── trait 합성 엔진 wiring 검증 (목 백엔드) ──────────────────────
// 실제 인식 백엔드 대신 목을 주입해 검출→크롭→인식 결선이 도는지만 확인한다.
// (목은 테스트 안에만 존재 — 라이브러리는 가짜 데이터를 노출하지 않는다.)

struct MockDetector;
impl TextDetector for MockDetector {
    fn detect(&self, _frame: &Frame) -> Result<Vec<TextBox>> {
        Ok(vec![TextBox { bbox: BBox::new(0, 0, 10, 10), confidence: 0.9 }])
    }
}

struct MockRecognizer;
impl TextRecognizer for MockRecognizer {
    fn recognize(&self, crops: &[Crop]) -> Result<Vec<Recognized>> {
        Ok(crops
            .iter()
            .map(|c| Recognized { text: "TEXT".into(), confidence: 0.8, bbox: c.bbox })
            .collect())
    }
}

#[test]
fn engine_wires_detect_crop_recognize() {
    let engine = OcrEngine::new(MockDetector, MockRecognizer);
    let frame = Frame::from_image(solid(20, 20, [0, 0, 0]));

    let out = engine.read(&frame).expect("read 성공");
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].text, "TEXT");
    assert_eq!(out[0].bbox, BBox::new(0, 0, 10, 10), "인식 결과에 원본 좌표가 되붙는다");
}
