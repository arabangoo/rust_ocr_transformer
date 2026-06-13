//! 해자 1 — SSIM 샘플링 게이트(README 9장).
//!
//! **모든 프레임을 OCR하지 않는다.** 직전 키프레임과 충분히 유사하면 버린다.
//! 30fps 영상에서 자막은 초당 수 프레임만 바뀌므로, 통과율을 한 자릿수 %까지 낮춰
//! 인식 호출 횟수를 직접 줄인다 — 그게 "빠르다"의 정체다(README 5.3).
//!
//! 구조적 유사도(SSIM)는 픽셀 차이(MSE)와 달리 밝기·대비·구조를 함께 보므로
//! 인코딩 노이즈에 강하다. 비교는 다운스케일 그레이스케일로 충분하다.

use image::{DynamicImage, GrayImage};

use crate::types::Frame;

/// SSIM 게이트. 직전 통과 프레임을 키프레임으로 들고 비교한다.
pub struct SamplingGate {
    last_keyframe: Option<GrayImage>,
    /// 이 값보다 SSIM 이 높으면 "동일"로 보고 스킵. 0.0-1.0. 높을수록 둔감(덜 버림).
    threshold: f64,
    /// 비교용 다운스케일 크기(픽셀). 작을수록 빠르고 노이즈에 둔감.
    scale: (u32, u32),
}

impl SamplingGate {
    /// 기본값 — 임계값 0.98, 64x64 다운스케일.
    pub fn new(threshold: f64) -> Self {
        Self { last_keyframe: None, threshold, scale: (64, 64) }
    }

    /// 다운스케일 크기까지 지정.
    pub fn with_scale(threshold: f64, scale: (u32, u32)) -> Self {
        Self { last_keyframe: None, threshold, scale }
    }

    /// true = 통과(새 프레임 — 인식 대상), false = 스킵(직전과 사실상 동일).
    ///
    /// 첫 프레임은 항상 통과한다(비교 대상이 없으므로).
    pub fn admit(&mut self, frame: &Frame) -> bool {
        let gray = downscale_gray(&frame.image, self.scale);
        match &self.last_keyframe {
            Some(prev) if ssim(prev, &gray) >= self.threshold => false,
            _ => {
                self.last_keyframe = Some(gray);
                true
            }
        }
    }

    /// 직전 키프레임을 잊는다(장면 전환 강제 리셋 등에 사용).
    pub fn reset(&mut self) {
        self.last_keyframe = None;
    }
}

/// 이미지를 고정 크기 그레이스케일로 다운스케일. SSIM 비교의 전처리.
pub fn downscale_gray(image: &DynamicImage, (w, h): (u32, u32)) -> GrayImage {
    image
        .resize_exact(w, h, image::imageops::FilterType::Triangle)
        .to_luma8()
}

/// 두 동일 크기 그레이스케일 이미지의 전역 SSIM(structural similarity).
///
/// 반환값은 통상 0.0-1.0 (완전 동일 = 1.0). 크기가 다르면 0.0 을 반환한다.
/// 평균·분산·공분산 기반의 단일 윈도 SSIM — 게이트 판정에는 이 전역 근사로 충분하다.
/// (자막 ROI 한정 SSIM·적응형 임계값은 README 9.1 의 후속 고도화 항목.)
pub fn ssim(a: &GrayImage, b: &GrayImage) -> f64 {
    if a.dimensions() != b.dimensions() {
        return 0.0;
    }
    let n = (a.width() * a.height()) as f64;
    if n == 0.0 {
        return 1.0;
    }

    let pa = a.as_raw();
    let pb = b.as_raw();

    let mut sum_a = 0.0;
    let mut sum_b = 0.0;
    for i in 0..pa.len() {
        sum_a += pa[i] as f64;
        sum_b += pb[i] as f64;
    }
    let mean_a = sum_a / n;
    let mean_b = sum_b / n;

    let mut var_a = 0.0;
    let mut var_b = 0.0;
    let mut cov = 0.0;
    for i in 0..pa.len() {
        let da = pa[i] as f64 - mean_a;
        let db = pb[i] as f64 - mean_b;
        var_a += da * da;
        var_b += db * db;
        cov += da * db;
    }
    var_a /= n;
    var_b /= n;
    cov /= n;

    // 안정화 상수(8비트 동적 범위 L=255 기준 표준값).
    let c1 = (0.01 * 255.0_f64).powi(2);
    let c2 = (0.03 * 255.0_f64).powi(2);

    let numerator = (2.0 * mean_a * mean_b + c1) * (2.0 * cov + c2);
    let denominator = (mean_a * mean_a + mean_b * mean_b + c1) * (var_a + var_b + c2);
    numerator / denominator
}
