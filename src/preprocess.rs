//! 재사용 전처리 — 이미지를 모델 입력 텐서(NCHW f32)로 변환한다.
//!
//! 모델 종류(검출·인식·분류·객체검출)와 무관하게 공유하는 순수 함수 모음이다.
//! 고정 입력 형상에 맞춰 letterbox(비율 유지 + 패딩) 또는 높이 고정 리사이즈를 수행하고,
//! 모델 계열별 정규화(ImageNet 평균/표준편차, 또는 [-1,1])를 적용한다.

use image::DynamicImage;

/// ImageNet 정규화 상수 — 검출/분류 모델 다수가 이 평균/표준편차를 기대한다.
pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// letterbox 리사이즈(비율 유지, 좌상단 배치) + 채널별 정규화 → NCHW f32 평탄화.
///
/// 반환: (data[3*in_h*in_w], scale). scale 은 원본→입력공간 배율로, 검출 박스를 원본
/// 좌표로 되돌릴 때 쓴다. 패딩 영역은 0으로 둔다(좌상단 배치라 pad 오프셋은 0).
pub fn letterbox_chw(
    img: &DynamicImage,
    in_h: usize,
    in_w: usize,
    mean: [f32; 3],
    std: [f32; 3],
) -> (Vec<f32>, f32) {
    let (ow, oh) = (img.width().max(1), img.height().max(1));
    let scale = (in_w as f32 / ow as f32).min(in_h as f32 / oh as f32);
    let nw = ((ow as f32 * scale).round() as u32).clamp(1, in_w as u32);
    let nh = ((oh as f32 * scale).round() as u32).clamp(1, in_h as u32);
    let resized = img
        .resize_exact(nw, nh, image::imageops::FilterType::Triangle)
        .to_rgb8();

    let plane = in_h * in_w;
    let mut data = vec![0f32; 3 * plane];
    for y in 0..nh as usize {
        for x in 0..nw as usize {
            let p = resized.get_pixel(x as u32, y as u32);
            for ch in 0..3 {
                data[ch * plane + y * in_w + x] = (p[ch] as f32 / 255.0 - mean[ch]) / std[ch];
            }
        }
    }
    (data, scale)
}

/// 정사각/고정 크기 강제 리사이즈(비율 무시) + 채널별 정규화 → NCHW f32.
/// 분류·방향 추정처럼 전역 구조만 보면 되는 모델(PP-LCNet 등)의 표준 입력 형태다.
/// letterbox 와 달리 패딩 없이 (in_h, in_w) 로 직접 늘린다.
pub fn resize_chw(
    img: &DynamicImage,
    in_h: usize,
    in_w: usize,
    mean: [f32; 3],
    std: [f32; 3],
) -> Vec<f32> {
    let resized = img
        .resize_exact(in_w as u32, in_h as u32, image::imageops::FilterType::Triangle)
        .to_rgb8();
    let plane = in_h * in_w;
    let mut data = vec![0f32; 3 * plane];
    for y in 0..in_h {
        for x in 0..in_w {
            let p = resized.get_pixel(x as u32, y as u32);
            for ch in 0..3 {
                data[ch * plane + y * in_w + x] = (p[ch] as f32 / 255.0 - mean[ch]) / std[ch];
            }
        }
    }
    data
}

/// 높이 고정(in_h) 비율 유지 리사이즈, 폭은 in_w 로 우측 0-pad, [-1,1] 정규화 → NCHW.
/// 텍스트 인식(CRNN/SVTR) 계열의 표준 입력 형태다.
pub fn fixed_height_chw(img: &DynamicImage, in_h: usize, in_w: usize) -> Vec<f32> {
    let (ow, oh) = (img.width().max(1), img.height().max(1));
    let scale = in_h as f32 / oh as f32;
    let nw = ((ow as f32 * scale).round() as u32).clamp(1, in_w as u32);
    let resized = img
        .resize_exact(nw, in_h as u32, image::imageops::FilterType::Triangle)
        .to_rgb8();

    let plane = in_h * in_w;
    let mut data = vec![0f32; 3 * plane];
    for y in 0..in_h {
        for x in 0..nw as usize {
            let p = resized.get_pixel(x as u32, y as u32);
            for ch in 0..3 {
                data[ch * plane + y * in_w + x] = (p[ch] as f32 / 255.0 - 0.5) / 0.5;
            }
        }
    }
    data
}
