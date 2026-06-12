// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Token-weighted prefill load estimation for [`RouterMode::LeastPrefillLoaded`].
//!
//! The router weighs each in-flight request by its input sequence length so a
//! worker holding long / multimodal prompts counts as more loaded than one
//! holding several short prompts. For a multimodal request the dominant cost is
//! the image tokens, but the language model only sees those *after* the vision
//! processor expands each `<image>` placeholder into
//! `grid_thw.prod() / merge_size^2` tokens. At the routing point (frontend, no
//! decode) `token_ids` still holds the single unexpanded placeholder, so a naive
//! length undercounts an image by orders of magnitude.
//!
//! Running the vision processor (resize + patchify + ViT) at the router would be
//! expensive, so we estimate the expanded token count directly from the image's
//! pixel dimensions using the same closed form the processor uses:
//!
//! ```text
//! factor           = patch_size * merge_size              // Qwen3-VL: 16 * 2 = 32
//! (h_bar, w_bar)   = smart_resize(h, w, factor, min_px, max_px)
//! num_image_tokens = h_bar * w_bar / factor^2             // == grid.prod() / merge^2
//! ```
//!
//! Only `data:` URIs (inline base64 — the `vllm bench serve` case) and already
//! decoded descriptors are measured here; `http(s)` URLs would require a network
//! fetch and are skipped (such an image contributes only its placeholder token).
//!
//! Defaults match `Qwen/Qwen3-VL-8B-Instruct`'s `preprocessor_config.json` and
//! can be overridden per deployment via the `DYN_ROUTER_MM_*` env vars.

use std::io::Cursor;
use std::sync::OnceLock;

use base64::{Engine as _, engine::general_purpose};

use crate::preprocessor::PreprocessedRequest;
use crate::protocols::common::preprocessor::MultimodalData;

/// Vision-token estimation parameters. Defaults target Qwen3-VL.
#[derive(Clone, Copy, Debug)]
pub struct MmPrefillTokenConfig {
    /// ViT patch size in px (`patch_size`).
    pub patch_size: u32,
    /// Spatial merge factor (`spatial_merge_size`).
    pub merge_size: u32,
    /// Lower clamp on the resized pixel count (`size.shortest_edge`).
    pub min_pixels: u64,
    /// Upper clamp on the resized pixel count (`size.longest_edge`).
    pub max_pixels: u64,
}

impl Default for MmPrefillTokenConfig {
    fn default() -> Self {
        // Qwen/Qwen3-VL-8B-Instruct preprocessor_config.json
        Self {
            patch_size: 16,
            merge_size: 2,
            min_pixels: 65_536,     // 256 * 256
            max_pixels: 16_777_216, // 4096 * 4096
        }
    }
}

impl MmPrefillTokenConfig {
    /// `patch_size * merge_size`: both the smart-resize rounding factor and the
    /// square root of pixels-per-token (`factor^2` pixels collapse to one token).
    fn factor(&self) -> u64 {
        (self.patch_size.max(1) * self.merge_size.max(1)) as u64
    }
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(default)
}

/// Process-wide config, parsed once from the environment.
fn config() -> &'static MmPrefillTokenConfig {
    static CONFIG: OnceLock<MmPrefillTokenConfig> = OnceLock::new();
    CONFIG.get_or_init(|| {
        let d = MmPrefillTokenConfig::default();
        MmPrefillTokenConfig {
            patch_size: env_u64("DYN_ROUTER_MM_PATCH_SIZE", d.patch_size as u64) as u32,
            merge_size: env_u64("DYN_ROUTER_MM_MERGE_SIZE", d.merge_size as u64) as u32,
            min_pixels: env_u64("DYN_ROUTER_MM_MIN_PIXELS", d.min_pixels),
            max_pixels: env_u64("DYN_ROUTER_MM_MAX_PIXELS", d.max_pixels),
        }
    })
}

/// Mirror of HF `smart_resize`: round each side to a multiple of `factor`, then
/// scale so the total pixel count lands within `[min_pixels, max_pixels]` while
/// preserving aspect ratio. Returns `(height_bar, width_bar)` in pixels.
fn smart_resize(height: u32, width: u32, cfg: &MmPrefillTokenConfig) -> (u64, u64) {
    let factor = cfg.factor();
    let ff = factor as f64;
    let h = height.max(1) as f64;
    let w = width.max(1) as f64;

    // `max(factor, round/floor/ceil(x / factor) * factor)` — never below one patch-merge unit.
    let round_to = |x: f64| ((x / ff).round() as u64).max(1) * factor;
    let floor_to = |x: f64| ((x / ff).floor() as u64).max(1) * factor;
    let ceil_to = |x: f64| ((x / ff).ceil() as u64).max(1) * factor;

    let mut h_bar = round_to(h);
    let mut w_bar = round_to(w);

    if h_bar * w_bar > cfg.max_pixels {
        let beta = ((h * w) / cfg.max_pixels as f64).sqrt();
        h_bar = floor_to(h / beta);
        w_bar = floor_to(w / beta);
    } else if h_bar * w_bar < cfg.min_pixels {
        let beta = (cfg.min_pixels as f64 / (h * w)).sqrt();
        h_bar = ceil_to(h * beta);
        w_bar = ceil_to(w * beta);
    }
    (h_bar, w_bar)
}

/// Estimated number of expanded LLM tokens a single image of the given pixel
/// dimensions contributes to the prefill sequence.
pub fn estimate_image_tokens(width: u32, height: u32, cfg: &MmPrefillTokenConfig) -> u64 {
    let (h_bar, w_bar) = smart_resize(height, width, cfg);
    let factor = cfg.factor();
    (h_bar * w_bar) / (factor * factor)
}

/// `(width, height)` of a multimodal image item, if cheaply available.
///
/// `Decoded` descriptors carry the shape directly. `data:` URIs are base64
/// decoded and their header parsed (header only — no full pixel decode).
/// `http(s)` URLs return `None`; we deliberately avoid a network fetch on the
/// routing hot path.
fn image_dimensions(item: &MultimodalData) -> Option<(u32, u32)> {
    match item {
        MultimodalData::Decoded(desc) => desc
            .image_height_width()
            .map(|(h, w)| (w as u32, h as u32)),
        MultimodalData::Url(url) => dimensions_from_data_uri(url.as_str()),
        MultimodalData::RawUrl(s) => dimensions_from_data_uri(s),
    }
}

/// Parse `(width, height)` from a `data:` URI's inline base64 image header.
fn dimensions_from_data_uri(uri: &str) -> Option<(u32, u32)> {
    let payload = uri.strip_prefix("data:")?.split_once(',')?.1;
    if payload.is_empty() {
        return None;
    }
    let bytes = general_purpose::STANDARD.decode(payload).ok()?;
    image::ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .ok()?
        .into_dimensions()
        .ok()
}

/// Total prefill weight (input-token load) for a request under
/// [`RouterMode::LeastPrefillLoaded`].
///
/// - When an MM-expanded routing sequence is already attached (e.g. set by a
///   dedicated MM router worker), its length is authoritative and used as-is —
///   it already accounts for the multimodal tokens, so we do not re-estimate.
/// - Otherwise: the text `token_ids` length plus a pixel-derived estimate of the
///   expanded tokens for each inline/decoded image. The unexpanded placeholder
///   already in `token_ids` (one per image) leaves the estimate high by at most
///   one token per image — negligible against hundreds/thousands of image tokens.
pub fn prefill_token_weight(request: &PreprocessedRequest) -> u64 {
    if let Some(mm) = request.mm_routing_info.as_ref() {
        if !mm.routing_token_ids.is_empty() {
            return mm.routing_token_ids.len() as u64;
        }
    }

    let cfg = config();
    let mut weight = request.token_ids.len() as u64;

    if let Some(images) = request
        .multi_modal_data
        .as_ref()
        .and_then(|mm| mm.get("image_url"))
    {
        for item in images {
            if let Some((width, height)) = image_dimensions(item) {
                weight += estimate_image_tokens(width, height, cfg);
            }
        }
    }

    weight
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageFormat, RgbImage};

    const QWEN3VL: MmPrefillTokenConfig = MmPrefillTokenConfig {
        patch_size: 16,
        merge_size: 2,
        min_pixels: 65_536,
        max_pixels: 16_777_216,
    };

    fn png_data_uri(width: u32, height: u32) -> String {
        let img = RgbImage::new(width, height);
        let mut buf = Cursor::new(Vec::new());
        img.write_to(&mut buf, ImageFormat::Png).unwrap();
        let b64 = general_purpose::STANDARD.encode(buf.into_inner());
        format!("data:image/png;base64,{b64}")
    }

    #[test]
    fn tokens_for_aligned_image_no_clamp() {
        // 1024x1024 is already a multiple of factor (32) and within [min,max] px.
        // tokens = 1024*1024 / 32^2 = 1024.
        assert_eq!(estimate_image_tokens(1024, 1024, &QWEN3VL), 1024);
        // 1280x768 -> 1280*768 / 1024 = 960.
        assert_eq!(estimate_image_tokens(1280, 768, &QWEN3VL), 960);
    }

    #[test]
    fn rounds_each_side_to_factor() {
        // 1000x1000 rounds to 992x992 (round(1000/32)=31 -> 992).
        // tokens = 992*992 / 1024 = 961.
        assert_eq!(estimate_image_tokens(1000, 1000, &QWEN3VL), 961);
    }

    #[test]
    fn tiny_image_is_floored_to_min_pixels() {
        // 16x16 is far below min_pixels (65536); smart_resize scales it up so the
        // token count is floored, not ~0.
        let tokens = estimate_image_tokens(16, 16, &QWEN3VL);
        assert!(tokens >= QWEN3VL.min_pixels / (QWEN3VL.factor() * QWEN3VL.factor()) - 1);
        assert!(tokens >= 60, "expected a floored count, got {tokens}");
    }

    #[test]
    fn huge_image_is_capped_to_max_pixels() {
        // 10000x10000 = 1e8 px is far above max_pixels (16.7M); the token count
        // saturates near max_pixels/factor^2 = 16384 rather than ~1e8/1024.
        let tokens = estimate_image_tokens(10_000, 10_000, &QWEN3VL);
        let cap = QWEN3VL.max_pixels / (QWEN3VL.factor() * QWEN3VL.factor());
        assert!(tokens <= cap, "{tokens} should be <= cap {cap}");
        assert!(tokens > cap * 9 / 10, "{tokens} should be near cap {cap}");
    }

    #[test]
    fn dimensions_parsed_from_data_uri() {
        let uri = png_data_uri(640, 480);
        assert_eq!(dimensions_from_data_uri(&uri), Some((640, 480)));
    }

    #[test]
    fn http_url_yields_no_dimensions() {
        assert_eq!(
            dimensions_from_data_uri("https://example.com/cat.png"),
            None
        );
    }

    #[test]
    fn weight_is_text_only_without_images() {
        let request = PreprocessedRequest::builder()
            .model("test".to_string())
            .token_ids(vec![1, 2, 3, 4, 5])
            .stop_conditions(Default::default())
            .sampling_options(Default::default())
            .output_options(Default::default())
            .build()
            .unwrap();
        assert_eq!(prefill_token_weight(&request), 5);
    }

    #[test]
    fn weight_adds_image_token_estimate() {
        let mut mm = std::collections::HashMap::new();
        let uri = url::Url::parse(&png_data_uri(1024, 1024)).unwrap();
        mm.insert("image_url".to_string(), vec![MultimodalData::Url(uri)]);

        let request = PreprocessedRequest::builder()
            .model("test".to_string())
            .token_ids(vec![1, 2, 3]) // 3 text/placeholder tokens
            .multi_modal_data(Some(mm))
            .stop_conditions(Default::default())
            .sampling_options(Default::default())
            .output_options(Default::default())
            .build()
            .unwrap();

        // 3 text tokens + 1024 image tokens.
        assert_eq!(prefill_token_weight(&request), 3 + 1024);
    }

    #[test]
    fn expanded_routing_sequence_takes_precedence() {
        use crate::protocols::common::preprocessor::MmRoutingInfo;

        let request = PreprocessedRequest::builder()
            .model("test".to_string())
            .token_ids(vec![1, 2, 3])
            .mm_routing_info(Some(MmRoutingInfo {
                routing_token_ids: vec![0; 777],
                block_mm_infos: vec![],
            }))
            .stop_conditions(Default::default())
            .sampling_options(Default::default())
            .output_options(Default::default())
            .build()
            .unwrap();

        // Uses the already-expanded routing sequence length, not token_ids.
        assert_eq!(prefill_token_weight(&request), 777);
    }
}
