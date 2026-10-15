//! Bounded PNG snapshots for graphical buddy playback outside the transcript.

// Rust guideline compliant 2026-02-21
use serde::{Deserialize, Serialize};

/// Fits an escaped JSON frame comfortably inside the console's 2 MiB limit.
pub const MAX_BYTES: usize = 256 * 1024;
/// Caps a decoded frame at four MiB of RGBA pixels.
pub const MAX_EDGE: u32 = 1024;

/// The latest animation frame in one engine generation.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Frame {
    pub revision: u64,
    pub png: Vec<u8>,
}

impl std::fmt::Debug for Frame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BuddyFrame")
            .field("revision", &self.revision)
            .field("bytes", &self.png.len())
            .finish()
    }
}

impl Frame {
    /// Validates transport size and dimensions before allocating decoded pixels.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.revision != 0 && self.png.len() <= MAX_BYTES && self.dimensions().is_some()
    }

    /// Reads the required PNG header; full decoding still validates the image.
    #[must_use]
    pub fn dimensions(&self) -> Option<(u32, u32)> {
        if self.png.get(..16)? != b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR" {
            return None;
        }
        let width = u32::from_be_bytes(self.png.get(16..20)?.try_into().ok()?);
        let height = u32::from_be_bytes(self.png.get(20..24)?.try_into().ok()?);
        ((1..=MAX_EDGE).contains(&width) && (1..=MAX_EDGE).contains(&height))
            .then_some((width, height))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graphical_buddy_frames_bound_dimensions_payload_and_debug_output() {
        let mut frame = Frame {
            revision: 1,
            png: b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR".to_vec(),
        };
        frame.png.extend(24_u32.to_be_bytes());
        frame.png.extend(10_u32.to_be_bytes());
        assert!(frame.is_valid());
        assert_eq!(frame.dimensions(), Some((24, 10)));
        assert!(!format!("{frame:?}").contains("IHDR"));
        frame.revision = 0;
        assert!(!frame.is_valid());
        frame.revision = 1;
        frame.png[16..20].copy_from_slice(&(MAX_EDGE + 1).to_be_bytes());
        assert!(!frame.is_valid());
        frame.png[16..20].copy_from_slice(&24_u32.to_be_bytes());
        frame.png.resize(MAX_BYTES + 1, 0);
        assert!(!frame.is_valid());
    }
}
