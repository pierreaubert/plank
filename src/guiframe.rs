//! Bounded graphical frame snapshots and input shared with the client.

use serde::{Deserialize, Serialize};

/// Largest supported character grid, bounding wire size and UI work per tick.
pub const MAX_WIDTH: u16 = 160;
/// Maximum supported grid height.
pub const MAX_HEIGHT: u16 = 64;
/// Maximum committed text bytes per event, bounding queued input memory.
pub const MAX_TEXT_BYTES: usize = 512;
/// Maximum typed characters per event, bounding guest calls before the next tick.
pub const MAX_TEXT_CHARS: usize = 128;

/// Validates a single-line Unicode text commitment before guest dispatch.
#[must_use]
pub fn valid_text(text: &str) -> bool {
    !text.is_empty()
        && text.len() <= MAX_TEXT_BYTES
        && text.chars().count() <= MAX_TEXT_CHARS
        && !text.chars().any(char::is_control)
}

/// A frame cell with its original foreground and optional background.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Cell {
    pub x: u16,
    pub y: u16,
    pub ch: char,
    pub fg: [u8; 3],
    pub bg: Option<[u8; 3]>,
    pub bold: bool,
}

/// One complete, transient frame; never conversation text or replayed input.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Snapshot {
    pub id: String,
    pub component: String,
    pub w: u16,
    pub h: u16,
    pub cells: Vec<Cell>,
}

impl std::fmt::Debug for Snapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FrameSnapshot")
            .field("id", &self.id)
            .field("size", &(self.w, self.h))
            .field("cells", &self.cells.len())
            .finish_non_exhaustive()
    }
}

fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 128
        && id
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_' | b'.'))
}

fn valid_size(w: u16, h: u16) -> bool {
    w > 0 && h > 0 && w <= MAX_WIDTH && h <= MAX_HEIGHT
}

impl Snapshot {
    /// Validates a grid before it is published or displayed.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        valid_id(&self.id)
            && valid_id(&self.component)
            && valid_size(self.w, self.h)
            // Match the packed decoder's allowance for overlapping glyphs.
            && self.cells.len() <= usize::from(self.w) * usize::from(self.h) + 512
            && self
                .cells
                .iter()
                .all(|c| c.x < self.w && c.y < self.h && !c.ch.is_control())
    }
}

/// An input event scoped to one open frame identity.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Request {
    pub id: String,
    #[serde(flatten)]
    pub input: Input,
}

impl std::fmt::Debug for Request {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FrameRequest")
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}

/// Keyboard, mouse, viewport, and close controls understood by the host.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum Input {
    Key { code: String, text: Option<char> },
    Text { text: String },
    Mouse { kind: String, x: u16, y: u16 },
    Resize { w: u16, h: u16 },
    Close,
}

impl std::fmt::Debug for Input {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::Key { .. } => "Key",
            Self::Text { .. } => "Text",
            Self::Mouse { .. } => "Mouse",
            Self::Resize { .. } => "Resize",
            Self::Close => "Close",
        };
        f.write_str(name)
    }
}

impl Request {
    /// Rejects oversized keys and coordinates before queuing an input.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        valid_id(&self.id)
            && match &self.input {
                Input::Key { code, text } => {
                    !code.is_empty()
                        && code.len() <= 64
                        && !code.chars().any(char::is_control)
                        && text.is_none_or(|c| !c.is_control())
                }
                Input::Text { text } => valid_text(text),
                Input::Mouse { kind, x, y } => {
                    matches!(
                        kind.as_str(),
                        "down" | "up" | "drag" | "move" | "scroll_up" | "scroll_down"
                    ) && *x < MAX_WIDTH
                        && *y < MAX_HEIGHT
                }
                Input::Resize { w, h } => valid_size(*w, *h),
                Input::Close => true,
            }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn graphical_frame_text_is_bounded_unicode_and_redacted() {
        let request = Request {
            id: "frame-text".into(),
            input: Input::Text {
                text: "日本語 e\u{301} 🦀".into(),
            },
        };
        assert!(request.is_valid());
        assert_eq!(
            serde_json::from_str::<Request>(&serde_json::to_string(&request).unwrap()).unwrap(),
            request
        );
        assert!(!format!("{request:?} {:?}", request.input).contains("日本語"));
        for text in [
            String::new(),
            "\n".into(),
            "\t".into(),
            "x\u{1b}".into(),
            "a".repeat(MAX_TEXT_CHARS + 1),
        ] {
            assert!(!valid_text(&text));
        }
        assert!(valid_text(&"🦀".repeat(MAX_TEXT_CHARS)));
        assert!(!valid_text(&"🦀".repeat(MAX_TEXT_CHARS + 1)));
    }

    #[test]
    fn graphical_frames_validate_bounds_and_redact_visible_content_and_keys() {
        let mut frame = Snapshot {
            id: "frame-1".into(),
            component: "dev.plank.test".into(),
            w: 2,
            h: 1,
            cells: vec![Cell {
                x: 1,
                y: 0,
                ch: '🦀',
                fg: [1, 2, 3],
                bg: Some([4, 5, 6]),
                bold: true,
            }],
        };
        assert!(frame.is_valid());
        assert_eq!(
            serde_json::from_str::<Snapshot>(&serde_json::to_string(&frame).unwrap()).unwrap(),
            frame
        );
        assert!(!format!("{frame:?}").contains('🦀'));
        frame.cells[0].x = 2;
        assert!(!frame.is_valid());
        let mut request = Request {
            id: "frame-1".into(),
            input: Input::Key {
                code: "secret".into(),
                text: Some('s'),
            },
        };
        assert!(request.is_valid());
        assert!(!format!("{request:?}").contains("secret"));
        request.input = Input::Resize {
            w: MAX_WIDTH + 1,
            h: 1,
        };
        assert!(!request.is_valid());
    }
}
