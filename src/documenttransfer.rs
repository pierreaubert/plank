// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Bounded, ordered UTF-8 document transfers with explicit completion.
//!
//! Callers bind one receiver to an editor identity and an authorization grant.
//! Partial transfers never yield document text; cancellation drops the receiver.

// Rust guideline compliant 2026-02-21
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::borrow::Cow;

/// Maximum document size, matching Plank's text-file editor.
pub const DOCUMENT_BYTES: usize = 32 * 1024 * 1024;
/// Maximum UTF-8 payload per frame before JSON escaping.
pub const CHUNK_BYTES: usize = 32 * 1024;

/// Computes a lowercase SHA-256 checksum for document bytes.
#[must_use]
pub fn digest(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

/// One frame inside an editor-scoped transfer.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
pub enum Frame {
    /// Announces the exact length and lowercase SHA-256 digest.
    Begin {
        /// Opaque identity, unique for each Save attempt.
        transfer: String,
        /// Expected document length in UTF-8 bytes.
        total: usize,
        /// Lowercase hexadecimal SHA-256 digest.
        sha256: String,
    },
    /// Appends the next contiguous UTF-8 segment.
    Chunk {
        /// Identity announced by Begin.
        transfer: String,
        /// Byte offset of this segment in the complete document.
        offset: usize,
        /// Nonempty text bounded by [`CHUNK_BYTES`].
        text: String,
    },
    /// Requests completion after every byte has arrived.
    End {
        /// Identity announced by Begin.
        transfer: String,
    },
}

impl std::fmt::Debug for Frame {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Begin { total, .. } => formatter.debug_tuple("Begin").field(total).finish(),
            Self::Chunk { offset, text, .. } => formatter
                .debug_struct("Chunk")
                .field("offset", offset)
                .field("bytes", &text.len())
                .finish_non_exhaustive(),
            Self::End { .. } => formatter.write_str("End"),
        }
    }
}

struct Pending {
    transfer: String,
    total: usize,
    sha256: String,
    hash: Sha256,
    text: String,
}

/// One bounded, incomplete transfer, independent of its transport.
#[derive(Default)]
pub struct Receiver {
    pending: Option<Pending>,
}

impl std::fmt::Debug for Receiver {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Receiver")
            .field(
                "received",
                &self.pending.as_ref().map(|pending| pending.text.len()),
            )
            .finish_non_exhaustive()
    }
}

impl Receiver {
    /// Accepts a frame, returning text only after validated completion.
    ///
    /// A new valid Begin replaces a partial attempt. Frames for other attempts
    /// leave the current attempt unchanged. Invalid chunks or completion for
    /// the current attempt discard it, requiring a fresh Begin.
    ///
    /// # Errors
    /// Returns a static diagnostic for invalid identities, bounds, ordering or checksums.
    pub fn accept(&mut self, frame: &Frame) -> Result<Option<String>, &'static str> {
        if let Frame::Begin {
            transfer,
            total,
            sha256,
        } = frame
        {
            if !valid_id(transfer) || *total > DOCUMENT_BYTES || !valid_hash(sha256) {
                return Err("Invalid document transfer announcement");
            }
            self.pending = Some(Pending {
                transfer: transfer.clone(),
                total: *total,
                sha256: sha256.clone(),
                hash: Sha256::new(),
                text: String::new(),
            });
            return Ok(None);
        }
        let Some(pending) = self.pending.as_mut() else {
            return Err("Document transfer has not begun");
        };
        match frame {
            Frame::Chunk {
                transfer,
                offset,
                text,
            } => {
                if transfer != &pending.transfer {
                    return Err("Document chunk belongs to another transfer");
                }
                if *offset != pending.text.len()
                    || text.is_empty()
                    || text.len() > CHUNK_BYTES
                    || text.len() > pending.total.saturating_sub(pending.text.len())
                {
                    self.pending = None;
                    return Err("Invalid document chunk length or offset");
                }
                pending.hash.update(text.as_bytes());
                // Grow geometrically, capped by the announced length; exact
                // per-chunk growth could repeatedly copy the whole document.
                let required = pending.text.len() + text.len();
                if required > pending.text.capacity() {
                    let capacity = pending
                        .text
                        .capacity()
                        .saturating_mul(2)
                        .max(required)
                        .min(pending.total);
                    pending.text.reserve_exact(capacity - pending.text.len());
                }
                pending.text.push_str(text);
                Ok(None)
            }
            Frame::End { transfer } => {
                if transfer != &pending.transfer {
                    return Err("Document completion belongs to another transfer");
                }
                let pending = self
                    .pending
                    .take()
                    .ok_or("Document transfer has not begun")?;
                if pending.text.len() != pending.total
                    || hex(&pending.hash.finalize()) != pending.sha256
                {
                    return Err("Incomplete or corrupt document transfer");
                }
                Ok(Some(pending.text))
            }
            Frame::Begin { .. } => Err("Unexpected document announcement"),
        }
    }
}

/// Lazily encodes one document without collecting its frames in memory.
pub struct Sender<'a> {
    transfer: Cow<'a, str>,
    text: Cow<'a, str>,
    offset: usize,
    stage: u8,
}

impl std::fmt::Debug for Sender<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Sender")
            .field("bytes", &self.text.len())
            .finish_non_exhaustive()
    }
}

impl<'a> Sender<'a> {
    /// Creates a bounded frame iterator over borrowed document text.
    ///
    /// # Errors
    /// Rejects invalid transfer identities or documents larger than [`DOCUMENT_BYTES`].
    pub fn new(transfer: &'a str, text: &'a str) -> Result<Self, &'static str> {
        if !valid_id(transfer) || text.len() > DOCUMENT_BYTES {
            return Err("Invalid document transfer identity or size");
        }
        Ok(Self {
            transfer: Cow::Borrowed(transfer),
            text: Cow::Borrowed(text),
            offset: 0,
            stage: 0,
        })
    }
}

impl Sender<'static> {
    /// Creates an owned iterator for asynchronous transport queues.
    ///
    /// # Errors
    /// Rejects invalid transfer identities or documents larger than [`DOCUMENT_BYTES`].
    pub fn owned(transfer: String, text: String) -> Result<Self, &'static str> {
        if !valid_id(&transfer) || text.len() > DOCUMENT_BYTES {
            return Err("Invalid document transfer identity or size");
        }
        Ok(Self {
            transfer: Cow::Owned(transfer),
            text: Cow::Owned(text),
            offset: 0,
            stage: 0,
        })
    }
}

impl Iterator for Sender<'_> {
    type Item = Frame;

    fn next(&mut self) -> Option<Frame> {
        if self.stage == 0 {
            self.stage = 1;
            return Some(Frame::Begin {
                transfer: self.transfer.to_string(),
                total: self.text.len(),
                sha256: digest(self.text.as_bytes()),
            });
        }
        if self.stage == 1 && self.offset < self.text.len() {
            let mut end = (self.offset + CHUNK_BYTES).min(self.text.len());
            while !self.text.is_char_boundary(end) {
                end -= 1;
            }
            let frame = Frame::Chunk {
                transfer: self.transfer.to_string(),
                offset: self.offset,
                text: self.text[self.offset..end].into(),
            };
            self.offset = end;
            return Some(frame);
        }
        if self.stage == 1 {
            self.stage = 2;
            return Some(Frame::End {
                transfer: self.transfer.to_string(),
            });
        }
        None
    }
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8] = b"0123456789abcdef";
    bytes
        .iter()
        .flat_map(|byte| {
            [
                char::from(DIGITS[usize::from(byte >> 4)]),
                char::from(DIGITS[usize::from(byte & 15)]),
            ]
        })
        .collect()
}

fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 128
        && id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
}

fn valid_hash(hash: &str) -> bool {
    hash.len() == 64
        && hash
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn document_digest_matches_the_standard_empty_sha256() {
        assert_eq!(
            digest(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn document_transfers_preserve_unicode_escaping_and_empty_documents() {
        for text in [String::new(), "\"\\\u{0}\r\n🦀é".repeat(50_000)] {
            let mut receiver = Receiver::default();
            let mut result = None;
            for frame in Sender::new("save-1", &text).unwrap() {
                let encoded = serde_json::to_string(&frame).unwrap();
                assert!(encoded.len() < CHUNK_BYTES * 6 + 1024);
                let decoded = serde_json::from_str(&encoded).unwrap();
                let next = receiver.accept(&decoded).unwrap();
                assert!(result.is_none());
                result = next;
            }
            assert_eq!(result.as_deref(), Some(text.as_str()));
            assert!(
                receiver
                    .accept(&Frame::End {
                        transfer: "save-1".into()
                    })
                    .is_err()
            );
        }
    }

    #[test]
    fn document_transfers_refuse_incomplete_corrupt_and_out_of_order_data() {
        let text = "private document";
        let frames: Vec<_> = Sender::new("save-1", text).unwrap().collect();
        let mut receiver = Receiver::default();
        receiver.accept(&frames[0]).unwrap();
        assert!(receiver.accept(&frames[2]).is_err());
        receiver.accept(&frames[0]).unwrap();
        assert!(
            receiver
                .accept(&Frame::Chunk {
                    transfer: "save-1".into(),
                    offset: 1,
                    text: text.into()
                })
                .is_err()
        );
        assert!(receiver.accept(&frames[1]).is_err());
        receiver.accept(&frames[0]).unwrap();
        receiver
            .accept(&Frame::Chunk {
                transfer: "save-1".into(),
                offset: 0,
                text: "changed document".into(),
            })
            .unwrap();
        assert!(receiver.accept(&frames[2]).is_err());
        assert!(!format!("{:?}", frames[1]).contains(text));
    }

    #[test]
    fn document_transfers_accept_the_file_limit_and_reject_larger_documents() {
        let mut text = "x".repeat(DOCUMENT_BYTES + 1);
        assert!(Sender::new("limit", &text).is_err());
        text.pop();
        let mut receiver = Receiver::default();
        let mut complete = None;
        for frame in Sender::new("limit", &text).unwrap() {
            complete = receiver.accept(&frame).unwrap();
        }
        assert!(
            complete.as_deref().is_some_and(|actual| actual == text),
            "Transfer must preserve the complete document"
        );
    }

    #[test]
    fn document_transfers_ignore_stale_attempts_and_enforce_bounds() {
        let frames: Vec<_> = Sender::new("current", "ok").unwrap().collect();
        let mut receiver = Receiver::default();
        receiver.accept(&frames[0]).unwrap();
        assert!(
            receiver
                .accept(&Frame::End {
                    transfer: "old".into()
                })
                .is_err()
        );
        receiver.accept(&frames[1]).unwrap();
        assert_eq!(receiver.accept(&frames[2]).unwrap(), Some("ok".into()));
        for frame in [
            Frame::Begin {
                transfer: "x".into(),
                total: DOCUMENT_BYTES + 1,
                sha256: "0".repeat(64),
            },
            Frame::Begin {
                transfer: "bad id".into(),
                total: 0,
                sha256: "0".repeat(64),
            },
            Frame::Begin {
                transfer: "x".into(),
                total: 0,
                sha256: "Z".repeat(64),
            },
        ] {
            assert!(receiver.accept(&frame).is_err());
        }
        assert!(
            serde_json::from_str::<Frame>(r#"{"operation":"end","transfer":"x","extra":true}"#)
                .is_err()
        );
        receiver
            .accept(&Frame::Begin {
                transfer: "large-chunk".into(),
                total: CHUNK_BYTES + 1,
                sha256: "0".repeat(64),
            })
            .unwrap();
        assert!(
            receiver
                .accept(&Frame::Chunk {
                    transfer: "large-chunk".into(),
                    offset: 0,
                    text: "x".repeat(CHUNK_BYTES + 1),
                })
                .is_err()
        );
    }
}
