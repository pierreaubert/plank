//! Streams complete UTF-8 exports in bounded frames without retaining their bodies.
//!
//! Offsets and the final byte count detect missing or reordered chunks. The
//! SHA-256 in the begin frame lets a receiver verify the assembled file before
//! exposing it. Transfers are transient: reconnecting receivers must request a
//! new export rather than accepting a partial replay.

// Rust guideline compliant 2026-02-21
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Maximum unescaped UTF-8 payload per chunk.
///
/// Worst-case JSON escaping stays well below the client's 2 MiB frame limit.
pub const CHUNK_BYTES: usize = 32 * 1024;

/// Identifies the view that owns a complete-file artifact.
#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Purpose {
    /// A conversation exported with `/export`.
    #[default]
    Conversation,
    /// The complete standalone Insights report.
    Insights,
}

/// One event in a complete export transfer.
#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "phase", rename_all = "snake_case")]
pub enum Transfer {
    /// Starts an export; the name is a suggested basename, never a host path.
    Begin {
        /// Artifact category; older frames are conversation exports.
        #[serde(default)]
        purpose: Purpose,
        /// Identity unique within the owning agent channel.
        id: String,
        /// Suggested client filename.
        name: String,
        /// `md` or `html`.
        format: String,
        /// Exact expected UTF-8 byte count.
        bytes: u64,
        /// Lowercase SHA-256 of the complete file.
        sha256: String,
    },
    /// Carries a contiguous piece of the rendered file.
    Chunk {
        /// Owning transfer identity.
        id: String,
        /// Expected byte offset in the destination.
        offset: u64,
        /// UTF-8 file bytes, preserved verbatim.
        text: String,
    },
    /// Completes the transfer after all bytes were emitted.
    End {
        /// Owning transfer identity.
        id: String,
        /// Exact emitted byte count.
        bytes: u64,
    },
    /// Retires a partial transfer without affecting the engine conversation.
    Abort { id: String, reason: String },
}

impl Transfer {
    /// Checks bounded metadata and chunks before queue allocation.
    pub(crate) fn is_valid(&self) -> bool {
        let valid_id = |id: &str| {
            !id.is_empty()
                && id.len() <= 64
                && id
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
        };
        match self {
            Self::Begin {
                purpose,
                id,
                name,
                format,
                sha256,
                ..
            } => {
                valid_id(id)
                    && !name.is_empty()
                    && name.len() <= 255
                    && name != "."
                    && name != ".."
                    && !name
                        .chars()
                        .any(|c| c.is_control() || matches!(c, '/' | '\\'))
                    && matches!(format.as_str(), "md" | "html")
                    && (*purpose != Purpose::Insights || format == "html")
                    && sha256.len() == 64
                    && sha256
                        .bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            }
            Self::Chunk { id, text, .. } => {
                valid_id(id) && !text.is_empty() && text.len() <= CHUNK_BYTES
            }
            Self::End { id, .. } => valid_id(id),
            Self::Abort { id, reason } => {
                valid_id(id) && reason.len() <= 512 && !reason.chars().any(char::is_control)
            }
        }
    }
}

impl std::fmt::Debug for Transfer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Begin { id, bytes, .. } => f
                .debug_struct("ExportBegin")
                .field("id", id)
                .field("bytes", bytes)
                .finish_non_exhaustive(),
            Self::Chunk { id, offset, text } => f
                .debug_struct("ExportChunk")
                .field("id", id)
                .field("offset", offset)
                .field("bytes", &text.len())
                .finish_non_exhaustive(),
            Self::End { id, bytes } => f
                .debug_struct("ExportEnd")
                .field("id", id)
                .field("bytes", bytes)
                .finish(),
            Self::Abort { id, .. } => f
                .debug_struct("ExportAbort")
                .field("id", id)
                .finish_non_exhaustive(),
        }
    }
}

/// Emits one export incrementally, stopping immediately if its receiver fails.
pub(crate) fn emit<E>(
    purpose: Purpose,
    id: &str,
    name: &str,
    format: super::Format,
    body: &str,
    mut send: impl FnMut(Transfer) -> Result<(), E>,
) -> Result<(), E> {
    let id = id.to_owned();
    // Host path components and control characters cannot become download names.
    let name = name.rsplit(['/', '\\']).next().unwrap_or("");
    let name = if name.is_empty()
        || name.len() > 255
        || matches!(name, "." | "..")
        || name.chars().any(char::is_control)
    {
        format!("session.{}", format.extension())
    } else {
        name.to_owned()
    };
    let bytes = body.len() as u64;
    send(Transfer::Begin {
        purpose,
        id: id.clone(),
        name,
        format: format.extension().into(),
        bytes,
        sha256: crate::downloader::hex(&Sha256::digest(body.as_bytes())),
    })?;
    let mut offset = 0;
    while offset < body.len() {
        let mut end = offset.saturating_add(CHUNK_BYTES).min(body.len());
        while !body.is_char_boundary(end) {
            end -= 1;
        }
        send(Transfer::Chunk {
            id: id.clone(),
            offset: offset as u64,
            text: body[offset..end].into(),
        })?;
        offset = end;
    }
    send(Transfer::End { id, bytes })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn export_transfer_preserves_large_unicode_files_and_bounds_each_frame() {
        // Includes controls requiring six-byte JSON escapes and split-edge Unicode.
        let body = format!(
            "{}🦀\r\n{}",
            "a".repeat(CHUNK_BYTES - 1),
            "\u{0001}β🦀".repeat(500_000)
        );
        assert!(body.len() > 2 * 1024 * 1024);
        let mut output = String::new();
        let mut identity = None;
        let mut ended = false;
        emit(
            Purpose::Conversation,
            "export-large",
            "/private/repo/transcript.html",
            super::super::Format::Html,
            &body,
            |event| {
                assert!(serde_json::to_vec(&event).unwrap().len() < 256 * 1024);
                match event {
                    Transfer::Begin {
                        purpose: Purpose::Conversation,
                        id,
                        name,
                        format,
                        bytes,
                        sha256,
                    } => {
                        assert_eq!(name, "transcript.html");
                        assert_eq!(format, "html");
                        assert_eq!(bytes, body.len() as u64);
                        assert_eq!(
                            sha256,
                            crate::downloader::hex(&Sha256::digest(body.as_bytes()))
                        );
                        identity = Some(id);
                    }
                    Transfer::Begin { .. } => panic!("unexpected artifact purpose"),
                    Transfer::Chunk { id, offset, text } => {
                        assert_eq!(Some(id), identity);
                        assert_eq!(offset, output.len() as u64);
                        assert!(text.len() <= CHUNK_BYTES);
                        output.push_str(&text);
                    }
                    Transfer::End { id, bytes } => {
                        assert_eq!(Some(id), identity);
                        assert_eq!(bytes, output.len() as u64);
                        ended = true;
                    }
                    Transfer::Abort { .. } => panic!("encoder cannot abort a successful transfer"),
                }
                Ok::<_, ()>(())
            },
        )
        .unwrap();
        assert_eq!(output, body);
        assert!(ended);
    }

    #[test]
    fn export_transfer_stops_on_consumer_failure_without_emitting_completion() {
        let mut calls = 0;
        let result = emit(
            Purpose::Conversation,
            "export-failure",
            "test.md",
            super::super::Format::Markdown,
            &"x".repeat(CHUNK_BYTES * 3),
            |event| {
                calls += 1;
                assert!(!matches!(event, Transfer::End { .. }));
                if calls == 2 {
                    Err("consumer closed")
                } else {
                    Ok(())
                }
            },
        );
        assert_eq!(result, Err("consumer closed"));
        assert_eq!(calls, 2);
    }

    #[test]
    fn export_transfer_empty_files_complete_and_debug_redacts_contents() {
        let mut events = Vec::new();
        emit(
            Purpose::Conversation,
            "export-empty",
            "unsafe\nname",
            super::super::Format::Markdown,
            "",
            |event| {
                events.push(event);
                Ok::<_, ()>(())
            },
        )
        .unwrap();
        assert_eq!(events.len(), 2);
        assert!(
            matches!(&events[0], Transfer::Begin { name, bytes: 0, .. } if name == "session.md")
        );
        assert!(matches!(&events[1], Transfer::End { bytes: 0, .. }));
        assert!(matches!(&events[0], Transfer::Begin { sha256, .. }
            if sha256 == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"));
        let chunk = Transfer::Chunk {
            id: "export-1".into(),
            offset: 0,
            text: "secret transcript".into(),
        };
        assert!(!format!("{chunk:?}").contains("secret"));
    }
}
