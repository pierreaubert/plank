// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Matches explicit console and remote edits to a pending editor request.

// Rust guideline compliant 2026-02-21
use std::sync::Mutex;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Writer {
    Console,
    Remote(super::control::ControlStamp),
}

/// Keeps editor payloads within the shared native/browser editing limit.
pub(crate) const DOCUMENT_BYTES: usize = 256 * 1024;

#[derive(Debug)]
struct Pending {
    id: String,
    reply: Option<EditReply>,
    transfer: Option<(Writer, crate::documenttransfer::Receiver)>,
}

/// An explicit Save or Cancel, distinct from a request still awaiting input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum EditReply {
    Save(String),
    Cancel,
}

impl EditReply {
    pub(crate) fn into_text(self) -> Option<String> {
        match self {
            Self::Save(text) => Some(text),
            Self::Cancel => None,
        }
    }
}

/// A single editor shared by the host UI and its authenticated controller.
#[derive(Debug, Default)]
pub(crate) struct Editors(Mutex<Option<Pending>>);

impl Editors {
    pub(crate) fn open(&self, id: &str) {
        *self.0.lock().expect("editor slot poisoned") = Some(Pending {
            id: id.into(),
            reply: None,
            transfer: None,
        });
    }

    pub(crate) fn reply(&self, id: &str, text: &serde_json::Value) -> bool {
        let reply = match text {
            serde_json::Value::Null => EditReply::Cancel,
            serde_json::Value::String(text) if text.len() <= DOCUMENT_BYTES => {
                EditReply::Save(text.clone())
            }
            _ => return false,
        };
        let mut slot = self.0.lock().expect("editor slot poisoned");
        let Some(pending) = slot
            .as_mut()
            .filter(|pending| pending.id == id && pending.reply.is_none())
        else {
            return false;
        };
        pending.reply = Some(reply);
        pending.transfer = None;
        true
    }

    /// Accepts chunked Saves only within their original controller grant.
    pub(crate) fn reply_from(
        &self,
        owner: super::control::ControlStamp,
        id: &str,
        text: &serde_json::Value,
    ) -> bool {
        self.reply_owned(Writer::Remote(owner), id, text)
    }

    /// Accepts pipe-owned Saves without mixing them with a remote transfer.
    pub(crate) fn reply_console(&self, id: &str, text: &serde_json::Value) -> bool {
        self.reply_owned(Writer::Console, id, text)
    }

    fn reply_owned(&self, owner: Writer, id: &str, text: &serde_json::Value) -> bool {
        if !text.is_object() {
            return self.reply(id, text);
        }
        let Ok(frame) = serde_json::from_value::<crate::documenttransfer::Frame>(text.clone())
        else {
            return false;
        };
        let mut slot = self.0.lock().expect("editor slot poisoned");
        let Some(pending) = slot
            .as_mut()
            .filter(|pending| pending.id == id && pending.reply.is_none())
        else {
            return false;
        };
        if matches!(frame, crate::documenttransfer::Frame::Begin { .. }) {
            // A new controller must begin its own attempt; it cannot finish
            // bytes accepted under a revoked grant, even after a same-ID regrant.
            if pending
                .transfer
                .as_ref()
                .is_none_or(|(stamp, _)| *stamp != owner)
            {
                pending.transfer = Some((owner, crate::documenttransfer::Receiver::default()));
            }
        }
        let Some((_, receiver)) = pending
            .transfer
            .as_mut()
            .filter(|(stamp, _)| *stamp == owner)
        else {
            return false;
        };
        match receiver.accept(&frame) {
            Ok(Some(text)) => {
                pending.reply = Some(EditReply::Save(text));
                pending.transfer = None;
                true
            }
            Ok(None) => true,
            Err(_) => false,
        }
    }

    pub(crate) fn take_reply(&self, id: &str) -> Option<EditReply> {
        let mut slot = self.0.lock().expect("editor slot poisoned");
        if slot
            .as_ref()
            .is_some_and(|pending| pending.id == id && pending.reply.is_some())
        {
            slot.take().and_then(|pending| pending.reply)
        } else {
            None
        }
    }

    pub(crate) fn close(&self, id: &str) {
        let _ = self.finish(id);
    }

    /// Closes atomically while retaining an already accepted reply.
    pub(crate) fn finish(&self, id: &str) -> Option<EditReply> {
        let mut slot = self.0.lock().expect("editor slot poisoned");
        if slot.as_ref().is_some_and(|pending| pending.id == id) {
            slot.take().and_then(|pending| pending.reply)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remote_editors_finish_keeps_an_accepted_reply_and_retires_identity() {
        let editors = Editors::default();
        editors.open("e1");
        assert!(editors.reply("e1", &serde_json::json!("accepted")));
        assert!(editors.finish("wrong").is_none());
        assert_eq!(
            editors.finish("e1"),
            Some(EditReply::Save("accepted".into()))
        );
        assert!(editors.finish("e1").is_none());
        assert!(!editors.reply("e1", &serde_json::Value::Null));
    }

    #[test]
    fn remote_editors_never_mix_console_bytes_with_remote_grants() {
        let editors = Editors::default();
        let mut control = super::super::control::ControlPolicy::new(false, false);
        control.request(7);
        let owner = control.stamp(7).unwrap();
        let local: Vec<_> = crate::documenttransfer::Sender::new("local-save", "local draft")
            .unwrap()
            .map(|frame| serde_json::to_value(frame).unwrap())
            .collect();
        let remote: Vec<_> = crate::documenttransfer::Sender::new("remote-save", "remote draft")
            .unwrap()
            .map(|frame| serde_json::to_value(frame).unwrap())
            .collect();
        editors.open("e1");
        assert!(editors.reply_console("e1", &local[0]));
        assert!(!editors.reply_from(owner, "e1", &local[1]));
        assert!(editors.reply_from(owner, "e1", &remote[0]));
        assert!(!editors.reply_console("e1", &remote[1]));
        assert!(editors.take_reply("e1").is_none());
        for frame in &local {
            assert!(editors.reply_console("e1", frame));
        }
        assert_eq!(
            editors.take_reply("e1"),
            Some(EditReply::Save("local draft".into()))
        );
    }

    #[test]
    fn remote_editors_partial_transfers_never_survive_cancel_close_or_replacement() {
        let editors = Editors::default();
        let mut control = super::super::control::ControlPolicy::new(false, false);
        control.request(7);
        let owner = control.stamp(7).unwrap();
        let frames: Vec<_> = crate::documenttransfer::Sender::new("save-1", "private draft")
            .unwrap()
            .map(|frame| serde_json::to_value(frame).unwrap())
            .collect();
        for action in ["cancel", "close", "replace"] {
            editors.open("e1");
            assert!(editors.reply_from(owner, "e1", &frames[0]));
            assert!(editors.reply_from(owner, "e1", &frames[1]));
            assert!(editors.take_reply("e1").is_none());
            match action {
                "cancel" => {
                    assert!(editors.reply_from(owner, "e1", &serde_json::Value::Null));
                    assert_eq!(editors.take_reply("e1"), Some(EditReply::Cancel));
                }
                "close" => editors.close("e1"),
                _ => editors.open("e2"),
            }
            assert!(!editors.reply_from(owner, "e1", &frames[2]));
            assert!(editors.take_reply("e1").is_none());
            assert!(!format!("{editors:?}").contains("private draft"));
        }
    }

    #[test]
    fn remote_editors_require_current_id_bounded_text_or_explicit_cancel() {
        let editors = Editors::default();
        editors.open("e1");
        assert!(editors.take_reply("e1").is_none());
        for (id, text) in [
            ("old", serde_json::json!("text")),
            ("e1", serde_json::json!(true)),
            ("e1", serde_json::json!({})),
            ("e1", serde_json::json!("x".repeat(DOCUMENT_BYTES + 1))),
        ] {
            assert!(!editors.reply(id, &text));
        }
        assert!(editors.reply("e1", &serde_json::json!("café\r\n🦀\n")));
        assert!(!editors.reply("e1", &serde_json::Value::Null));
        assert_eq!(
            editors.take_reply("e1"),
            Some(EditReply::Save("café\r\n🦀\n".into()))
        );
        assert!(editors.take_reply("e1").is_none());
        editors.open("e2");
        assert!(editors.reply("e2", &serde_json::Value::Null));
        assert_eq!(editors.take_reply("e2"), Some(EditReply::Cancel));
        editors.open("e3");
        editors.close("old");
        editors.close("e3");
        assert!(!editors.reply("e3", &serde_json::json!("late")));
    }
}
