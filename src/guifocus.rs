//! Expiring graphical focus state read independently of agent polling.

// Rust guideline compliant 2026-02-21
use serde::{Deserialize, Serialize};

/// Maximum focus lifetime after the graphical owner stops publishing.
pub const LIFETIME_MS: u64 = 5_000;

/// A short-lived focus snapshot with no session or document content.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Snapshot {
    pub focused: bool,
    pub expires_ms: u64,
}

impl Snapshot {
    /// Whether this snapshot can suppress an unfocused-only notification.
    #[must_use]
    pub fn is_focused(self, now_ms: u64) -> bool {
        self.focused
            && self.expires_ms > now_ms
            && self.expires_ms.saturating_sub(now_ms) <= LIFETIME_MS
    }
}

/// Reads bounded focus state, treating stale or unreadable state as unfocused.
#[must_use]
pub fn read_focused(path: &std::path::Path, now_ms: u64) -> bool {
    use std::io::Read as _;
    let Ok(file) = std::fs::File::open(path) else {
        return false;
    };
    let mut body = Vec::new();
    // The packet has two scalar fields; 256 bytes leaves room for JSON spacing.
    if file.take(257).read_to_end(&mut body).is_err() || body.len() > 256 {
        return false;
    }
    serde_json::from_slice::<Snapshot>(&body).is_ok_and(|snapshot| snapshot.is_focused(now_ms))
}

/// Wall-clock milliseconds shared across the daemon and its child process.
#[must_use]
pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| {
            u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gui_focus_expires_and_rejects_future_clock_jumps() {
        let snapshot = Snapshot {
            focused: true,
            expires_ms: 6_000,
        };
        assert!(snapshot.is_focused(1_000));
        assert!(snapshot.is_focused(5_999));
        assert!(!snapshot.is_focused(6_000));
        assert!(!snapshot.is_focused(999));
        assert!(
            !Snapshot {
                focused: false,
                ..snapshot
            }
            .is_focused(2_000)
        );
    }

    #[test]
    fn gui_focus_mailbox_is_bounded_and_fails_open_for_notifications() {
        let path = std::env::temp_dir().join(format!(
            "plank-focus-test-{}-{}",
            std::process::id(),
            now_ms()
        ));
        assert!(!read_focused(&path, 1_000));
        for body in [
            "{}".into(),
            "invalid".into(),
            r#"{"focused":true,"expires_ms":1000}"#.into(),
            format!(
                r#"{{"focused":true,"expires_ms":2000,"pad":"{}"}}"#,
                "x".repeat(256)
            ),
        ] {
            std::fs::write(&path, body).unwrap();
            assert!(!read_focused(&path, 1_000));
        }
        std::fs::write(&path, r#"{"focused":true,"expires_ms":2000}"#).unwrap();
        assert!(read_focused(&path, 1_000));
        std::fs::remove_file(path).unwrap();
    }
}
