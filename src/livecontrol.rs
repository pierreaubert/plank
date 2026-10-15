// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Controls that can change an active Plank turn without submitting another prompt.

// Rust guideline compliant 2026-02-21
use serde::{Deserialize, Serialize};

/// A narrowly scoped setting accepted independently of the prompt queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(from = "Wire", into = "Wire")]
pub enum LiveControl {
    /// Enables or disables automatic micro-compaction.
    MicroCompaction(bool),
    /// Enables or disables repetition loop guards.
    LoopGuard(bool),
    /// Records the hints preference for the next turn boundary.
    Hints(bool),
    /// Changes desktop delivery policy independently of the active prompt.
    Notifications(NotifyMode),
}

/// When desktop notifications fire.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NotifyMode {
    /// On every qualifying turn event.
    #[default]
    Always,
    /// Only when the session is not visibly focused in the delivering client.
    Unfocused,
    /// Never deliver desktop notifications.
    Never,
}

impl NotifyMode {
    /// Returns the settings-file spelling.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Always => "always",
            Self::Unfocused => "unfocused",
            Self::Never => "never",
        }
    }

    /// Parses modes and their legacy boolean spellings.
    #[must_use]
    pub fn parse(text: &str) -> Option<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "always" | "true" | "on" => Some(Self::Always),
            "unfocused" => Some(Self::Unfocused),
            "never" | "false" | "off" => Some(Self::Never),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(tag = "control", rename_all = "snake_case", deny_unknown_fields)]
enum Wire {
    MicroCompaction { enabled: bool },
    LoopGuard { enabled: bool },
    Hints { enabled: bool },
    Notifications { mode: NotifyMode },
}

impl From<Wire> for LiveControl {
    fn from(value: Wire) -> Self {
        match value {
            Wire::MicroCompaction { enabled } => Self::MicroCompaction(enabled),
            Wire::LoopGuard { enabled } => Self::LoopGuard(enabled),
            Wire::Hints { enabled } => Self::Hints(enabled),
            Wire::Notifications { mode } => Self::Notifications(mode),
        }
    }
}

impl From<LiveControl> for Wire {
    fn from(value: LiveControl) -> Self {
        match value {
            LiveControl::MicroCompaction(enabled) => Self::MicroCompaction { enabled },
            LiveControl::LoopGuard(enabled) => Self::LoopGuard { enabled },
            LiveControl::Hints(enabled) => Self::Hints { enabled },
            LiveControl::Notifications(mode) => Self::Notifications { mode },
        }
    }
}

impl LiveControl {
    /// Parses explicit values without interpreting a bare command as a toggle.
    #[must_use]
    pub fn parse(text: &str) -> Option<Self> {
        let mut words = text.split_whitespace();
        let name = words.next()?;
        let argument = words.next()?;
        if words.next().is_some() {
            return None;
        }
        if name == "/notify" {
            return NotifyMode::parse(argument).map(Self::Notifications);
        }
        let enabled = match argument {
            "on" => true,
            "off" => false,
            _ => return None,
        };
        match name {
            "/mc" => Some(Self::MicroCompaction(enabled)),
            "/loopguard" | "/lg" => Some(Self::LoopGuard(enabled)),
            "/hints" => Some(Self::Hints(enabled)),
            _ => None,
        }
    }

    /// Returns the additional capability required by newer independent controls.
    #[must_use]
    pub fn required_feature(self) -> Option<&'static str> {
        match self {
            Self::Hints(_) => Some("live_hints"),
            Self::Notifications(_) => Some("live_notify"),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::LiveControl;
    #[test]
    fn live_controls_require_exact_names_and_values() {
        assert_eq!(
            LiveControl::parse(" /mc off "),
            Some(LiveControl::MicroCompaction(false))
        );
        assert_eq!(
            LiveControl::parse("/lg on"),
            Some(LiveControl::LoopGuard(true))
        );
        assert_eq!(
            LiveControl::parse("/hints on"),
            Some(LiveControl::Hints(true))
        );
        assert_eq!(
            LiveControl::parse("/hints off"),
            Some(LiveControl::Hints(false))
        );
        for text in [
            "/mc",
            "/mc toggle",
            "/mc off /quit",
            "/mcfoo on",
            "/think off",
            "/hints",
            "/hints on /quit",
            "/hints yes",
        ] {
            assert!(LiveControl::parse(text).is_none());
        }
        let control = LiveControl::LoopGuard(false);
        assert_eq!(
            serde_json::to_value(control).unwrap(),
            serde_json::json!({"control":"loop_guard","enabled":false})
        );
        assert_eq!(
            serde_json::from_str::<LiveControl>(&serde_json::to_string(&control).unwrap()).unwrap(),
            control
        );
        for (text, mode) in [
            ("/notify always", super::NotifyMode::Always),
            ("/notify unfocused", super::NotifyMode::Unfocused),
            ("/notify OFF", super::NotifyMode::Never),
        ] {
            let control = LiveControl::Notifications(mode);
            assert_eq!(LiveControl::parse(text), Some(control));
            let wire = serde_json::to_value(control).unwrap();
            assert_eq!(
                wire,
                serde_json::json!({"control":"notifications","mode":mode.as_str()})
            );
            assert_eq!(
                serde_json::from_value::<LiveControl>(wire).unwrap(),
                control
            );
        }
        for text in ["/notify", "/notify never /quit", "/notify sometimes"] {
            assert!(LiveControl::parse(text).is_none());
        }
        for wire in [
            serde_json::json!({"control":"notifications","enabled":false}),
            serde_json::json!({"control":"notifications","mode":"unknown"}),
            serde_json::json!({"control":"loop_guard","enabled":false,"extra":true}),
        ] {
            assert!(serde_json::from_value::<LiveControl>(wire).is_err());
        }
    }
}
