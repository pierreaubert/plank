//! Bounded pages of completed delegated conversations shared with graphical clients.

use serde::{Deserialize, Serialize};

/// Retained completed conversations, matching the host's repro archive.
pub const MAX_RUNS: usize = 8;
/// Text pieces per page, bounding render work and wire size.
pub const PAGE_CHUNKS: usize = 16;
/// Maximum bytes per piece; pieces always end on Unicode boundaries.
pub const CHUNK_BYTES: usize = 4096;

/// Parses the optional stable run ID and positive page number.
///
/// Returns `None` for arguments outside `[run-id [--page N]]`.
#[must_use]
pub fn parse_request(argument: &str) -> Option<(Option<u64>, u64)> {
    let mut words = argument.split_whitespace();
    let Some(id) = words.next() else {
        return Some((None, 1));
    };
    let id = id.parse::<u64>().ok().filter(|id| *id > 0)?;
    let page = match words.next() {
        None => 1,
        Some("--page") => words.next()?.parse::<u64>().ok().filter(|page| *page > 0)?,
        _ => return None,
    };
    words.next().is_none().then_some((Some(id), page))
}

/// One completed run's stable identity and display metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Run {
    pub id: u64,
    pub label: String,
    pub task: String,
    pub outcome: String,
}

/// One lossless piece of a message, retaining its role and continuation state.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Chunk {
    pub role: String,
    pub text: String,
    pub continued: bool,
}

impl std::fmt::Debug for Chunk {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HistoryChunk")
            .field("role", &self.role)
            .field("bytes", &self.text.len())
            .finish_non_exhaustive()
    }
}

/// Requested history page; conversation bodies are absent from the run inventory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Page {
    pub runs: Vec<Run>,
    pub selected: Option<u64>,
    pub total: u64,
    pub page: u64,
    pub chunks: Vec<Chunk>,
}

impl Page {
    /// Validates identities and bounds before rendering remote history.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        let pages = self.total.div_ceil(PAGE_CHUNKS as u64).max(1);
        self.runs.len() <= MAX_RUNS
            && self.runs.iter().enumerate().all(|(index, run)| {
                run.id > 0
                    && !self.runs[..index].iter().any(|other| other.id == run.id)
                    && run.label.len() <= 128
                    && run.task.len() <= 1024
                    && run.outcome.len() <= 1024
                    && !run
                        .label
                        .chars()
                        .chain(run.task.chars())
                        .chain(run.outcome.chars())
                        .any(char::is_control)
            })
            && self.page > 0
            && self.page <= pages
            && self
                .selected
                .is_none_or(|id| self.runs.iter().any(|run| run.id == id))
            && (self.selected.is_some() || (self.total == 0 && self.chunks.is_empty()))
            && self.chunks.len() as u64
                == self
                    .total
                    .saturating_sub((self.page - 1) * PAGE_CHUNKS as u64)
                    .min(PAGE_CHUNKS as u64)
            && self.chunks.iter().all(|chunk| {
                chunk.text.len() <= CHUNK_BYTES
                    && matches!(
                        chunk.role.as_str(),
                        "user" | "assistant" | "tool" | "system"
                    )
            })
    }
}

/// Builds one page without copying the entire conversation into intermediate storage.
#[must_use]
pub fn page<'a>(
    runs: Vec<Run>,
    selected: Option<u64>,
    messages: impl Iterator<Item = (&'a str, &'a str)> + Clone,
    requested: u64,
) -> Page {
    let pieces = |(_, text): (&str, &str)| {
        let mut count = 1u64;
        let mut rest = text;
        while rest.len() > CHUNK_BYTES {
            let mut end = CHUNK_BYTES;
            while !rest.is_char_boundary(end) {
                end -= 1;
            }
            rest = &rest[end..];
            count += 1;
        }
        count
    };
    let total: u64 = messages.clone().map(pieces).sum();
    let page = requested.clamp(1, total.div_ceil(PAGE_CHUNKS as u64).max(1));
    let first = (page - 1) * PAGE_CHUNKS as u64;
    let mut ordinal = 0;
    let mut chunks = Vec::new();
    for (role, text) in messages {
        let mut rest = text;
        let mut continued = false;
        loop {
            let mut end = rest.len().min(CHUNK_BYTES);
            while !rest.is_char_boundary(end) {
                end -= 1;
            }
            if ordinal >= first && chunks.len() < PAGE_CHUNKS {
                chunks.push(Chunk {
                    role: role.into(),
                    text: rest[..end].into(),
                    continued,
                });
            }
            ordinal += 1;
            if chunks.len() == PAGE_CHUNKS {
                break;
            }
            rest = &rest[end..];
            if rest.is_empty() {
                break;
            }
            continued = true;
        }
        if chunks.len() == PAGE_CHUNKS {
            break;
        }
    }
    Page {
        runs,
        selected,
        total,
        page,
        chunks,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn subagent_history_rejects_forged_selection_and_invalid_page_shapes() {
        let runs = vec![Run {
            id: 3,
            label: "child".into(),
            task: "inspect".into(),
            outcome: "report".into(),
        }];
        let valid = page(
            runs,
            Some(3),
            [("assistant", "private history")].into_iter(),
            1,
        );
        assert!(valid.is_valid());
        assert!(!format!("{valid:?}").contains("private history"));
        let mut forged = valid.clone();
        forged.selected = Some(4);
        assert!(!forged.is_valid());
        forged = valid.clone();
        forged.runs.push(forged.runs[0].clone());
        assert!(!forged.is_valid());
        forged = valid.clone();
        forged.total = u64::MAX;
        assert!(!forged.is_valid());
        forged = valid.clone();
        forged.page = 0;
        assert!(!forged.is_valid());
        forged = valid.clone();
        forged.chunks[0].text = "x".repeat(CHUNK_BYTES + 1);
        assert!(!forged.is_valid());
        forged = valid;
        forged.chunks[0].role = "executable".into();
        assert!(!forged.is_valid());
    }

    #[test]
    fn subagent_history_pages_reconstruct_long_unicode_messages_without_loss() {
        let runs = vec![Run {
            id: 7,
            label: "review".into(),
            task: "review parser".into(),
            outcome: "report".into(),
        }];
        let long = "🦀界 e\u{301}\n".repeat(10000);
        let messages = [
            ("user", "task"),
            ("assistant", long.as_str()),
            ("tool", ""),
            ("assistant", "report"),
        ];
        let first = page(runs.clone(), Some(7), messages.into_iter(), 1);
        assert!(first.is_valid());
        assert_eq!(first.chunks.len(), PAGE_CHUNKS);
        let mut rebuilt = String::new();
        let mut empty_tool = false;
        for number in 1..=first.total.div_ceil(PAGE_CHUNKS as u64) {
            let current = page(runs.clone(), Some(7), messages.into_iter(), number);
            assert!(current.is_valid());
            assert!(serde_json::to_vec(&current).unwrap().len() < 512 * 1024);
            for chunk in current.chunks {
                empty_tool |= chunk.role == "tool" && chunk.text.is_empty();
                if chunk.role == "assistant" {
                    rebuilt.push_str(&chunk.text);
                }
            }
        }
        assert_eq!(rebuilt, format!("{long}report"));
        assert!(empty_tool);
        let last = page(runs, Some(7), messages.into_iter(), u64::MAX);
        assert!(last.is_valid());
        assert_eq!(last.page, first.total.div_ceil(PAGE_CHUNKS as u64));
    }
}
