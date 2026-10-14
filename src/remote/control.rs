// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Remote-control interface (issue #25): a loopback WebSocket server that lets
//! another process or machine mirror a running plank instance and, by policy,
//! drive it. This is the CLI-only, backend-free variant specified in
//! `docs/REMOTE-CONTROL-DESIGN.md`.
//!
//! A remote client is *another front-end* over the existing
//! [`crate::worker::UiEvent`] stream: the worker broadcasts events to a
//! [`crate::worker::BroadcastBus`], each connection subscribes and pumps them
//! out as JSON frames, and inbound frames push prompts / interrupts / `/btw`
//! questions into the shared [`crate::worker::TurnShared`] — exactly what the
//! local UI does. There is one engine, one session, one transcript (design §7).
//!
//! Transport is blocking [`tungstenite`] on dedicated threads (no async
//! runtime), matching plank's synchronous worker idiom (design §4.7). The
//! server binds `127.0.0.1` only; off-box reach is the user's SSH tunnel.
//!
//! ## What this module implements
//! - The versioned JSON wire protocol ([`ServerFrame`] / [`ClientFrame`]), a
//!   near-1:1 image of `UiEvent` + `TurnShared` (design §4.3), with lossless
//!   round-trip.
//! - Token auth: generation, constant-time comparison, mandatory first-frame
//!   handshake (design §4.6).
//! - The single-controller [`ControlPolicy`] (one controller, many mirrors;
//!   design §4.4).
//! - The accept/connection threads: auth, `hello`, `snapshot` replay, live
//!   mirroring, `status` coalescing, and inbound control into `TurnShared`.
//!
//! ## Live wiring (issue #25, done)
//! - The worker's stream events mirror onto the bus and remote
//!   `prompt`/`command`/`btw`/`interrupt` frames drive the real agent through the
//!   TUI turn loop: `busy_ui_loop` mirrors inline and drives turns from the idle
//!   loop. There is no headless equivalent — `/rc` is the only way to start the
//!   server and it can only be typed in the TUI.
//! - `command` (slash) frames route through the shared slash dispatcher: they
//!   land in the shared queue and the turn loop sends `/`-prefixed lines through
//!   the same path the local REPL/TUI uses.
//! - Reconnect grace window ([`CONTROL_GRACE`]): a dropped controller's slot is
//!   held so a brief drop can reclaim it via `resume_from`.
//!
//! ## Follow-up (issue #25, done)
//! - `/rc` is TUI-only: the TUI's idle loop drains the remote queue straight
//!   from the shared [`crate::worker::TurnShared`] and starts a turn as if the
//!   queued line had been typed locally, mirroring output back onto the bus
//!   the same way a local turn does.
//! - The CLI client `plank remote <url>` connects to this server, authenticates,
//!   mirrors output, and sends `prompt`/`command`/`btw`/`interrupt` frames (see
//!   [`crate::remote::client`]).
//!
//! ## Hardening (issue #25, done)
//! - `Origin` allow-list on the WebSocket upgrade: missing / loopback Origins
//!   are always allowed (native clients send none), other browser Origins must
//!   appear in [`ServerConfig::allowed_origins`] or the upgrade is refused with
//!   an HTTP 403 (see [`origin_allowed`]). `/rc` always starts loopback-only
//!   with an empty allow-list, which is sufficient since a missing or loopback
//!   Origin is always accepted; a non-loopback setup would need this wired to
//!   a config source again.
//! - Bounded per-client outbound queue: the socket write buffer is capped
//!   ([`ServerConfig::queue_max`]); a slow/stalled client whose unsent output
//!   exceeds the cap is evicted (slow-consumer backpressure) instead of
//!   buffered without bound. The bus prunes the dropped subscriber on the next
//!   broadcast.
//! - A minimal self-contained static web client served at `GET /` (see
//!   [`WEB_CLIENT_HTML`]); it speaks the exact JSON frames below.

use std::io::{ErrorKind, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tungstenite::Message;

use crate::status::{Status, WorkerState};
use crate::worker::{BroadcastBus, SeqEvent, TurnShared, UiEvent};

/// Wire protocol version carried in every frame's `v` field. Adding frame
/// types is backward-compatible; changing existing ones bumps this (design §7).
pub const PROTOCOL_VERSION: u32 = 1;

/// How often the connection loop wakes to pump the bus and poll the socket.
const POLL_INTERVAL: Duration = Duration::from_millis(50);

/// Minimum spacing between `status` frames per connection (coalescing, §4.9):
/// at most ~10/s. Text frames are never coalesced.
const STATUS_MIN_INTERVAL: Duration = Duration::from_millis(100);

/// The self-contained static web client served at `GET /`. No external deps:
/// all CSS/JS is inline. It authenticates with a token and speaks the same
/// [`PROTOCOL_VERSION`] JSON frames as the native client.
pub const WEB_CLIENT_HTML: &str = include_str!("web_client.html");

/// The plank logo shown in the web client's header, served at `GET /logo.png`.
/// A 128px downscale of `assets/logo.png` (the full-size original is ~1 MB,
/// which has no business inside the binary or on every page load).
pub const WEB_CLIENT_LOGO: &[u8] = include_bytes!("../../assets/logo-web.png");

// --- Protocol ---------------------------------------------------------------

/// Flattened, serializable view of [`Status`] for `status` frames (§4.3). The
/// worker state is a lowercase string so the schema is language-neutral.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StatusWire {
    /// Worker state name (`idle`, `prefill`, `generating`, ...).
    pub state: String,
    /// Prefill tokens done / total.
    pub prefill_done: i32,
    /// Prefill tokens total.
    pub prefill_total: i32,
    /// Tokens generated so far.
    pub generated: i32,
    /// Generation throughput, tokens per second.
    pub gen_tps: f64,
    /// Prefill throughput, tokens per second.
    pub prefill_tps: f64,
    /// Seconds elapsed in the current operation.
    pub elapsed_secs: f64,
    /// Context tokens in use.
    pub ctx_used: i32,
    /// Context window size.
    pub ctx_size: i32,
    /// Error text (empty unless `state == "error"`).
    pub error: String,
    /// Working directory label, home collapsed to `~` (the footer's dir prefix).
    pub cwd: String,
    /// Current git branch, empty outside a repo.
    pub branch: String,
    /// Engine origin: a provider or remote host, else `(local)`.
    pub origin: String,
    /// Reasoning level short name (`off`, `low`, `med`, `high`, `max`).
    pub think: String,
}

fn state_name(s: WorkerState) -> &'static str {
    match s {
        WorkerState::Idle => "idle",
        WorkerState::Prefill => "prefill",
        WorkerState::Generating => "generating",
        WorkerState::Compacting => "compacting",
        WorkerState::Saving => "saving",
        WorkerState::Error => "error",
        WorkerState::Stopped => "stopped",
    }
}

/// Remote-host policy without the bearer token or transcript contents.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HostWire {
    pub endpoint: Option<String>,
    pub holder: String,
    pub controller: Option<u64>,
    /// Whether a client may take the local controller slot without a grant.
    pub automatic: bool,
    /// A disconnected controller still owns its reconnect grace reservation.
    pub reserved: bool,
    pub pending: Vec<u64>,
    pub pending_total: usize,
}

impl HostWire {
    /// Reports that no host is running.
    #[must_use]
    pub fn off() -> Self {
        Self {
            endpoint: None,
            holder: "off".into(),
            controller: None,
            automatic: false,
            reserved: false,
            pending: Vec::new(),
            pending_total: 0,
        }
    }
}

/// Requested activity, context, usage, and throughput measurements from the host.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MetricsWire {
    /// Delegated runs captured from the shared terminal roster.
    Subagents {
        rows: Vec<SubagentWire>,
        total: usize,
        running: usize,
    },
    /// Bounded statistics and prose from the completed insights report.
    Insights {
        markdown: String,
        /// Checksum of the completed standalone HTML, linking its download to this view.
        #[serde(default)]
        html_sha256: Option<String>,
    },
    /// Requested engine catalog and host availability, independent of model load.
    Engines {
        snapshot: crate::enginefit::EngineInventory,
    },
    /// Year-wide activity and scoped figures from saved-session metadata.
    Activity {
        snapshot: crate::stats::ActivitySnapshot,
    },
    /// Raw tokenizer estimates; resident usage includes conversation framing.
    Context {
        model: String,
        window: u64,
        used: u64,
        /// System, MCP, messages, AGENTS.md, and memory, in that order.
        categories: [u64; 5],
    },
    /// Engine-agnostic session totals; input includes cached provider tokens.
    Usage {
        model: String,
        input: u64,
        output: u64,
        cache_read: u64,
        cache_write: u64,
        provider_passes: Option<u64>,
        wall_secs: f64,
    },
    /// Process-wide rings: decode windows and completed prefill passes.
    Throughput {
        generation: Vec<f64>,
        prefill: Vec<f64>,
    },
}

impl MetricsWire {
    /// Returns the independently coalesced snapshot category.
    #[must_use]
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Subagents { .. } => "subagents",
            Self::Insights { .. } => "insights",
            Self::Engines { .. } => "engines",
            Self::Activity { .. } => "activity",
            Self::Context { .. } => "context",
            Self::Usage { .. } => "usage",
            Self::Throughput { .. } => "throughput",
        }
    }
}

/// Bounded delegated-run metadata, independent of its output buffer.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SubagentWire {
    #[serde(default)]
    pub output: Option<String>,
    #[serde(default)]
    pub output_truncated: bool,
    #[serde(default)]
    pub live: Option<SubagentLiveWire>,
    #[serde(default)]
    pub id: u64,
    #[serde(default)]
    pub report: Option<String>,
    pub label: String,
    pub task: String,
    pub running: bool,
    pub elapsed_ms: u64,
    pub prefill: u64,
    pub generated: u64,
}

impl SubagentWire {
    /// Captures active rows first and at most 64 bounded display summaries.
    #[must_use]
    pub fn capture(pane: &crate::tui::SubPane, now: u64) -> MetricsWire {
        let clip = |text: &str, bytes: usize| {
            let mut text: String = text
                .chars()
                .map(|c| if c.is_control() { ' ' } else { c })
                .collect();
            let mut end = text.len().min(bytes);
            while !text.is_char_boundary(end) {
                end -= 1;
            }
            text.truncate(end);
            text
        };
        let mut runs: Vec<_> = pane.runs.iter().collect();
        runs.sort_by_key(|run| !run.running);
        MetricsWire::Subagents {
            total: runs.len(),
            running: runs.iter().filter(|run| run.running).count(),
            rows: runs
                .into_iter()
                .take(64)
                .map(|run| {
                    let (output, clipped) = bounded_subagent_output(&run.output);
                    Self {
                        output_truncated: !output.is_empty() && (run.output_truncated || clipped),
                        output: (!output.is_empty()).then_some(output),
                        live: run.live.map(|live| SubagentLiveWire {
                            prefilling: live.prefilling,
                            prefill_done: u64::try_from(
                                live.prefill_done.max(0).min(live.prefill_total.max(0)),
                            )
                            .unwrap_or(0),
                            prefill_total: u64::try_from(live.prefill_total.max(0)).unwrap_or(0),
                            generated: u64::try_from(live.generated.max(0)).unwrap_or(0),
                        }),
                        id: run.id,
                        report: run.report.as_deref().map(bounded_subagent_report),
                        label: clip(&run.label, 128),
                        task: clip(&run.task, 1024),
                        running: run.running,
                        elapsed_ms: run.elapsed_ms(now),
                        prefill: run.prefill,
                        generated: run.generated,
                    }
                })
                .collect(),
        }
    }
}

/// Counters for an unfinished pass; completed totals exclude these estimates.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SubagentLiveWire {
    pub prefilling: bool,
    pub prefill_done: u64,
    pub prefill_total: u64,
    /// Streamed text pieces; online providers may group multiple tokens.
    pub generated: u64,
}

/// Clips report text within the graphical client's escaped frame budget.
pub(crate) fn bounded_subagent_report(report: &str) -> String {
    // Sixty-four worst-case escaped rows must fit the client's 2 MiB frame.
    const REPORT_BYTES: usize = 4096;
    const NOTICE: &str =
        "\n[Report truncated; the complete result remains in the parent conversation.]";
    let mut text: String = report
        .chars()
        .filter(|c| !c.is_control() || matches!(c, '\n' | '\r' | '\t'))
        .take(REPORT_BYTES + 1)
        .collect();
    if text.len() > REPORT_BYTES {
        let mut end = REPORT_BYTES - NOTICE.len();
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        text.truncate(end);
        text.push_str(NOTICE);
    }
    text
}

/// Retains the newest 4 KiB of sanitized output without splitting UTF-8.
pub(crate) const SUBAGENT_OUTPUT_BYTES: usize = 4096;

/// Retains the newest sanitized output within the delegated-stream byte budget.
pub(crate) fn bounded_subagent_output(text: &str) -> (String, bool) {
    // Together with each 4 KiB report, 64 rows fit the client's 2 MiB frame.
    let mut tail: String = text
        .chars()
        .rev()
        .filter(|c| !c.is_control() || matches!(c, '\n' | '\r' | '\t'))
        .take(SUBAGENT_OUTPUT_BYTES + 1)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    let clipped = tail.len() > SUBAGENT_OUTPUT_BYTES;
    if clipped {
        let mut start = tail.len() - SUBAGENT_OUTPUT_BYTES;
        while !tail.is_char_boundary(start) {
            start += 1;
        }
        tail.drain(..start);
    }
    (tail, clipped)
}

/// A bounded task row shared by native and browser clients.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TaskRowWire {
    pub id: u32,
    pub subject: String,
    pub status: String,
    pub active_form: Option<String>,
}

impl TaskRowWire {
    /// Captures at most 64 tasks, prioritizing active and pending work.
    #[must_use]
    pub fn capture(list: &crate::tasks::TaskList) -> Vec<Self> {
        let clip = |text: &str| {
            let mut text: String = text
                .chars()
                .map(|c| if c.is_control() { ' ' } else { c })
                .collect();
            let mut end = text.len().min(512);
            while !text.is_char_boundary(end) {
                end -= 1;
            }
            text.truncate(end);
            text
        };
        let mut tasks = list.tasks().iter().collect::<Vec<_>>();
        tasks.sort_by_key(|task| {
            (
                match task.status {
                    crate::tasks::TaskStatus::InProgress => 0,
                    crate::tasks::TaskStatus::Pending => 1,
                    crate::tasks::TaskStatus::Completed => 2,
                },
                task.id,
            )
        });
        tasks
            .into_iter()
            .take(64)
            .map(|task| Self {
                id: task.id,
                subject: clip(&task.subject),
                status: task.status.as_str().into(),
                active_form: task.active_form.as_deref().map(clip),
            })
            .collect()
    }
}

/// Detached model-update progress, independent of the active model turn.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DownloadsWire {
    /// Current detached helper, absent when no trustworthy state is available.
    pub state: Option<DownloadStateWire>,
    /// A recorded manifest job can be started or resumed.
    pub pending: bool,
    /// Requested cancellation, while the helper still reports active work.
    pub cancel_requested: Option<String>,
}

/// Bounded progress from the detached engine-host downloader.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DownloadStateWire {
    /// Model family and manifest version.
    pub set: String,
    pub version: u32,
    /// Current artifact and one-based position within the job.
    pub current: String,
    pub index: usize,
    pub of: usize,
    /// Current artifact progress, not the combined size of the whole job.
    pub done_bytes: u64,
    pub total_bytes: u64,
    pub rate_bps: u64,
    /// Downloading, rehashing, verifying, staged, failed, cancelled, or unknown.
    pub phase: String,
    /// Whether a cancelled helper retained its partial files.
    pub partials_kept: Option<bool>,
    pub error: Option<String>,
}

impl DownloadsWire {
    /// Reads the helper's published state and pending job without starting work.
    #[must_use]
    pub fn capture_in(root: &std::path::Path) -> Self {
        let clip = |text: &str, limit: usize| {
            let text: String = text
                .chars()
                .filter(|character| !character.is_control())
                .collect();
            let mut end = text.len().min(limit);
            while !text.is_char_boundary(end) {
                end -= 1;
            }
            text[..end].to_owned()
        };
        let state = crate::downloader::read_state_in(root)
            .filter(|state| {
                // Terminal results remain useful after the detached helper exits.
                matches!(
                    state.phase,
                    crate::downloader::Phase::Staged
                        | crate::downloader::Phase::Failed
                        | crate::downloader::Phase::Cancelled
                ) || !crate::downloader::is_stale(
                    state,
                    crate::downloader::now_epoch(),
                    &crate::downloader::pid_alive,
                )
            })
            .map(|state| DownloadStateWire {
                set: clip(&state.set, 256),
                version: state.version,
                current: clip(&state.current, 256),
                index: state.index.min(state.of),
                of: state.of,
                done_bytes: state.done_bytes.min(state.total_bytes),
                total_bytes: state.total_bytes,
                rate_bps: state.rate_bps,
                phase: match state.phase {
                    crate::downloader::Phase::Rehashing => "rehashing",
                    crate::downloader::Phase::Downloading => "downloading",
                    crate::downloader::Phase::Verifying => "verifying",
                    crate::downloader::Phase::Staged => "staged",
                    crate::downloader::Phase::Failed => "failed",
                    crate::downloader::Phase::Cancelled => "cancelled",
                    crate::downloader::Phase::Unknown => "unknown",
                }
                .into(),
                partials_kept: (state.phase == crate::downloader::Phase::Cancelled)
                    .then(|| crate::downloader::cancelled_kept(&state)),
                error: state.error.map(|error| clip(&error, 4096)),
            });
        let active = state.as_ref().is_some_and(|state| {
            matches!(
                state.phase.as_str(),
                "rehashing" | "downloading" | "verifying" | "unknown"
            )
        });
        let cancel_requested = active
            .then(|| crate::downloader::read_cancel_in(root))
            .flatten()
            .map(|cancel| match cancel {
                crate::downloader::Cancel::Keep => "keep".into(),
                crate::downloader::Cancel::Delete => "delete".into(),
            });
        Self {
            state,
            pending: crate::downloader::pending_job_in(root).is_some(),
            cancel_requested,
        }
    }

    /// Reads current progress from the running engine host's Plank directory.
    #[must_use]
    pub fn capture() -> Self {
        Self::capture_in(&crate::manifest::plank_dir())
    }
}

/// One requested page of the engine-host KV cache lineage.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CacheWire {
    /// Number of lineage rows, including synthetic group headings.
    pub total: usize,
    /// One-based page with 32 rows per page.
    pub page: usize,
    /// Engine-authored storage and reclamation summary.
    pub summary: String,
    /// Bounded rows in tree order.
    pub rows: Vec<CacheRowWire>,
}

/// One cache blob or synthetic lineage heading.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CacheRowWire {
    /// Full fingerprint paired with the file handle, absent for ambiguous rows.
    pub fingerprint: Option<String>,
    /// Stable opaque file handle, paired with the full fingerprint.
    #[serde(default)]
    pub entry: Option<String>,
    /// Indentation within the cache lineage.
    pub depth: usize,
    /// Role and name as reported by the engine.
    pub label: String,
    /// Role-specific engine/model or session details.
    pub detail: String,
    /// Size, hits, age and expiration markers.
    pub status: String,
    /// Current pin state.
    pub pinned: bool,
}

impl CacheWire {
    /// Captures bounded lineage rows while refusing ambiguous command identities.
    #[must_use]
    pub fn capture(view: &crate::kvpane::KvPane, requested: usize) -> Self {
        Self::capture_with_entries(view, requested, &[])
    }

    /// Captures rows from the same scan used to construct their lineage view.
    #[must_use]
    pub fn capture_with_entries(
        view: &crate::kvpane::KvPane,
        requested: usize,
        entries: &[(std::path::PathBuf, crate::kvmeta::KvMeta)],
    ) -> Self {
        // Match other compact activity pages; no cache bodies cross the wire.
        const PAGE_SIZE: usize = 32;
        let clip = |text: &str, limit: usize| {
            let text: String = text
                .chars()
                .filter(|character| !character.is_control())
                .collect();
            let mut end = text.len().min(limit);
            while !text.is_char_boundary(end) {
                end -= 1;
            }
            text[..end].to_owned()
        };
        let rows = view.rows();
        let handles: Vec<_> = entries
            .iter()
            .map(|(path, _)| cache_entry_handle(path))
            .collect();
        let mut handle_counts = std::collections::HashMap::new();
        for handle in &handles {
            *handle_counts.entry(handle).or_insert(0usize) += 1;
        }
        let mut counts = std::collections::HashMap::new();
        for fingerprint in rows.iter().filter_map(|row| row.fingerprint.as_ref()) {
            *counts.entry(fingerprint).or_insert(0usize) += 1;
        }
        let total = rows.len();
        let page = requested.clamp(1, total.div_ceil(PAGE_SIZE).max(1));
        let rows = rows
            .iter()
            .skip((page - 1) * PAGE_SIZE)
            .take(PAGE_SIZE)
            .map(|row| {
                let entry = row
                    .idx
                    .and_then(|index| entries.get(index).zip(handles.get(index)))
                    .filter(|((_, meta), handle)| {
                        row.fingerprint.as_ref() == Some(&meta.fingerprint)
                            && handle_counts.get(handle) == Some(&1)
                    })
                    .map(|(_, handle)| handle.clone());
                let fingerprint = row
                    .fingerprint
                    .as_ref()
                    .filter(|fingerprint| {
                        (entry.is_some() || counts.get(fingerprint) == Some(&1))
                            && !fingerprint.is_empty()
                            && fingerprint.len() <= 4096
                            && !fingerprint.chars().any(|character| {
                                character.is_control() || character.is_whitespace()
                            })
                    })
                    .cloned();
                CacheRowWire {
                    entry: fingerprint.as_ref().and(entry),
                    fingerprint,
                    depth: row.depth,
                    label: clip(&row.label, 512),
                    detail: clip(&row.detail, 1024),
                    status: clip(&row.right, 512),
                    pinned: row.pinned,
                }
            })
            .collect();
        Self {
            total,
            page,
            summary: clip(&view.footer(), 512),
            rows,
        }
    }
}

/// Returns a stable opaque identifier without exposing an engine-host file path.
#[must_use]
pub fn cache_entry_handle(path: &std::path::Path) -> String {
    crate::session::sha1_hex(path.as_os_str().as_encoded_bytes())
}

/// One requested inventory of the running engine's extensions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExtensionsWire {
    /// Inventory category: plugins, skills, templates, hooks, mcp, or agents.
    pub kind: String,
    /// Engine-authored plain-text report, bounded independently of the transcript.
    pub report: String,
    /// Session switch for skills and hooks; absent for the other categories.
    pub enabled: Option<bool>,
}

impl ExtensionsWire {
    /// Captures an inventory without allowing terminal escapes or unbounded frames.
    #[must_use]
    pub fn capture(kind: &str, report: &str, enabled: Option<bool>) -> Self {
        // Six reports fit comfortably within the client's frame and replay limits.
        const LIMIT: usize = 64 * 1024;
        let mut report: String = report
            .chars()
            .filter(|character| !character.is_control() || matches!(character, '\n' | '\t'))
            .collect();
        if report.len() > LIMIT {
            let mut end = LIMIT;
            while !report.is_char_boundary(end) {
                end -= 1;
            }
            report.truncate(end);
            report.push_str("\nInventory truncated; request the full slash-command report.");
        }
        Self {
            kind: kind.into(),
            report,
            enabled,
        }
    }
}

/// One page of conversation checkpoints or fork points.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HistoryWire {
    /// `checkpoints` or `forks`.
    pub kind: String,
    /// Number of available points.
    pub total: usize,
    /// One-based page, with 32 rows per page.
    pub page: usize,
    /// Named checkpoints or numbered fork points.
    pub rows: Vec<HistoryRowWire>,
    /// Bounded tree report for fork pages; empty for checkpoint pages.
    pub report: String,
}

/// One rollback or fork point without its transcript or engine payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HistoryRowWire {
    /// Complete command argument; absent when too large for inline controls.
    pub key: Option<String>,
    /// Display name or numbered fork point.
    pub label: String,
    /// Conversation summary.
    pub detail: String,
}

/// One page of editable engine settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConfigWire {
    /// Number of built-in and contributed fields.
    pub total: usize,
    /// One-based page, with 32 rows per page.
    pub page: usize,
    /// Editable fields with their authoritative values.
    pub fields: Vec<ConfigFieldWire>,
}

/// One editable setting reported by the engine.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConfigFieldWire {
    /// Path accepted by `/config`.
    pub path: String,
    /// Human-facing label.
    pub label: String,
    /// Current value; empty means unset.
    pub value: String,
    /// Accepted shape or enumerated values.
    pub hint: String,
    /// False when the value exceeds the inline editor limit.
    pub editable: bool,
}

/// A bounded page of saved conversations on the engine host.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SessionsWire {
    /// Total number of saved conversations.
    pub total: usize,
    /// One-based page, with 64 rows per page.
    pub page: usize,
    /// Metadata without transcripts or payloads.
    pub sessions: Vec<SavedSessionWire>,
}

/// Metadata identifying a saved conversation on the engine host.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SavedSessionWire {
    /// Complete session identifier.
    pub id: String,
    /// Human-readable title.
    pub title: String,
    /// User-assigned tag.
    pub tag: String,
    /// Engine-host project directory.
    pub cwd: String,
    /// Last save or creation time.
    pub updated: u64,
}

impl SessionsWire {
    /// Captures a metadata page, clipping display text at UTF-8 boundaries.
    #[must_use]
    pub fn capture(entries: &[crate::session::SessionEntry], page: usize) -> Self {
        let clip = |text: &str| {
            // Bounds worst-case JSON escaping across a 64-row frame.
            let mut end = text.len().min(1024);
            while !text.is_char_boundary(end) {
                end -= 1;
            }
            text[..end].to_owned()
        };
        let page = page.clamp(1, entries.len().div_ceil(64).max(1));
        Self {
            total: entries.len(),
            page,
            sessions: entries
                .iter()
                .skip((page - 1) * 64)
                .take(64)
                .map(|entry| SavedSessionWire {
                    id: entry.id.clone(),
                    title: clip(&entry.title),
                    tag: clip(&entry.tag),
                    cwd: clip(&entry.cwd),
                    updated: entry.last_used.max(entry.created_at),
                })
                .collect(),
        }
    }
}

/// Bounded portable snapshot of the engine-owned background job table.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JobsWire {
    /// Number of running jobs, including omitted rows.
    pub running: usize,
    /// Number of tracked jobs, including omitted rows.
    pub total: usize,
    /// Running rows first, then finished rows, capped at 64.
    pub jobs: Vec<JobWire>,
}

/// One engine-owned job; output paths belong to the engine host.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JobWire {
    /// Engine job identifier.
    pub id: i64,
    /// Shell process identifier.
    pub pid: u32,
    /// Elapsed time at capture.
    pub elapsed_secs: f64,
    /// Running, done, or `timed_out`.
    pub state: String,
    /// Exit code for finished jobs.
    pub exit_code: Option<i64>,
    /// Engine-host output file path.
    pub path: String,
}

impl JobsWire {
    /// Captures job rows without transferring process ownership.
    #[must_use]
    pub fn capture(rows: &[crate::tools::bash::JobRow]) -> Self {
        use crate::tools::bash::JobState;
        let running = |row: &&crate::tools::bash::JobRow| row.state == JobState::Running;
        Self {
            running: rows.iter().filter(running).count(),
            total: rows.len(),
            jobs: rows
                .iter()
                .filter(running)
                .chain(rows.iter().filter(|row| !running(row)))
                // Keeps snapshots below the GUI's 2 MiB frame ceiling.
                .take(64)
                .map(|row| {
                    let (state, exit_code) = match row.state {
                        JobState::Running => ("running", None),
                        JobState::Done(code) => ("done", Some(code)),
                        JobState::TimedOut(code) => ("timed_out", Some(code)),
                    };
                    let mut path = row.path.to_string_lossy().into_owned();
                    // Bound escaped paths together with the 64-row frame cap.
                    if path.len() > 4096 {
                        let mut end = 4096 - 3;
                        while !path.is_char_boundary(end) {
                            end -= 1;
                        }
                        path.truncate(end);
                        path.push_str("...");
                    }
                    JobWire {
                        id: row.id,
                        pid: row.pid,
                        elapsed_secs: row.started.elapsed().as_secs_f64(),
                        state: state.into(),
                        exit_code,
                        path,
                    }
                })
                .collect(),
        }
    }
}

impl From<&Status> for StatusWire {
    fn from(s: &Status) -> Self {
        Self {
            state: state_name(s.state).to_owned(),
            prefill_done: s.prefill_done,
            prefill_total: s.prefill_total,
            generated: s.generated,
            gen_tps: s.gen_tps,
            prefill_tps: s.prefill_tps,
            elapsed_secs: s.elapsed_secs,
            ctx_used: s.ctx_used,
            ctx_size: s.ctx_size,
            error: s.error.clone(),
            // The footer's "where am I?" segments. They are process-wide rather
            // than per-snapshot, but they ride along here so a remote front-end
            // can show the same header the local footer does without a second
            // frame kind — and so a client attaching mid-session learns them
            // from the first status frame instead of only on a change.
            cwd: crate::status::cwd_label(),
            branch: crate::status::git_branch_label().unwrap_or_default(),
            origin: crate::status::engine_origin_label(),
            think: s.think.short_name().into_owned(),
        }
    }
}

/// One scrollback entry in a `snapshot` frame: a prior server message with its
/// bus sequence id.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScrollbackEntry {
    /// Bus sequence id of the original event.
    pub id: u64,
    /// The replayed message.
    #[serde(flatten)]
    pub msg: ServerMsg,
}

/// Server → client message body, discriminated by `type` (design §4.3).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerMsg {
    /// Sent once on connect.
    Hello {
        /// Protocol version the server speaks.
        protocol_version: u32,
        /// plank crate version.
        plank_version: String,
        /// Numeric session id assigned to this connection.
        session_id: u64,
        /// Whether this session currently holds control.
        controller: bool,
        /// Slash commands produce an explicit completion frame.
        #[serde(default)]
        command_completion: bool,
        /// Ordinary prompts produce an ID-matched completion frame.
        #[serde(default)]
        prompt_completion: bool,
        /// Slash controls produce ID-matched completion frames.
        #[serde(default)]
        command_request_completion: bool,
        /// Inputs retain their accepting controller lease until idle dispatch.
        #[serde(default)]
        ownership_fencing: bool,
    },
    /// Scrollback replay on connect / resume.
    Snapshot {
        /// Prior output frames (oldest first).
        scrollback: Vec<ScrollbackEntry>,
        /// Highest sequence id represented, for the client to `resume_from`.
        highest_id: Option<u64>,
    },
    /// Rendered assistant text.
    Visible {
        /// Text payload.
        text: String,
    },
    /// Thinking text.
    Think {
        /// Text payload.
        text: String,
    },
    /// Tool banner text.
    Tool {
        /// Text payload.
        text: String,
    },
    /// Stream error text.
    Error {
        /// Text payload.
        text: String,
    },
    /// A dim log line.
    Dim {
        /// Text payload.
        text: String,
    },
    /// A plain log line.
    Plain {
        /// Text payload.
        text: String,
    },
    /// A user-echo line.
    UserEcho {
        /// Text payload.
        text: String,
    },
    /// Terminate the in-progress rendered line.
    EndLine,
    /// A `/btw` side answer is starting.
    BtwBegin,
    /// The `/btw` answer finished.
    BtwEnd,
    /// Main-pass checkpoint marker.
    MainCheckpoint,
    /// Main-pass rollback marker.
    MainRollback,
    /// Status footer snapshot (coalesced, §4.9).
    Status {
        /// Flattened status fields.
        status: StatusWire,
    },
    /// Task-list snapshot, including the authoritative task and goal report.
    Tasks {
        /// Completed task count.
        completed: usize,
        /// Total task count.
        total: usize,
        /// Full task/goal report; absent in older servers.
        #[serde(default)]
        report: String,
        /// Bounded editable rows; omitted by older servers.
        #[serde(default)]
        rows: Vec<TaskRowWire>,
        #[serde(default)]
        goal: Option<String>,
    },
    /// Background jobs owned by the engine, not the UI process.
    Jobs {
        /// Bounded snapshot with running/total counters.
        snapshot: JobsWire,
    },
    /// Requested page of engine-host saved conversations.
    Sessions {
        /// Bounded metadata page.
        snapshot: SessionsWire,
    },
    /// Requested page of editable engine settings.
    Config {
        /// Bounded settings metadata.
        snapshot: ConfigWire,
    },
    /// Current remote-host endpoint, policy, and waiting control requests.
    Host { snapshot: HostWire },
    /// Requested context, usage, or throughput snapshot.
    Metrics { snapshot: MetricsWire },
    /// Detached model-update progress.
    Downloads {
        /// Detached model-update helper state and cancellation status.
        snapshot: DownloadsWire,
    },
    /// Requested page of cache lineage.
    Cache {
        /// Bounded cache lineage page.
        snapshot: CacheWire,
    },
    /// Requested inventory of running extensions.
    Extensions {
        /// Bounded inventory report and optional session switch.
        snapshot: ExtensionsWire,
    },
    /// Requested checkpoint or branch metadata.
    History {
        /// Bounded conversation metadata.
        snapshot: HistoryWire,
    },
    /// Rendered side-answer text, explicitly separate from the main conversation.
    Aside { role: String, text: String },
    /// Transient complete-file export event; absent from reconnect scrollback.
    Artifact {
        event: crate::export::transfer::Transfer,
    },
    /// The transcript was replaced (`/clear`, `/new`, `/switch`, `/resume`).
    /// Everything the client has shown so far belongs to a session that is
    /// gone: it clears its log rather than appending under a stale transcript.
    Reset,
    /// The turn finished. Carries the same headline and body as the local
    /// desktop notification so an attached client can raise its own; it is not
    /// transcript text and must not be logged as output.
    Notify {
        /// Notification headline.
        title: String,
        /// Notification body.
        body: String,
    },
    /// A slash command finished, even when it did not generate a model turn.
    CommandDone {
        /// Submitted command and arguments.
        text: String,
    },
    /// An identified remote prompt finished its own dispatcher turn.
    PromptDone { request_id: String },
    /// An identified slash command completed, independent of matching text.
    CommandRequestDone { request_id: String },
    /// Built-in commands and extensions loaded by the running agent.
    Controls {
        /// Versioned engine catalog, independent of the remote transport version.
        controls: serde_json::Value,
    },
    /// A structured question waiting for an explicit controller answer.
    Question {
        /// Question identity, options and single/multiple selection mode.
        request: serde_json::Value,
    },
    /// A question closed, including when the local operator answered it.
    QuestionDone {
        /// Identity of the closed question.
        question_id: String,
    },
    /// An answer was rejected without cancelling the running turn.
    AnswerRejected {
        /// Question the client attempted to answer.
        question_id: String,
        /// Why the answer could not be applied.
        reason: String,
    },
    /// A document waiting for explicit Save or Cancel.
    Editor {
        /// Editor identity, title and original text.
        request: serde_json::Value,
    },
    /// A document editor closed after submission or interruption.
    EditorDone {
        /// Identity of the closed editor.
        document_id: String,
    },
    /// A document reply could not be accepted; the turn remains active.
    EditRejected {
        /// Editor the client attempted to submit.
        document_id: String,
        /// Why the reply could not be applied.
        reason: String,
    },
    /// A control request from a non-controller was refused.
    ControlDenied {
        /// Human-readable reason.
        reason: String,
    },
    /// This session's control role changed after the `hello` frame: the local
    /// operator ran `/grant`, control was released back, or a grace window
    /// lapsed. Sent on transition only, from the session's own connection
    /// thread, so a client never has to infer its role from silence.
    Control {
        /// Whether this session now holds control.
        controller: bool,
    },
    /// Reply to a client `ping`.
    Pong,
    /// The server is closing this session.
    Bye {
        /// Human-readable reason.
        reason: String,
    },
}

impl ServerMsg {
    /// Maps a worker [`UiEvent`] to its wire message, or `None` if the event
    /// has no wire representation and must not be sent.
    #[must_use]
    pub fn from_event(ev: &UiEvent) -> Option<Self> {
        Some(match ev {
            // `Markdown` shares this arm deliberately: the wire has no document
            // frame, and a plan is markdown meant to be read as such, so it goes
            // down the one frame remote clients already render as markdown.
            UiEvent::Visible(t) | UiEvent::Markdown(t) => Self::Visible { text: t.clone() },
            UiEvent::Think(t) => Self::Think { text: t.clone() },
            UiEvent::Tool(t) => Self::Tool { text: t.clone() },
            UiEvent::Error(t) => Self::Error { text: t.clone() },
            UiEvent::Dim(t) => Self::Dim { text: t.clone() },
            // Remote clients get the status line pre-styled, like the card
            // below: the wire has no dedicated frame for it.
            UiEvent::SystemStatus(t) => Self::Plain {
                text: crate::status::system_line(t, true),
            },
            // Remote clients render the card as a pre-styled ANSI text block.
            UiEvent::EditCard(p) => Self::Dim {
                text: p.to_ansi(true),
            },
            UiEvent::Plain(t) => Self::Plain { text: t.clone() },
            UiEvent::SessionReset => Self::Reset,
            UiEvent::Notify { title, body } => Self::Notify {
                title: title.clone(),
                body: body.clone(),
            },
            UiEvent::CommandDone { text } => Self::CommandDone { text: text.clone() },
            UiEvent::PromptDone { request_id } => Self::PromptDone { request_id: request_id.clone() },
            UiEvent::CommandRequestDone { request_id } => Self::CommandRequestDone { request_id: request_id.clone() },
            UiEvent::Controls(controls) => Self::Controls { controls: controls.clone() },
            UiEvent::Question(request) => Self::Question { request: request.clone() },
            UiEvent::QuestionDone { id } => Self::QuestionDone { question_id: id.clone() },
            UiEvent::Editor(request) => Self::Editor { request: request.clone() },
            UiEvent::EditorDone { id } => Self::EditorDone { document_id: id.clone() },
            UiEvent::UserEcho(t) => Self::UserEcho { text: t.clone() },
            UiEvent::EndLine => Self::EndLine,
            UiEvent::BtwBegin => Self::BtwBegin,
            UiEvent::BtwEnd => Self::BtwEnd,
            UiEvent::MainCheckpoint => Self::MainCheckpoint,
            UiEvent::MainRollback => Self::MainRollback,
            UiEvent::Status(s) => Self::Status {
                status: StatusWire::from(s),
            },
            UiEvent::Tasks(tv) => {
                let (completed, total) = tv.counter().unwrap_or((0, 0));
                Self::Tasks { completed, total, report: tv.wire_report(), rows: tv.wire_rows().to_vec(), goal: tv.wire_goal().map(str::to_owned) }
            }
            UiEvent::Jobs(rows) => Self::Jobs { snapshot: rows.clone() },
            UiEvent::Sessions(snapshot) => Self::Sessions { snapshot: snapshot.clone() },
            UiEvent::Config(snapshot) => Self::Config { snapshot: snapshot.clone() },
            UiEvent::Downloads(snapshot) => Self::Downloads { snapshot: snapshot.clone() },
            UiEvent::Metrics(snapshot) => Self::Metrics { snapshot: snapshot.clone() },
            UiEvent::Host(snapshot) => Self::Host { snapshot: snapshot.clone() },
            UiEvent::Cache(snapshot) => Self::Cache { snapshot: snapshot.clone() },
            UiEvent::Extensions(snapshot) => Self::Extensions { snapshot: snapshot.clone() },
            UiEvent::History(snapshot) => Self::History { snapshot: snapshot.clone() },
            UiEvent::Artifact(event) => Self::Artifact { event: event.clone() },
            UiEvent::Btw(inner) => {
                let (role, text) = match inner.as_ref() {
                    UiEvent::Visible(text) | UiEvent::Markdown(text) | UiEvent::Plain(text) => ("answer", text),
                    UiEvent::Think(text) => ("thinking", text),
                    UiEvent::Tool(text) => ("tool", text),
                    UiEvent::Error(text) => ("error", text),
                    UiEvent::Dim(text) | UiEvent::UserEcho(text) | UiEvent::SystemStatus(text) => ("notice", text),
                    _ => return None,
                };
                Self::Aside { role: role.into(), text: text.clone() }
            },
            // Sub-agent groundwork (not wired up yet): no remote frame exists
            // for the sub-agent buffer, so these don't cross the wire. A later
            // task adds dedicated frames once a remote client can view it.
            // `QueuedJoined` is local pane state for a different reason: a
            // remote client never saw the queued prompt, so a payload-free
            // "it joined" would say nothing.
            UiEvent::SubStart { .. }
            | UiEvent::SubEnd
            | UiEvent::SubTokens { .. }
            | UiEvent::Sub(_)
            // The live preview counter rewrites a line in place, which the wire
            // cannot express; remote clients skip the tick and still receive the
            // permanent `└ N lines` summary as ordinary `Think` text.
            | UiEvent::PreviewStatus(_)
            | UiEvent::QueuedJoined => {
                return None;
            }
        })
    }

    /// True for a `status` frame (used by the connection loop's coalescing).
    #[must_use]
    fn is_status(&self) -> bool {
        matches!(self, Self::Status { .. })
    }
}

/// A server → client frame: the versioned envelope wrapping a [`ServerMsg`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ServerFrame {
    /// Protocol version.
    pub v: u32,
    /// Sequence id (bus id for mirrored events; `None` for control frames).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub id: Option<u64>,
    /// The message body.
    #[serde(flatten)]
    pub msg: ServerMsg,
}

impl ServerFrame {
    /// A frame with no sequence id (control/handshake frames).
    #[must_use]
    pub fn control(msg: ServerMsg) -> Self {
        Self {
            v: PROTOCOL_VERSION,
            id: None,
            msg,
        }
    }

    /// A frame carrying a bus sequence id (mirrored events).
    #[must_use]
    pub fn seq(id: u64, msg: ServerMsg) -> Self {
        Self {
            v: PROTOCOL_VERSION,
            id: Some(id),
            msg,
        }
    }

    /// Serializes to a JSON text string.
    ///
    /// # Errors
    /// Returns the `serde_json` error if serialization fails (never expected
    /// for these types).
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

/// Client → server message body, discriminated by `type` (design §4.3).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClientMsg {
    /// An ordinary prompt whose completion must be identified unambiguously.
    SubmitPrompt { request_id: String, text: String },
    /// A slash command with an unambiguous completion identity.
    SubmitCommand { request_id: String, text: String },
    /// Cancels only the identified input, rather than an unrelated host turn.
    CancelPrompt { request_id: String },
    /// Authenticates a mirror without acquiring or reclaiming control.
    ObserveAuth { token: String },
    /// Mandatory first frame.
    Auth {
        /// Shared bearer token.
        token: String,
        /// Optional resume point: replay only events with a greater id.
        #[serde(default)]
        resume_from: Option<u64>,
    },
    /// Submit a prompt (starts a turn or queues while busy).
    Prompt {
        /// Prompt text.
        text: String,
    },
    /// Submit a `/btw` side question (ephemeral; allowed from mirrors).
    Btw {
        /// Question text.
        text: String,
    },
    /// Interrupt the current turn.
    Interrupt,
    /// Answer the current structured question; restricted to the controller.
    Answer {
        /// Identity from the question frame.
        question_id: String,
        /// Explicit numbered choice(s), or zero to decline.
        answer: String,
    },
    /// Save a document or explicitly cancel, restricted to the controller.
    Edit {
        /// Identity from the editor frame.
        document_id: String,
        /// Explicit string content to save or JSON null to cancel.
        text: serde_json::Value,
    },
    /// A `/slash` command. Queued like a prompt and routed through the local
    /// slash dispatcher when the turn loop drains it (issue #25); the web client
    /// sends slash lines as `prompt` too, so both take the same path. Commands
    /// that cannot work remotely are refused with a `control_denied` carrying
    /// the reason — see [`crate::config::slash_command_remote_refusal`].
    Command {
        /// Command text including the leading slash.
        text: String,
    },
    /// Ask to become the controller.
    RequestControl,
    /// Give up control.
    ReleaseControl,
    /// Liveness probe; the server replies `pong`.
    Ping,
}

/// A client → server frame: the versioned envelope wrapping a [`ClientMsg`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ClientFrame {
    /// Protocol version.
    pub v: u32,
    /// Optional client-assigned id (echoed in errors; currently informational).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub id: Option<u64>,
    /// Opt-in support for complete-file transfer events on the initial auth frame.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub export_stream: Option<bool>,
    /// The message body.
    #[serde(flatten)]
    pub msg: ClientMsg,
}

impl ClientFrame {
    /// A frame wrapping `msg` at the current protocol version.
    #[must_use]
    pub fn new(msg: ClientMsg) -> Self {
        Self {
            v: PROTOCOL_VERSION,
            id: None,
            export_stream: None,
            msg,
        }
    }

    /// Serializes to a JSON text string.
    ///
    /// # Errors
    /// Returns the `serde_json` error if serialization fails.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Parses a frame from a JSON text string.
    ///
    /// # Errors
    /// Returns the `serde_json` error if the text is not a valid frame.
    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }
}

// --- Auth -------------------------------------------------------------------

/// Generates a 32-byte, base64url (unpadded) bearer token from the OS CSPRNG.
/// No default token exists; this is called when `/rc` starts a server without
/// an explicit token (design §4.6).
#[must_use]
pub fn generate_token() -> String {
    let mut bytes = [0u8; 32];
    if fill_random(&mut bytes).is_err() {
        // Extremely unlikely; fall back to time+addr entropy so we never hand
        // out an all-zero (predictable) token.
        let t = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0u128, |d| d.as_nanos());
        // Deliberate byte-wise truncation to spread entropy across the buffer.
        #[allow(clippy::cast_possible_truncation)]
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = ((t >> (i % 16 * 8)) as u8) ^ (i as u8).wrapping_mul(31);
        }
    }
    base64url(&bytes)
}

fn fill_random(buf: &mut [u8]) -> std::io::Result<()> {
    use std::io::Read;
    let mut f = std::fs::File::open("/dev/urandom")?;
    f.read_exact(buf)
}

/// Base64url encoding without padding (RFC 4648 §5).
fn base64url(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = u32::from(chunk[0]);
        let b1 = chunk.get(1).map_or(0, |b| u32::from(*b));
        let b2 = chunk.get(2).map_or(0, |b| u32::from(*b));
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(ALPHABET[((n >> 18) & 63) as usize] as char);
        out.push(ALPHABET[((n >> 12) & 63) as usize] as char);
        if chunk.len() > 1 {
            out.push(ALPHABET[((n >> 6) & 63) as usize] as char);
        }
        if chunk.len() > 2 {
            out.push(ALPHABET[(n & 63) as usize] as char);
        }
    }
    out
}

/// Constant-time byte-slice equality. Unequal lengths short-circuit (token
/// length is fixed and not secret), equal lengths compare without early exit.
#[must_use]
pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b) {
        diff |= x ^ y;
    }
    diff == 0
}

// --- Control policy ---------------------------------------------------------

/// Who currently holds control (may submit prompts / interrupts).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Holder {
    /// The local TUI/REPL user.
    Local,
    /// A remote session, by session id.
    Remote(u64),
    /// No one holds control (headless, between remote controllers).
    Free,
}

/// Outcome of a remote `request_control`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RequestOutcome {
    /// Control was granted to the requester.
    Granted,
    /// Denied; carries a reason for a `control_denied` frame.
    Denied(String),
    /// A local user is present and did not pre-authorize transfer: the request
    /// must be surfaced locally and wait for an explicit `/grant`.
    NeedsLocalGrant,
}

/// How long a dropped controller's slot is held for reconnection before it is
/// released to the local user / freed (design §4.8 reconnect grace window).
pub const CONTROL_GRACE: Duration = Duration::from_secs(15);

/// A queued input's accepting controller and grant generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ControlStamp {
    pub(crate) session: u64,
    epoch: u64,
}

/// One controller lease; queued inputs cannot outlive the grant that accepted them.
#[derive(Debug)]
pub struct ControlPolicy {
    holder: Holder,
    epoch: u64,
    local_present: bool,
    allow_control: bool,
    /// A controller that dropped and whose slot is reserved until the deadline
    /// for a reconnecting client to reclaim via `resume_from` (design §4.8).
    grace: Option<(u64, std::time::Instant)>,
    /// Sessions that asked for control and got [`RequestOutcome::NeedsLocalGrant`],
    /// oldest first. The local `/grant` command reads this: without it the
    /// requester's id lives only in the dim line broadcast at request time, and
    /// there is nothing for the operator to say yes *to*.
    pending: Vec<u64>,
}

impl ControlPolicy {
    /// New policy. With a local front-end present the local user holds control
    /// initially; headless starts [`Holder::Free`]. `allow_control` lets a
    /// remote take control without an explicit local `/grant`.
    #[must_use]
    pub fn new(local_present: bool, allow_control: bool) -> Self {
        Self {
            holder: if local_present {
                Holder::Local
            } else {
                Holder::Free
            },
            local_present,
            epoch: 0,
            allow_control,
            grace: None,
            pending: Vec::new(),
        }
    }

    /// Current holder.
    #[must_use]
    pub fn holder(&self) -> Holder {
        self.holder
    }

    fn new_holder(&mut self, holder: Holder) {
        self.epoch = self
            .epoch
            .checked_add(1)
            .expect("controller epoch exhausted");
        self.holder = holder;
        self.grace = None;
    }

    /// Reclaims control for explicit local input, invalidating the previous grant.
    pub fn local_input(&mut self) -> bool {
        self.local_present = true;
        if self.holder == Holder::Local {
            return false;
        }
        self.new_holder(Holder::Local);
        true
    }

    pub(crate) fn stamp(&mut self, session: u64) -> Option<ControlStamp> {
        self.expire_grace();
        self.remote_can_control(session).then_some(ControlStamp {
            session,
            epoch: self.epoch,
        })
    }

    /// Drops a stale grace reservation once its window has elapsed, releasing
    /// the reserved slot to the local user (if present) or freeing it. Called
    /// at the start of every decision so a lapsed reconnect never blocks.
    fn expire_grace(&mut self) {
        if let Some((session, deadline)) = self.grace
            && std::time::Instant::now() >= deadline
        {
            self.grace = None;
            if self.holder == Holder::Remote(session) {
                self.new_holder(if self.local_present {
                    Holder::Local
                } else {
                    Holder::Free
                });
            }
        }
    }

    /// Begins the reconnect grace window for a dropping controller: the slot is
    /// held (holder unchanged) until [`CONTROL_GRACE`] elapses, letting a
    /// reconnecting client reclaim it with [`ControlPolicy::reclaim`]. A
    /// non-controller disconnect just releases as before (design §4.8).
    pub fn disconnect(&mut self, session: u64) {
        // A client that hangs up while waiting is no longer waiting: otherwise
        // `/grant` hands control to a socket that is already gone.
        self.pending.retain(|s| *s != session);
        if self.holder == Holder::Remote(session) {
            self.grace = Some((session, std::time::Instant::now() + CONTROL_GRACE));
        } else {
            self.release(session);
        }
    }

    /// A reconnecting client reclaims a slot still inside its grace window,
    /// transferring control to the new session id. Returns whether control was
    /// reclaimed (design §4.8).
    pub fn reclaim(&mut self, new_session: u64) -> bool {
        self.expire_grace();
        if let Some((prev, _)) = self.grace
            && self.holder == Holder::Remote(prev)
        {
            self.new_holder(Holder::Remote(new_session));
            return true;
        }
        false
    }

    /// Whether the given remote session may currently submit control frames.
    #[must_use]
    pub fn remote_can_control(&self, session: u64) -> bool {
        self.holder == Holder::Remote(session)
    }

    /// A remote session requests control.
    pub fn request(&mut self, session: u64) -> RequestOutcome {
        self.expire_grace();
        if self.holder == Holder::Remote(session) {
            return RequestOutcome::Granted;
        }
        match self.holder {
            Holder::Remote(_) => RequestOutcome::Denied("another client holds control".to_owned()),
            Holder::Local if self.local_present && !self.allow_control => {
                // Remember the asker so `/grant` has a target. Re-requesting is
                // idempotent rather than a queue of duplicates.
                if !self.pending.contains(&session) {
                    self.pending.push(session);
                }
                RequestOutcome::NeedsLocalGrant
            }
            _ => {
                self.new_holder(Holder::Remote(session));
                RequestOutcome::Granted
            }
        }
    }

    /// Sessions awaiting a local `/grant`, oldest request first.
    #[must_use]
    pub fn pending(&self) -> &[u64] {
        &self.pending
    }

    /// The local operator grants control to a remote session (via `/grant`).
    ///
    /// The whole pending list is cleared, not just `session`: there is exactly
    /// one controller, so granting one request answers every other waiting
    /// request with "no". Those clients already hold a `control_denied`, and a
    /// fresh `request_control` re-queues them.
    pub fn grant(&mut self, session: u64) {
        self.new_holder(Holder::Remote(session));
        self.pending.clear();
    }

    /// Grants the oldest pending request, returning the session that got it.
    /// `None` when nothing is waiting — what `/grant` with no argument uses.
    pub fn grant_next(&mut self) -> Option<u64> {
        let session = self.pending.first().copied()?;
        self.grant(session);
        Some(session)
    }

    /// Drops a session's pending request without granting it — the local
    /// operator declining, or the client disconnecting before an answer.
    /// Returns whether a request was actually dropped.
    pub fn deny(&mut self, session: u64) -> bool {
        let before = self.pending.len();
        self.pending.retain(|s| *s != session);
        self.pending.len() != before
    }

    /// A remote session releases control (explicitly or on disconnect). Control
    /// returns to the local user if present, else becomes free.
    pub fn release(&mut self, session: u64) {
        if self.holder == Holder::Remote(session) {
            self.new_holder(if self.local_present {
                Holder::Local
            } else {
                Holder::Free
            });
        }
    }
}

// --- Server -----------------------------------------------------------------

/// Shared state a running remote server exposes to its connection threads and
/// to the rest of plank.
#[derive(Debug)]
pub struct RemoteState {
    /// Loopback WebSocket endpoint, without authentication credentials.
    pub endpoint: String,
    /// The event fan-out bus the worker broadcasts into.
    pub bus: Arc<BroadcastBus>,
    /// Per-turn shared state (interrupt / queued / btw).
    pub shared: Arc<TurnShared>,
    /// The single-controller policy.
    pub control: Mutex<ControlPolicy>,
    /// Shared bearer token every client must present in its `auth` frame.
    pub token: String,
    /// Browser `Origin`s allowed on the WS upgrade (besides missing/loopback).
    allowed_origins: Vec<String>,
    /// Per-client outbound queue cap in bytes (slow-consumer eviction).
    queue_max: usize,
    session_ids: AtomicU64,
    shutdown: AtomicBool,
}

impl RemoteState {
    /// Checks and queues input under the same controller-policy lock.
    fn enqueue_input(
        &self,
        session: u64,
        id: Option<String>,
        text: String,
        command: bool,
    ) -> Result<(), String> {
        let mut policy = self
            .control
            .lock()
            .map_err(|_| "control policy is poisoned")?;
        let owner = policy
            .stamp(session)
            .ok_or("this client does not hold control")?;
        match id {
            Some(id) => self.shared.push_remote_request(owner, id, text, command),
            None => self.shared.push_legacy_request(owner, text),
        }
    }

    /// Claims a queued input only while its original grant remains valid.
    pub(crate) fn begin_input(&self, id: &str, text: &str) -> bool {
        let mut policy = self.control.lock().expect("control policy poisoned");
        let stale = self
            .shared
            .request_owner(id)
            .is_some_and(|owner| policy.stamp(owner.session) != Some(owner));
        if !stale {
            return self.shared.begin_prompt_request(id);
        }
        drop(policy);
        self.reject_input(id, text);
        false
    }

    pub(crate) fn reject_input(&self, id: &str, text: &str) {
        self.bus.broadcast(UiEvent::Error(
            "Remote input rejected: control changed before dispatch\n".into(),
        ));
        if self.shared.request_is_legacy(id) && !text.trim_start().starts_with('/') {
            self.bus.broadcast(UiEvent::Notify {
                title: "Input rejected".into(),
                body: "Control changed before queued input started".into(),
            });
        }
        self.complete_input(id, text);
    }

    /// Retires input and preserves the submitting client's completion protocol.
    pub(crate) fn complete_input(&self, id: &str, text: &str) {
        let legacy = self.shared.request_is_legacy(id);
        let command = self.shared.is_command_request(id);
        self.shared.finish_prompt_request(id);
        let event = if legacy {
            if !command {
                return;
            }
            UiEvent::CommandDone { text: text.into() }
        } else if command {
            UiEvent::CommandRequestDone {
                request_id: id.into(),
            }
        } else {
            UiEvent::PromptDone {
                request_id: id.into(),
            }
        };
        self.bus.broadcast(event);
    }

    /// Creates a model-free bridge fixture without binding a network listener.
    #[cfg(test)]
    pub(crate) fn test_bridge() -> Arc<Self> {
        Arc::new(Self {
            endpoint: "ws://127.0.0.1:9000/".into(),
            bus: Arc::new(BroadcastBus::new()),
            shared: Arc::new(TurnShared::default()),
            control: Mutex::new(ControlPolicy::new(true, false)),
            token: "fixture".into(),
            allowed_origins: Vec::new(),
            queue_max: 1024,
            session_ids: AtomicU64::new(0),
            shutdown: AtomicBool::new(false),
        })
    }

    /// Captures live control state, expiring elapsed reconnect reservations.
    #[must_use]
    pub fn host_snapshot(&self) -> Option<HostWire> {
        let mut policy = self.control.lock().ok()?;
        policy.expire_grace();
        let (holder, controller) = match policy.holder() {
            Holder::Local => ("local", None),
            Holder::Free => ("free", None),
            Holder::Remote(id) => ("remote", Some(id)),
        };
        Some(HostWire {
            endpoint: Some(self.endpoint.clone()),
            holder: holder.into(),
            controller,
            automatic: policy.allow_control,
            reserved: policy.grace.is_some(),
            // Match the client's compact inventory bounds.
            pending: policy.pending().iter().copied().take(64).collect(),
            pending_total: policy.pending().len(),
        })
    }

    fn next_session(&self) -> u64 {
        self.session_ids.fetch_add(1, Ordering::Relaxed)
    }

    /// Tells every attached client the session is going away. Best-effort: the
    /// bus drops frames for a hung-up subscriber, exactly as for output.
    pub fn say_bye(&self, reason: &str) {
        self.bus.broadcast(crate::worker::UiEvent::Dim(format!(
            "[remote control off: {reason}]"
        )));
    }
}

/// A running remote-control server. Dropping it signals shutdown; call
/// [`RemoteServer::shutdown`] to stop deterministically.
#[derive(Debug)]
pub struct RemoteServer {
    /// Shared state (bus, turn state, control policy).
    pub state: Arc<RemoteState>,
    /// The actual bound address (useful when binding to port 0 in tests).
    pub local_addr: std::net::SocketAddr,
    accept: Option<JoinHandle<()>>,
}

/// Server-side configuration for a [`RemoteServer`]. Groups the bearer token,
/// control-policy seed, and the hardening knobs (`Origin` allow-list and the
/// per-client outbound queue cap) so [`RemoteServer::start`] stays a small,
/// stable signature.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Shared bearer secret every client must present.
    pub token: String,
    /// Whether a local front-end is present (seeds the control policy).
    pub local_present: bool,
    /// Whether a remote may take control without an explicit local `/grant`.
    pub allow_control: bool,
    /// Browser `Origin`s allowed on the WS upgrade (besides missing/loopback).
    pub allowed_origins: Vec<String>,
    /// Per-client outbound queue cap in bytes (slow-consumer eviction).
    pub queue_max: usize,
}

impl RemoteServer {
    /// Binds `addr` (loopback expected) and starts the accept thread. See
    /// [`ServerConfig`] for the auth / policy / hardening knobs.
    ///
    /// # Errors
    /// Returns an error if the address cannot be bound.
    pub fn start(
        addr: &str,
        cfg: ServerConfig,
        bus: Arc<BroadcastBus>,
        shared: Arc<TurnShared>,
    ) -> std::io::Result<Self> {
        let listener = TcpListener::bind(addr)?;
        // A short accept timeout lets the accept loop observe shutdown.
        listener.set_nonblocking(false)?;
        let local_addr = listener.local_addr()?;
        let state = Arc::new(RemoteState {
            endpoint: format!("ws://{local_addr}/"),
            bus,
            shared,
            control: Mutex::new(ControlPolicy::new(cfg.local_present, cfg.allow_control)),
            token: cfg.token,
            allowed_origins: cfg.allowed_origins,
            queue_max: cfg.queue_max,
            session_ids: AtomicU64::new(0),
            shutdown: AtomicBool::new(false),
        });
        let accept_state = Arc::clone(&state);
        let accept = std::thread::Builder::new()
            .name("plank-remote-accept".into())
            .spawn(move || accept_loop(&listener, &accept_state))?;
        Ok(Self {
            state,
            local_addr,
            accept: Some(accept),
        })
    }

    /// Signals shutdown and joins the accept thread.
    pub fn shutdown(&mut self) {
        self.state.shutdown.store(true, Ordering::Relaxed);
        // Nudge the blocking accept() by opening a throwaway connection.
        let _ = TcpStream::connect(self.local_addr);
        if let Some(h) = self.accept.take() {
            let _ = h.join();
        }
    }
}

impl Drop for RemoteServer {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn accept_loop(listener: &TcpListener, state: &Arc<RemoteState>) {
    for stream in listener.incoming() {
        if state.shutdown.load(Ordering::Relaxed) {
            break;
        }
        let Ok(stream) = stream else { continue };
        let conn_state = Arc::clone(state);
        let _ = std::thread::Builder::new()
            .name("plank-remote-conn".into())
            .spawn(move || {
                if let Err(e) = handle_connection(stream, &conn_state) {
                    // Connection errors are per-client and non-fatal.
                    let _ = e;
                }
            });
    }
}

/// Per-connection handler: reads the HTTP request head, serves the static web
/// client for a plain `GET /`, enforces the `Origin` allow-list on a WebSocket
/// upgrade, then runs auth + the mirror/control loop. Runs on its own thread;
/// a slow or dead client only affects itself.
fn handle_connection(mut stream: TcpStream, state: &Arc<RemoteState>) -> Result<(), String> {
    stream
        .set_read_timeout(Some(POLL_INTERVAL))
        .map_err(|e| e.to_string())?;
    // A bounded write timeout lets the outbound buffer build up (rather than
    // the thread blocking forever) when a client stops reading, so the queue
    // cap can evict it.
    let _ = stream.set_write_timeout(Some(POLL_INTERVAL));

    // Read the request head so we can both serve the static web client and
    // inspect `Origin` before completing any WebSocket upgrade.
    let head = read_http_head(&mut stream)?;
    let req = parse_request_head(&head);

    if !req.is_websocket_upgrade() {
        // Plain HTTP: serve the web client at `/`, else 404.
        serve_http(&mut stream, &req);
        return Ok(());
    }

    // Origin allow-list: missing / loopback Origins pass (native clients send
    // none); other browser Origins must be allow-listed or the upgrade is
    // refused with an HTTP 403 (design §8).
    if !origin_allowed(req.origin.as_deref(), &state.allowed_origins) {
        let _ = write_http_response(
            &mut stream,
            403,
            "Forbidden",
            "text/plain; charset=utf-8",
            b"origin not allowed\n",
        );
        return Ok(());
    }

    // Replay the consumed head bytes so tungstenite can parse the handshake
    // itself, and cap the outbound buffer for per-client backpressure.
    let prefixed = PrefixStream::new(head, stream);
    let config = tungstenite::protocol::WebSocketConfig::default()
        .write_buffer_size(0)
        .max_write_buffer_size(state.queue_max.max(1));
    let mut ws =
        tungstenite::accept_with_config(prefixed, Some(config)).map_err(|e| e.to_string())?;

    let Some((session_id, exports)) = do_handshake(&mut ws, state)? else {
        return Ok(()); // unauthorized; connection already closed
    };
    let loop_result = mirror_loop(&mut ws, state, session_id, exports);

    // On disconnect (including slow-consumer eviction) a controller's slot is
    // held for the reconnect grace window (§4.8) so a brief drop can reclaim it
    // via `resume_from`; a non-controller just releases.
    if let Ok(mut c) = state.control.lock() {
        c.disconnect(session_id);
    }
    let _ = ws.flush();
    loop_result
}

/// A minimal parsed HTTP request head: method, path, and the header values the
/// server cares about.
struct ParsedRequest {
    method: String,
    path: String,
    origin: Option<String>,
    upgrade_websocket: bool,
}

impl ParsedRequest {
    fn is_websocket_upgrade(&self) -> bool {
        self.upgrade_websocket
    }
}

/// Reads bytes from `stream` until the end of the HTTP header block
/// (`\r\n\r\n`) or a bounded deadline, tolerating read-timeout wake-ups. The
/// returned buffer is the raw head, replayed to tungstenite via [`PrefixStream`]
/// for a WebSocket upgrade.
fn read_http_head(stream: &mut TcpStream) -> Result<Vec<u8>, String> {
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    let mut buf = Vec::with_capacity(1024);
    let mut chunk = [0u8; 512];
    loop {
        if std::time::Instant::now() > deadline {
            return Err("timed out reading request head".to_owned());
        }
        if buf.windows(4).any(|w| w == b"\r\n\r\n") {
            break;
        }
        // Guard against an unbounded head.
        if buf.len() > 64 * 1024 {
            return Err("request head too large".to_owned());
        }
        match stream.read(&mut chunk) {
            Ok(0) => break, // EOF
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
            Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {}
            Err(e) => return Err(e.to_string()),
        }
    }
    Ok(buf)
}

/// Parses the request line and the headers we care about from a raw head.
/// Header names are matched case-insensitively (HTTP header names are
/// case-insensitive).
fn parse_request_head(head: &[u8]) -> ParsedRequest {
    let text = String::from_utf8_lossy(head);
    let mut lines = text.split("\r\n");
    let request_line = lines.next().unwrap_or("");
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_owned();
    let path = parts.next().unwrap_or("/").to_owned();

    let mut origin = None;
    let mut upgrade_websocket = false;
    for line in lines {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        let name = name.trim().to_ascii_lowercase();
        let value = value.trim();
        match name.as_str() {
            "origin" => origin = Some(value.to_owned()),
            "upgrade" if value.eq_ignore_ascii_case("websocket") => {
                upgrade_websocket = true;
            }
            _ => {}
        }
    }
    ParsedRequest {
        method,
        path,
        origin,
        upgrade_websocket,
    }
}

/// Decides whether a WebSocket upgrade with the given `Origin` is allowed.
/// Default policy (design §8): a missing Origin (native clients send none) and
/// any loopback Origin are allowed; every other browser Origin must appear in
/// the configured allow-list.
#[must_use]
pub fn origin_allowed(origin: Option<&str>, allowed: &[String]) -> bool {
    match origin {
        None => true,
        // Browsers send the literal "null" for opaque origins (e.g. a page
        // opened from a file:// URL); treat it as a non-loopback browser origin
        // that must be explicitly allow-listed.
        Some(o) => is_loopback_origin(o) || allowed.iter().any(|a| a.eq_ignore_ascii_case(o)),
    }
}

/// True if an `Origin` value refers to a loopback host (any scheme / port).
fn is_loopback_origin(origin: &str) -> bool {
    let after_scheme = origin.split_once("://").map_or(origin, |(_, rest)| rest);
    let host_port = after_scheme.split('/').next().unwrap_or(after_scheme);
    // Strip a trailing :port, taking care not to break an IPv6 literal.
    let host = if let Some(stripped) = host_port.strip_prefix('[') {
        stripped.split(']').next().unwrap_or(stripped)
    } else {
        host_port.rsplit_once(':').map_or(host_port, |(h, _)| h)
    };
    matches!(host, "localhost" | "127.0.0.1" | "::1")
}

/// Serves a plain HTTP request: the web client at `/` (or `/index.html`), its
/// logo at `/logo.png`, else a 404. Best-effort; write errors just drop the
/// connection. The query string is stripped before routing, so the tokenized
/// link `/?t=TOKEN` that `/remote-control` prints lands on the client page.
fn serve_http(stream: &mut TcpStream, req: &ParsedRequest) {
    let path = req.path.split('?').next().unwrap_or(&req.path);
    if !req.method.eq_ignore_ascii_case("GET") {
        let _ = write_http_response(
            stream,
            404,
            "Not Found",
            "text/plain; charset=utf-8",
            b"not found\n",
        );
        return;
    }
    if path == "/logo.png" {
        let _ = write_http_response(stream, 200, "OK", "image/png", WEB_CLIENT_LOGO);
    } else if matches!(path, "/" | "/index.html") {
        let _ = write_http_response(
            stream,
            200,
            "OK",
            "text/html; charset=utf-8",
            WEB_CLIENT_HTML.as_bytes(),
        );
    } else {
        let _ = write_http_response(
            stream,
            404,
            "Not Found",
            "text/plain; charset=utf-8",
            b"not found\n",
        );
    }
}

/// Writes a minimal HTTP/1.1 response and closes (no keep-alive).
fn write_http_response(
    stream: &mut TcpStream,
    status: u16,
    reason: &str,
    content_type: &str,
    body: &[u8],
) -> std::io::Result<()> {
    let header = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {len}\r\nConnection: close\r\n\r\n",
        len = body.len()
    );
    stream.write_all(header.as_bytes())?;
    stream.write_all(body)?;
    stream.flush()
}

/// A read/write adapter that replays a consumed byte prefix before delegating
/// to the underlying stream. Lets us peek the HTTP head (for static serving and
/// `Origin` inspection) and still hand the full handshake request to
/// tungstenite.
struct PrefixStream<S> {
    prefix: std::io::Cursor<Vec<u8>>,
    inner: S,
}

impl<S> PrefixStream<S> {
    fn new(prefix: Vec<u8>, inner: S) -> Self {
        Self {
            prefix: std::io::Cursor::new(prefix),
            inner,
        }
    }
}

impl<S: Read> Read for PrefixStream<S> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.prefix.read(buf)?;
        if n > 0 {
            return Ok(n);
        }
        self.inner.read(buf)
    }
}

impl<S: Write> Write for PrefixStream<S> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.inner.write(buf)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// Authenticates, assigns a session id, and sends the `hello` + `snapshot`
/// frames. Returns the session id and export support, or `None` if the client was
/// unauthorized (the connection is closed in that case).
fn do_handshake<S: std::io::Read + std::io::Write>(
    ws: &mut tungstenite::WebSocket<S>,
    state: &Arc<RemoteState>,
) -> Result<Option<(u64, bool)>, String> {
    let (resume_from, ok, observer, exports) = authenticate(ws, state)?;
    if !ok {
        let _ = ws.close(Some(tungstenite::protocol::CloseFrame {
            code: tungstenite::protocol::frame::coding::CloseCode::Library(4401),
            reason: "unauthorized".into(),
        }));
        let _ = ws.flush();
        return Ok(None);
    }

    let session_id = state.next_session();

    // A client presenting `resume_from` is reconnecting: reclaim the dropped
    // controller's slot if it is still inside its grace window (§4.8). Failing
    // that, headless (no local front-end) auto-requests control for
    // scriptability (design open-question §8, leaning auto-grant).
    let controller = {
        let mut c = state.control.lock().map_err(|e| e.to_string())?;
        initial_control(&mut c, session_id, resume_from, observer)
    };

    send(
        ws,
        &ServerFrame::control(ServerMsg::Hello {
            protocol_version: PROTOCOL_VERSION,
            plank_version: env!("CARGO_PKG_VERSION").to_owned(),
            session_id,
            controller,
            command_completion: true,
            prompt_completion: true,
            command_request_completion: true,
            ownership_fencing: true,
        }),
    )?;

    // Snapshot: replay scrollback tail (or events newer than resume_from).
    let (tail, highest_id) = state.bus.scrollback_since(resume_from);
    let scrollback = tail
        .into_iter()
        .filter_map(|s| {
            ServerMsg::from_event(&s.event).map(|msg| ScrollbackEntry { id: s.id, msg })
        })
        .collect();
    send(
        ws,
        &ServerFrame::control(ServerMsg::Snapshot {
            scrollback,
            highest_id,
        }),
    )?;
    Ok(Some((session_id, exports)))
}

/// The post-handshake loop: pump bus events to the socket and handle inbound
/// control frames until the client disconnects or the server shuts down.
fn mirror_loop<S: std::io::Read + std::io::Write>(
    ws: &mut tungstenite::WebSocket<S>,
    state: &Arc<RemoteState>,
    session_id: u64,
    exports: bool,
) -> Result<(), String> {
    // Subscribe after the snapshot so no event is missed or duplicated.
    let rx = if exports {
        MirrorReceiver::Exports(Box::new(state.bus.subscribe_exports()))
    } else {
        MirrorReceiver::Ordinary(state.bus.subscribe())
    };
    let mut last_status_at: Option<std::time::Instant> = None;
    let mut last_download_poll: Option<std::time::Instant> = None;
    let mut last_downloads = None;
    let mut last_host = None;
    // The role this client was last told about (the `hello` frame's value).
    // Polled rather than pushed: a `/grant` happens on the local UI thread,
    // which has no socket, and the bus carries transcript events for every
    // client rather than per-session facts. One `Mutex` read per poll tick.
    let mut had_control = is_controller(state, session_id)?;
    loop {
        if state.shutdown.load(Ordering::Relaxed) {
            let _ = send(
                ws,
                &ServerFrame::control(ServerMsg::Bye {
                    reason: "server shutting down".to_owned(),
                }),
            );
            break;
        }
        // Pump bus → socket, coalescing status frames.
        pump_bus(ws, &rx, &mut last_status_at)?;
        // Downloads are detached from model turns; a quiet/busy engine still
        // delivers updates. Read one small state file per second per connection.
        if last_download_poll.is_none_or(|last| last.elapsed() >= Duration::from_secs(1)) {
            if let Some(snapshot) = state.host_snapshot()
                && last_host.as_ref() != Some(&snapshot)
            {
                send(
                    ws,
                    &ServerFrame::control(ServerMsg::Host {
                        snapshot: snapshot.clone(),
                    }),
                )?;
                last_host = Some(snapshot);
            }
            let snapshot = DownloadsWire::capture();
            if last_downloads.as_ref() != Some(&snapshot) {
                send(
                    ws,
                    &ServerFrame::control(ServerMsg::Downloads {
                        snapshot: snapshot.clone(),
                    }),
                )?;
                last_downloads = Some(snapshot);
            }
            last_download_poll = Some(std::time::Instant::now());
        }

        // Tell the client when its role changed under it (`/grant`, a release,
        // a lapsed grace window). Edge-triggered: no frame while it is stable.
        let now_control = is_controller(state, session_id)?;
        if now_control != had_control {
            had_control = now_control;
            send(
                ws,
                &ServerFrame::control(ServerMsg::Control {
                    controller: now_control,
                }),
            )?;
        }

        // Poll the socket for one inbound frame (times out per POLL_INTERVAL).
        match ws.read() {
            Ok(Message::Text(txt)) => {
                if handle_client_frame(ws, state, session_id, &txt)? {
                    break; // client asked to close / fatal
                }
            }
            Ok(Message::Close(_)) => break,
            Ok(Message::Ping(p)) => {
                let _ = ws.send(Message::Pong(p));
            }
            Ok(_) => {}
            Err(tungstenite::Error::Io(e))
                if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {}
            Err(_) => break,
        }
    }
    Ok(())
}

/// Reads and validates the mandatory authentication frame. Returns
/// `(resume_from, authorized, observer)`. Other first frames are policy violations
/// (returned as unauthorized).
fn initial_control(
    policy: &mut ControlPolicy,
    session: u64,
    resume: Option<u64>,
    observer: bool,
) -> bool {
    if observer {
        return false;
    }
    if resume.is_some() && policy.reclaim(session) {
        true
    } else if matches!(policy.holder(), Holder::Free) {
        matches!(policy.request(session), RequestOutcome::Granted)
    } else {
        policy.remote_can_control(session)
    }
}

fn authenticate<S: std::io::Read + std::io::Write>(
    ws: &mut tungstenite::WebSocket<S>,
    state: &RemoteState,
) -> Result<(Option<u64>, bool, bool, bool), String> {
    // Wait (bounded) for the first text frame.
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        if std::time::Instant::now() > deadline {
            return Ok((None, false, false, false));
        }
        match ws.read() {
            Ok(Message::Text(txt)) => {
                let Ok(frame) = ClientFrame::from_json(&txt) else {
                    return Ok((None, false, false, false));
                };
                let exports = frame.export_stream.unwrap_or(false);
                return match frame.msg {
                    ClientMsg::Auth { token, resume_from } => {
                        let ok = constant_time_eq(token.as_bytes(), state.token.as_bytes());
                        Ok((resume_from, ok, false, exports))
                    }
                    ClientMsg::ObserveAuth { token } => Ok((
                        None,
                        constant_time_eq(token.as_bytes(), state.token.as_bytes()),
                        true,
                        exports,
                    )),
                    // Anything but auth first is a policy violation.
                    _ => Ok((None, false, false, false)),
                };
            }
            Ok(Message::Close(_)) => return Ok((None, false, false, false)),
            Ok(Message::Ping(p)) => {
                let _ = ws.send(Message::Pong(p));
            }
            Ok(_) => {}
            Err(tungstenite::Error::Io(e))
                if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {}
            Err(e) => return Err(e.to_string()),
        }
    }
}

#[derive(Debug)]
enum MirrorReceiver {
    Ordinary(std::sync::mpsc::Receiver<SeqEvent>),
    Exports(Box<crate::worker::ExportReceiver>),
}

impl MirrorReceiver {
    fn try_recv(&self) -> Result<SeqEvent, std::sync::mpsc::TryRecvError> {
        match self {
            Self::Ordinary(rx) => rx.try_recv(),
            Self::Exports(rx) => rx.try_recv(),
        }
    }
}

/// Drains available bus events to the socket, coalescing status frames.
fn pump_bus<S: std::io::Read + std::io::Write>(
    ws: &mut tungstenite::WebSocket<S>,
    rx: &MirrorReceiver,
    last_status_at: &mut Option<std::time::Instant>,
) -> Result<(), String> {
    let mut pending_status: Option<SeqEvent> = None;
    while let Ok(seq) = rx.try_recv() {
        let Some(msg) = ServerMsg::from_event(&seq.event) else {
            continue;
        };
        if msg.is_status() {
            pending_status = Some(seq); // keep only the newest
            continue;
        }
        send(ws, &ServerFrame::seq(seq.id, msg))?;
    }
    if let Some(seq) = pending_status {
        let now = std::time::Instant::now();
        let due = last_status_at.is_none_or(|prev| now.duration_since(prev) >= STATUS_MIN_INTERVAL);
        if due {
            if let Some(msg) = ServerMsg::from_event(&seq.event) {
                send(ws, &ServerFrame::seq(seq.id, msg))?;
            }
            *last_status_at = Some(now);
        }
    }
    // Push any buffered data from a prior transient stall; overflow evicts.
    flush_ws(ws)?;
    Ok(())
}

/// Queues a controller's identified prompt or replies with a denial.
fn queue_identified_prompt<S: std::io::Read + std::io::Write>(
    ws: &mut tungstenite::WebSocket<S>,
    state: &Arc<RemoteState>,
    session: u64,
    request_id: String,
    text: String,
) -> Result<(), String> {
    let result = state.enqueue_input(session, Some(request_id), text, false);
    if let Err(reason) = result {
        send(
            ws,
            &ServerFrame::control(ServerMsg::ControlDenied { reason }),
        )?;
    }
    Ok(())
}

/// Queues identified controls after checking controller and remote availability.
fn queue_identified_command<S: std::io::Read + std::io::Write>(
    ws: &mut tungstenite::WebSocket<S>,
    state: &Arc<RemoteState>,
    session: u64,
    request_id: String,
    text: String,
) -> Result<(), String> {
    let result = if let Some(reason) = crate::config::slash_command_remote_refusal(&text) {
        Err(reason.into())
    } else {
        state.enqueue_input(session, Some(request_id), text, true)
    };
    if let Err(reason) = result {
        send(
            ws,
            &ServerFrame::control(ServerMsg::ControlDenied { reason }),
        )?;
    }
    Ok(())
}

/// Cancels a controller's identified input.
fn cancel_identified_prompt<S: std::io::Read + std::io::Write>(
    ws: &mut tungstenite::WebSocket<S>,
    state: &Arc<RemoteState>,
    session: u64,
    request_id: &str,
) -> Result<(), String> {
    let completion = {
        let control = state.control.lock().map_err(|error| error.to_string())?;
        if control.remote_can_control(session) && state.shared.cancel_prompt_request(request_id) {
            let completion = if state.shared.is_command_request(request_id) {
                ServerMsg::CommandRequestDone {
                    request_id: request_id.to_owned(),
                }
            } else {
                ServerMsg::PromptDone {
                    request_id: request_id.to_owned(),
                }
            };
            state.shared.finish_prompt_request(request_id);
            Some(completion)
        } else {
            None
        }
    };
    if let Some(completion) = completion {
        send(ws, &ServerFrame::control(completion))?;
    }
    Ok(())
}

/// Queues compatible legacy input after checking control and pane availability.
fn queue_legacy_input<S: std::io::Read + std::io::Write>(
    ws: &mut tungstenite::WebSocket<S>,
    state: &Arc<RemoteState>,
    session: u64,
    text: String,
) -> Result<(), String> {
    if !is_controller(state, session)? {
        return send(
            ws,
            &ServerFrame::control(ServerMsg::ControlDenied {
                reason: "not the controller".into(),
            }),
        );
    }
    // Refuse host-only panes before their dispatcher could open one invisibly.
    if let Some(reason) = crate::config::slash_command_remote_refusal(&text) {
        return send(
            ws,
            &ServerFrame::control(ServerMsg::ControlDenied {
                reason: reason.into(),
            }),
        );
    }
    let command = text.trim_start().starts_with('/');
    match state.enqueue_input(session, None, text, command) {
        Ok(()) => Ok(()),
        Err(reason) => send(
            ws,
            &ServerFrame::control(ServerMsg::ControlDenied { reason }),
        ),
    }
}

/// Handles one inbound client frame. Returns `Ok(true)` to close the connection.
fn handle_client_frame<S: std::io::Read + std::io::Write>(
    ws: &mut tungstenite::WebSocket<S>,
    state: &Arc<RemoteState>,
    session_id: u64,
    txt: &str,
) -> Result<bool, String> {
    let Ok(frame) = ClientFrame::from_json(txt) else {
        // Ignore unparseable frames rather than dropping the connection.
        return Ok(false);
    };
    match frame.msg {
        // Re-auth mid-session is a no-op (already authenticated).
        ClientMsg::Auth { .. } | ClientMsg::ObserveAuth { .. } => {}
        ClientMsg::Ping => send(ws, &ServerFrame::control(ServerMsg::Pong))?,
        ClientMsg::SubmitPrompt { request_id, text } => {
            queue_identified_prompt(ws, state, session_id, request_id, text)?;
        }
        ClientMsg::SubmitCommand { request_id, text } => {
            queue_identified_command(ws, state, session_id, request_id, text)?;
        }
        ClientMsg::CancelPrompt { request_id } => {
            cancel_identified_prompt(ws, state, session_id, &request_id)?;
        }
        // `/btw` is ephemeral and allowed from mirrors (design §4.4/§7).
        ClientMsg::Btw { text } => {
            let _ = state.shared.push_btw(text);
        }
        ClientMsg::Prompt { text } | ClientMsg::Command { text } => {
            queue_legacy_input(ws, state, session_id, text)?;
        }
        ClientMsg::Interrupt => {
            let allowed = {
                let control = state.control.lock().map_err(|error| error.to_string())?;
                if control.remote_can_control(session_id) {
                    state.shared.interrupt.store(true, Ordering::Relaxed);
                    true
                } else {
                    false
                }
            };
            if !allowed {
                send(
                    ws,
                    &ServerFrame::control(ServerMsg::ControlDenied {
                        reason: "not the controller".to_owned(),
                    }),
                )?;
            }
        }
        ClientMsg::Answer {
            question_id,
            answer,
        } => {
            if !accept_question_answer(state, session_id, &question_id, &answer)? {
                send(
                    ws,
                    &ServerFrame::control(ServerMsg::AnswerRejected {
                        question_id,
                        reason: "question answer is stale, invalid, or not from the controller"
                            .into(),
                    }),
                )?;
            }
        }
        ClientMsg::Edit { document_id, text } => {
            handle_editor_reply(ws, state, session_id, document_id, &text)?;
        }
        ClientMsg::RequestControl => {
            let outcome = state
                .control
                .lock()
                .map_err(|e| e.to_string())?
                .request(session_id);
            match outcome {
                RequestOutcome::Granted => {}
                RequestOutcome::Denied(reason) => {
                    send(
                        ws,
                        &ServerFrame::control(ServerMsg::ControlDenied { reason }),
                    )?;
                }
                RequestOutcome::NeedsLocalGrant => {
                    // Surface the request to the local user; the request is now
                    // recorded as pending, and `Agent::grant_lines` answers it.
                    state.bus.broadcast(UiEvent::Dim(format!(
                        "[remote session {session_id} wants control — /grant or /grant {session_id} to allow]"
                    )));
                    send(
                        ws,
                        &ServerFrame::control(ServerMsg::ControlDenied {
                            reason: "awaiting local /grant".to_owned(),
                        }),
                    )?;
                }
            }
        }
        ClientMsg::ReleaseControl => {
            state
                .control
                .lock()
                .map_err(|e| e.to_string())?
                .release(session_id);
        }
    }
    Ok(false)
}

fn accept_question_answer(
    state: &RemoteState,
    session: u64,
    id: &str,
    answer: &str,
) -> Result<bool, String> {
    let control = state.control.lock().map_err(|error| error.to_string())?;
    Ok(control.remote_can_control(session) && state.shared.questions.answer(id, answer))
}

fn handle_editor_reply<S: std::io::Read + std::io::Write>(
    ws: &mut tungstenite::WebSocket<S>,
    state: &RemoteState,
    session: u64,
    id: String,
    text: &serde_json::Value,
) -> Result<(), String> {
    if !accept_editor_reply(state, session, &id, text)? {
        send(
            ws,
            &ServerFrame::control(ServerMsg::EditRejected {
                document_id: id,
                reason: "document reply is stale, invalid, or not from the controller".into(),
            }),
        )?;
    }
    Ok(())
}

fn accept_editor_reply(
    state: &RemoteState,
    session: u64,
    id: &str,
    text: &serde_json::Value,
) -> Result<bool, String> {
    let control = state.control.lock().map_err(|error| error.to_string())?;
    Ok(control.remote_can_control(session) && state.shared.editors.reply(id, text))
}

fn is_controller(state: &Arc<RemoteState>, session_id: u64) -> Result<bool, String> {
    Ok(state
        .control
        .lock()
        .map_err(|e| e.to_string())?
        .remote_can_control(session_id))
}

/// Error string used to signal slow-consumer eviction: the client's bounded
/// outbound queue overflowed. Returned up through the pump so the connection
/// loop tears down (the bus prunes the dropped subscriber on the next
/// broadcast).
const EVICT_SLOW: &str = "outbound queue full (slow consumer evicted)";

/// True for a transient write stall (buffered, retry later), not a hard error.
fn is_would_block(e: &tungstenite::Error) -> bool {
    matches!(e, tungstenite::Error::Io(io) if matches!(io.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut))
}

/// Queues `frame` on the connection's bounded outbound buffer and tries to
/// flush. A transient write stall keeps the frame buffered (returns `Ok`); an
/// overflow of the configured cap returns [`EVICT_SLOW`].
fn send<S: std::io::Read + std::io::Write>(
    ws: &mut tungstenite::WebSocket<S>,
    frame: &ServerFrame,
) -> Result<(), String> {
    let json = frame.to_json().map_err(|e| e.to_string())?;
    match ws.write(Message::Text(json.into())) {
        Ok(()) => {}
        Err(tungstenite::Error::WriteBufferFull(_)) => return Err(EVICT_SLOW.to_owned()),
        Err(e) if is_would_block(&e) => return Ok(()), // buffered; flush later
        Err(e) => return Err(e.to_string()),
    }
    flush_ws(ws)
}

/// Flushes buffered outbound data. A transient stall is not an error; a cap
/// overflow evicts.
fn flush_ws<S: std::io::Read + std::io::Write>(
    ws: &mut tungstenite::WebSocket<S>,
) -> Result<(), String> {
    match ws.flush() {
        Ok(()) => Ok(()),
        Err(tungstenite::Error::WriteBufferFull(_)) => Err(EVICT_SLOW.to_owned()),
        Err(e) if is_would_block(&e) => Ok(()),
        Err(e) => Err(e.to_string()),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn export_transfer_auth_opt_in_preserves_legacy_frame_bytes() {
        let legacy = ClientFrame::new(ClientMsg::Auth {
            token: "test".into(),
            resume_from: None,
        });
        assert!(!legacy.to_json().unwrap().contains("export_stream"));
        let modern =
            ClientFrame::from_json(r#"{"v":1,"type":"auth","token":"test","export_stream":true}"#)
                .unwrap();
        assert_eq!(modern.export_stream, Some(true));
        let legacy =
            ClientFrame::from_json(r#"{"v":1,"type":"observe_auth","token":"test"}"#).unwrap();
        assert_eq!(legacy.export_stream, None);
    }
    #[test]
    fn gui_saved_sessions_preserve_ids_and_bound_pages() {
        let entries: Vec<_> = (0..130)
            .map(|index| crate::session::SessionEntry {
                id: format!("session-{index}"),
                title: "é\"\n".repeat(2048),
                cwd: "界".repeat(2048),
                tag: "tag".repeat(2048),
                last_used: 10,
                ..crate::session::SessionEntry::default()
            })
            .collect();
        let page = super::SessionsWire::capture(&entries, 2);
        assert_eq!(page.total, 130);
        assert_eq!(page.page, 2);
        assert_eq!(page.sessions.len(), 64);
        assert_eq!(page.sessions[0].id, "session-64");
        assert_eq!(page.sessions[63].id, "session-127");
        assert!(
            page.sessions
                .iter()
                .all(|session| session.title.len() <= 1024
                    && session.tag.len() <= 1024
                    && session.cwd.len() <= 1024)
        );
        let frame = super::ServerFrame::seq(
            1,
            super::ServerMsg::from_event(&crate::worker::UiEvent::Sessions(page.clone())).unwrap(),
        );
        let encoded = serde_json::to_vec(&frame).unwrap();
        assert!(encoded.len() < 2 * 1024 * 1024);
        let decoded: super::ServerFrame = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded, frame);
        let last = super::SessionsWire::capture(&entries, usize::MAX);
        assert_eq!(last.page, 3);
        assert_eq!(last.sessions.len(), 2);
        let empty = super::SessionsWire::capture(&[], 0);
        assert_eq!(empty.page, 1);
        assert!(empty.sessions.is_empty());
    }

    #[test]
    fn gui_jobs_prioritize_running_rows_and_bound_escaped_wire_payloads() {
        use crate::tools::bash::{JobRow, JobState};
        let rows: Vec<_> = (1..=100)
            .map(|id| JobRow {
                id,
                pid: 10,
                started: std::time::Instant::now(),
                state: if id > 80 {
                    JobState::Running
                } else {
                    JobState::Done(0)
                },
                path: std::path::PathBuf::from("\u{1}".repeat(10_000)),
            })
            .collect();
        let snapshot = super::JobsWire::capture(&rows);
        assert_eq!(
            (snapshot.running, snapshot.total, snapshot.jobs.len()),
            (20, 100, 64)
        );
        assert!(snapshot.jobs[..20].iter().all(|job| job.state == "running"));
        assert!(snapshot.jobs.iter().all(|job| job.path.len() <= 4096));
        let wire = serde_json::to_string(&super::ServerMsg::Jobs { snapshot }).unwrap();
        assert!(wire.len() < 2 * 1024 * 1024);
        let mut unicode = rows[0].clone();
        unicode.path = std::path::PathBuf::from("🦀".repeat(10_000));
        let path = &super::JobsWire::capture(&[unicode]).jobs[0].path;
        assert!(path.len() <= 4096 && path.ends_with("..."));
    }
    #[test]
    fn remote_controls_never_enter_the_model_prompt_queue() {
        let remote = RemoteState::test_bridge();
        remote.control.lock().unwrap().grant(7);
        for text in ["ordinary prompt", "  /usage", "/custom-skill task"] {
            remote
                .enqueue_input(7, None, text.into(), text.trim_start().starts_with('/'))
                .unwrap();
        }
        assert!(remote.shared.take_queued().is_empty());
        assert!(remote.shared.take_commands().is_empty());
        let inputs = remote.shared.take_prompt_requests();
        assert_eq!(
            inputs
                .iter()
                .map(|(_, text)| text.as_str())
                .collect::<Vec<_>>(),
            ["ordinary prompt", "  /usage", "/custom-skill task"]
        );
        for (id, _) in inputs {
            assert!(remote.shared.request_owner(&id).is_some());
            assert!(remote.shared.request_is_legacy(&id));
        }
    }

    use super::*;

    #[test]
    fn ownership_fencing_rejects_prior_grants_even_after_the_same_client_reacquires() {
        let remote = RemoteState::test_bridge();
        remote.control.lock().unwrap().grant(7);
        remote
            .enqueue_input(7, Some("old".into()), "queued prompt".into(), false)
            .unwrap();
        {
            let mut policy = remote.control.lock().unwrap();
            policy.disconnect(7);
            assert!(policy.local_input());
            assert!(
                !policy.reclaim(8),
                "local takeover must retire reconnect reservation"
            );
            policy.grant(7);
        }
        let requests = remote.shared.take_prompt_requests();
        assert!(!remote.begin_input(&requests[0].0, &requests[0].1));
        assert!(!remote.shared.request_is_pending("old"));
        assert!(!remote.shared.cancel_prompt_request("old"));
        assert!(!remote.shared.interrupt.load(Ordering::Relaxed));
        let (events, _) = remote.bus.scrollback_since(None);
        assert_eq!(events.iter().filter(|event| matches!(&event.event, UiEvent::PromptDone {request_id} if request_id == "old")).count(), 1);
        assert!(
            remote
                .enqueue_input(7, Some("old".into()), "replay".into(), false)
                .is_err()
        );
    }

    #[test]
    fn ownership_fencing_covers_legacy_and_identified_inputs_and_preserves_new_grants() {
        let remote = RemoteState::test_bridge();
        remote.control.lock().unwrap().grant(7);
        remote
            .enqueue_input(7, Some("command-old".into()), "/command".into(), true)
            .unwrap();
        remote
            .enqueue_input(7, None, "/usage".into(), true)
            .unwrap();
        remote
            .enqueue_input(7, None, "old legacy prompt".into(), false)
            .unwrap();
        remote.control.lock().unwrap().grant(8);
        remote
            .enqueue_input(8, Some("new".into()), "new prompt".into(), false)
            .unwrap();
        let requests = remote.shared.take_prompt_requests();
        for (id, text) in &requests[..3] {
            assert!(!remote.begin_input(id, text));
        }
        assert!(remote.begin_input(&requests[3].0, &requests[3].1));
        remote.complete_input(&requests[3].0, &requests[3].1);
        let (events, _) = remote.bus.scrollback_since(None);
        assert!(events.iter().any(|event| matches!(&event.event, UiEvent::CommandRequestDone {request_id} if request_id == "command-old")));
        assert!(
            events.iter().any(
                |event| matches!(&event.event, UiEvent::CommandDone {text} if text == "/usage")
            )
        );
        assert!(events.iter().any(
            |event| matches!(&event.event, UiEvent::PromptDone {request_id} if request_id == "new")
        ));
        assert!(!remote.shared.has_prompt_requests());
        assert!(!remote.shared.request_is_pending("new"));
    }

    // --- protocol round-trip ---

    /// The end-of-turn notification crosses the wire: locally it is a desktop
    /// notification, which reaches only whoever is at that machine.
    #[test]
    fn notify_event_reaches_the_wire_with_both_fields() {
        let ev = UiEvent::Notify {
            title: "finished: rename the thing".into(),
            body: "…done.".into(),
        };
        let Some(msg) = ServerMsg::from_event(&ev) else {
            panic!("notify must have a wire representation");
        };
        let json = ServerFrame::control(msg).to_json().unwrap();
        assert!(json.contains(r#""type":"notify""#), "{json}");
        assert!(json.contains("finished: rename the thing"), "{json}");
        assert!(json.contains("…done."), "{json}");
    }

    /// A remote header shows the same "where am I?" segments as the local
    /// footer, so they ride on every status frame rather than a change event.
    #[test]
    fn status_wire_carries_the_footer_segments() {
        let st = Status {
            think: crate::engine::ThinkMode::Medium,
            ..Status::default()
        };
        let wire = StatusWire::from(&st);
        assert_eq!(wire.think, "med");
        assert_eq!(wire.origin, crate::status::engine_origin_label());
        assert_eq!(wire.cwd, crate::status::cwd_label());
        assert_eq!(
            wire.branch,
            crate::status::git_branch_label().unwrap_or_default()
        );
    }

    #[test]
    fn from_event_none_for_sub_agent_variants() {
        assert!(
            ServerMsg::from_event(&UiEvent::SubStart {
                label: "agent".into(),
                task: "t".into(),
            })
            .is_none()
        );
        assert!(ServerMsg::from_event(&UiEvent::SubEnd).is_none());
        assert!(
            ServerMsg::from_event(&UiEvent::Sub(Box::new(UiEvent::Visible("x".into())))).is_none()
        );
        assert!(ServerMsg::from_event(&UiEvent::Visible("v".into())).is_some());
    }

    #[test]
    fn frame_roundtrip_all_events() {
        let events = [
            UiEvent::Visible("v".into()),
            UiEvent::Think("t".into()),
            UiEvent::Tool("tool".into()),
            UiEvent::Error("e".into()),
            UiEvent::Dim("d".into()),
            UiEvent::Plain("p".into()),
            UiEvent::CommandDone {
                text: "/usage".into(),
            },
            UiEvent::Controls(serde_json::json!({"version":1,"commands":[]})),
            UiEvent::Tasks(crate::tui::TaskView::default()),
            UiEvent::Jobs(JobsWire::capture(&[])),
            UiEvent::Question(serde_json::json!({"id":"q1","options":[]})),
            UiEvent::QuestionDone { id: "q1".into() },
            UiEvent::Editor(serde_json::json!({"id":"e1","title":"File","text":"content"})),
            UiEvent::EditorDone { id: "e1".into() },
            UiEvent::UserEcho("u".into()),
            UiEvent::EndLine,
            UiEvent::BtwBegin,
            UiEvent::BtwEnd,
            UiEvent::MainCheckpoint,
            UiEvent::MainRollback,
            UiEvent::Status(Status {
                state: WorkerState::Generating,
                generated: 7,
                ctx_used: 10,
                ctx_size: 100,
                ..Status::default()
            }),
        ];
        for (i, ev) in events.iter().enumerate() {
            let frame = ServerFrame::seq(i as u64, ServerMsg::from_event(ev).unwrap());
            let json = frame.to_json().unwrap();
            let back: ServerFrame = serde_json::from_str(&json).unwrap();
            assert_eq!(frame, back, "roundtrip for {ev:?}");
            assert_eq!(back.v, PROTOCOL_VERSION);
            assert!(json.contains("\"v\":1"));
        }
    }

    #[test]
    fn observer_auth_never_claims_or_reclaims_the_controller_slot() {
        let frame = ClientFrame::new(ClientMsg::ObserveAuth {
            token: "fixture".into(),
        });
        let json = frame.to_json().unwrap();
        assert!(matches!(
            ClientFrame::from_json(&json).unwrap().msg,
            ClientMsg::ObserveAuth { .. }
        ));
        let mut free = ControlPolicy::new(false, true);
        assert!(!initial_control(&mut free, 9, None, true));
        assert_eq!(free.holder(), Holder::Free);
        let mut local = ControlPolicy::new(true, true);
        assert!(!initial_control(&mut local, 9, None, true));
        assert_eq!(local.holder(), Holder::Local);
        assert_eq!(free.request(7), RequestOutcome::Granted);
        free.disconnect(7);
        assert!(!initial_control(&mut free, 9, Some(10), true));
        assert_eq!(free.holder(), Holder::Remote(7));
        assert!(free.grace.is_some());
        assert!(initial_control(&mut free, 9, Some(10), false));
        assert_eq!(free.holder(), Holder::Remote(9));
    }

    #[test]
    fn host_snapshot_tracks_grants_and_expired_reservations_without_credentials() {
        let state = RemoteState {
            endpoint: "ws://127.0.0.1:9000/".into(),
            bus: Arc::new(BroadcastBus::new()),
            shared: Arc::new(TurnShared::default()),
            control: Mutex::new(ControlPolicy::new(true, false)),
            token: "never-publish-this-secret".into(),
            allowed_origins: Vec::new(),
            queue_max: 1024,
            session_ids: AtomicU64::new(0),
            shutdown: AtomicBool::new(false),
        };
        let local = state.host_snapshot().unwrap();
        assert_eq!(local.holder, "local");
        assert!(local.controller.is_none());
        for id in 1..=100 {
            assert_eq!(
                state.control.lock().unwrap().request(id),
                RequestOutcome::NeedsLocalGrant
            );
        }
        let waiting = state.host_snapshot().unwrap();
        assert_eq!(waiting.pending_total, 100);
        assert_eq!(waiting.pending, (1..=64).collect::<Vec<_>>());
        assert_eq!(state.control.lock().unwrap().grant_next(), Some(1));
        let granted = state.host_snapshot().unwrap();
        assert_eq!(granted.controller, Some(1));
        assert_eq!(granted.pending_total, 0);
        assert!(
            granted.pending.is_empty(),
            "granting one client declines the other waiting requests"
        );
        state.control.lock().unwrap().disconnect(1);
        assert!(state.host_snapshot().unwrap().reserved);
        state.control.lock().unwrap().grace = Some((
            1,
            std::time::Instant::now()
                .checked_sub(Duration::from_secs(1))
                .unwrap(),
        ));
        let expired = state.host_snapshot().unwrap();
        assert_eq!(expired.holder, "local");
        assert!(expired.controller.is_none());
        assert!(!expired.reserved);
        state.bus.broadcast(UiEvent::Visible("Keep answer".into()));
        for _ in 0..100 {
            state.bus.broadcast(UiEvent::Host(waiting.clone()));
        }
        state.bus.broadcast(UiEvent::Host(granted.clone()));
        let (tail, _) = state.bus.scrollback_since(None);
        assert_eq!(tail.len(), 2);
        assert!(
            matches!(&tail[1].event, UiEvent::Host(snapshot) if snapshot.controller == Some(1))
        );
        let frame = ServerMsg::from_event(&UiEvent::Host(granted)).unwrap();
        let json = serde_json::to_string(&frame).unwrap();
        assert!(!json.contains(&state.token));
        assert!(json.contains("\"type\":\"host\""));
        assert_eq!(HostWire::off().holder, "off");
    }

    #[test]
    fn remote_questions_allow_only_the_current_controller() {
        let state = RemoteState {
            endpoint: "ws://127.0.0.1:9000/".into(),
            bus: Arc::new(BroadcastBus::new()),
            shared: Arc::new(TurnShared::default()),
            control: Mutex::new(ControlPolicy::new(false, false)),
            token: "test".into(),
            allowed_origins: Vec::new(),
            queue_max: 1024,
            session_ids: AtomicU64::new(0),
            shutdown: AtomicBool::new(false),
        };
        assert_eq!(
            state.control.lock().unwrap().request(7),
            RequestOutcome::Granted
        );
        state.shared.questions.open(
            "rq1",
            crate::tools::ask::AskRequest {
                question: "Continue?".into(),
                header: "Confirm".into(),
                multi: false,
                allow_chat: false,
                options: vec![crate::tools::ask::AskOption {
                    label: "Yes".into(),
                    description: String::new(),
                }],
            },
        );
        assert!(!accept_question_answer(&state, 8, "rq1", "1").unwrap());
        assert!(!accept_question_answer(&state, 7, "old", "1").unwrap());
        assert!(accept_question_answer(&state, 7, "rq1", "1").unwrap());
        assert_eq!(
            state.shared.questions.take_answer("rq1"),
            Some(crate::tools::ask::AskOutcome::Answered(vec!["Yes".into()]))
        );
    }

    #[test]
    fn client_frame_roundtrip() {
        let msgs = [
            ClientMsg::Auth {
                token: "abc".into(),
                resume_from: Some(5),
            },
            ClientMsg::Prompt { text: "hi".into() },
            ClientMsg::Btw { text: "q".into() },
            ClientMsg::Interrupt,
            ClientMsg::Edit {
                document_id: "e1".into(),
                text: serde_json::Value::Null,
            },
            ClientMsg::Edit {
                document_id: "e1".into(),
                text: serde_json::json!("edited\n🦀"),
            },
            ClientMsg::Answer {
                question_id: "q1".into(),
                answer: "1".into(),
            },
            ClientMsg::Command {
                text: "/help".into(),
            },
            ClientMsg::RequestControl,
            ClientMsg::ReleaseControl,
            ClientMsg::Ping,
        ];
        for msg in msgs {
            let frame = ClientFrame::new(msg);
            let json = frame.to_json().unwrap();
            assert_eq!(ClientFrame::from_json(&json).unwrap(), frame);
        }
    }

    #[test]
    fn auth_defaults_resume_from_absent() {
        let f = ClientFrame::from_json(r#"{"v":1,"type":"auth","token":"x"}"#).unwrap();
        assert_eq!(
            f.msg,
            ClientMsg::Auth {
                token: "x".into(),
                resume_from: None
            }
        );
    }

    #[test]
    fn remote_editors_allow_only_the_controller_and_require_explicit_content() {
        let state = RemoteState {
            endpoint: "ws://127.0.0.1:9000/".into(),
            bus: Arc::new(BroadcastBus::new()),
            shared: Arc::new(TurnShared::default()),
            control: Mutex::new(ControlPolicy::new(false, false)),
            token: "test".into(),
            allowed_origins: Vec::new(),
            queue_max: 1024,
            session_ids: AtomicU64::new(0),
            shutdown: AtomicBool::new(false),
        };
        assert_eq!(
            state.control.lock().unwrap().request(7),
            RequestOutcome::Granted
        );
        state.shared.editors.open("re1");
        assert!(
            !accept_editor_reply(&state, 8, "re1", &serde_json::json!("unauthorized")).unwrap()
        );
        assert!(!accept_editor_reply(&state, 7, "old", &serde_json::json!("stale")).unwrap());
        assert!(accept_editor_reply(&state, 7, "re1", &serde_json::json!("saved")).unwrap());
        assert_eq!(
            state.shared.editors.take_reply("re1"),
            Some(crate::remote::editors::EditReply::Save("saved".into()))
        );
        assert!(ClientFrame::from_json(r#"{"v":1,"type":"edit","document_id":"re1"}"#).is_err());
        assert!(
            ClientFrame::from_json(r#"{"v":1,"type":"edit","document_id":"re1","text":null}"#)
                .is_ok()
        );
    }

    // --- auth primitives ---

    #[test]
    fn constant_time_eq_semantics() {
        assert!(constant_time_eq(b"secret", b"secret"));
        assert!(!constant_time_eq(b"secret", b"secreT"));
        assert!(!constant_time_eq(b"secret", b"secret2"));
        assert!(constant_time_eq(b"", b""));
    }

    #[test]
    fn generated_tokens_are_unique_and_urlsafe() {
        let a = generate_token();
        let b = generate_token();
        assert_ne!(a, b);
        assert_eq!(a.len(), 43); // 32 bytes base64url unpadded
        assert!(
            a.bytes()
                .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_')
        );
    }

    #[test]
    fn base64url_known_vector() {
        // "foobar" → Zm9vYmFy in standard/url base64.
        assert_eq!(base64url(b"foobar"), "Zm9vYmFy");
        assert_eq!(base64url(b"fo"), "Zm8");
    }

    // --- control policy ---

    #[test]
    fn headless_first_requester_becomes_controller() {
        let mut p = ControlPolicy::new(false, false);
        assert_eq!(p.holder(), Holder::Free);
        assert_eq!(p.request(1), RequestOutcome::Granted);
        assert!(p.remote_can_control(1));
        // A second client is denied while the first holds control.
        assert_eq!(
            p.request(2),
            RequestOutcome::Denied("another client holds control".to_owned())
        );
        assert!(!p.remote_can_control(2));
        // Release frees it for the next requester.
        p.release(1);
        assert_eq!(p.holder(), Holder::Free);
        assert_eq!(p.request(2), RequestOutcome::Granted);
    }

    /// The request records who asked, which is the whole point: the local
    /// `/grant` has nothing to act on otherwise.
    #[test]
    fn a_request_needing_a_grant_becomes_pending() {
        let mut p = ControlPolicy::new(true, false);
        assert!(p.pending().is_empty());
        assert_eq!(p.request(7), RequestOutcome::NeedsLocalGrant);
        assert_eq!(p.pending(), &[7]);
        // Re-asking is idempotent rather than a queue of duplicates.
        assert_eq!(p.request(7), RequestOutcome::NeedsLocalGrant);
        assert_eq!(p.pending(), &[7]);
    }

    #[test]
    fn grant_next_answers_the_oldest_request_and_clears_the_rest() {
        let mut p = ControlPolicy::new(true, false);
        p.request(1);
        p.request(2);
        assert_eq!(p.pending(), &[1, 2]);
        assert_eq!(p.grant_next(), Some(1));
        assert!(p.remote_can_control(1));
        // One controller, so granting 1 answers 2 with "no" rather than leaving
        // it queued to fire the moment control comes back.
        assert!(p.pending().is_empty());
        assert!(!p.remote_can_control(2));
    }

    #[test]
    fn grant_next_with_nothing_waiting_is_a_no_op() {
        let mut p = ControlPolicy::new(true, false);
        assert_eq!(p.grant_next(), None);
        assert_eq!(p.holder(), Holder::Local);
    }

    #[test]
    fn a_specific_grant_picks_one_waiter_out() {
        let mut p = ControlPolicy::new(true, false);
        p.request(1);
        p.request(2);
        p.grant(2);
        assert!(p.remote_can_control(2));
        assert!(p.pending().is_empty());
    }

    #[test]
    fn a_waiter_that_hangs_up_stops_being_pending() {
        let mut p = ControlPolicy::new(true, false);
        p.request(1);
        p.request(2);
        // Not the controller, so this is a plain release plus a pending prune:
        // granting a dead socket would strand control on nobody.
        p.disconnect(1);
        assert_eq!(p.pending(), &[2]);
        assert_eq!(p.grant_next(), Some(2));
    }

    #[test]
    fn deny_drops_a_request_without_granting_it() {
        let mut p = ControlPolicy::new(true, false);
        p.request(1);
        assert!(p.deny(1));
        assert!(!p.deny(1)); // already gone
        assert_eq!(p.holder(), Holder::Local);
        assert_eq!(p.grant_next(), None);
    }

    /// With `allow_control` seeded, a request is granted outright, so nothing is
    /// ever pending and `/grant` stays empty-handed — the `/rc on` path.
    #[test]
    fn preauthorized_requests_never_become_pending() {
        let mut p = ControlPolicy::new(true, true);
        assert_eq!(p.request(1), RequestOutcome::Granted);
        assert!(p.pending().is_empty());
    }

    #[test]
    fn local_present_requires_grant() {
        let mut p = ControlPolicy::new(true, false);
        assert_eq!(p.holder(), Holder::Local);
        assert_eq!(p.request(1), RequestOutcome::NeedsLocalGrant);
        assert!(!p.remote_can_control(1));
        // Explicit local grant transfers control.
        p.grant(1);
        assert!(p.remote_can_control(1));
        // Releasing returns control to the local user.
        p.release(1);
        assert_eq!(p.holder(), Holder::Local);
    }

    /// `/rc` starts the server with `allow_control` set: the local operator
    /// typing the command *is* the consent, so a browser's `request_control`
    /// must succeed even though the local front-end seeded `Holder::Local`.
    #[test]
    fn allow_control_grants_over_a_present_local() {
        let mut p = ControlPolicy::new(true, true);
        assert_eq!(p.holder(), Holder::Local);
        assert_eq!(p.request(1), RequestOutcome::Granted);
        assert_eq!(p.holder(), Holder::Remote(1));
        assert!(p.remote_can_control(1));
        // Releasing hands the slot back to the local user, not to the next remote.
        p.release(1);
        assert_eq!(p.holder(), Holder::Local);
        assert!(!p.remote_can_control(1));
    }

    #[test]
    fn allow_control_lets_remote_take_from_local() {
        let mut p = ControlPolicy::new(true, true);
        assert_eq!(p.request(1), RequestOutcome::Granted);
        assert!(p.remote_can_control(1));
    }

    #[test]
    fn disconnect_holds_slot_and_reconnect_reclaims() {
        let mut p = ControlPolicy::new(false, false);
        assert_eq!(p.request(1), RequestOutcome::Granted);
        // A drop keeps the slot reserved (grace window): another client cannot
        // grab it, but the reconnecting session can reclaim it.
        p.disconnect(1);
        assert_eq!(
            p.request(2),
            RequestOutcome::Denied("another client holds control".to_owned())
        );
        assert!(p.reclaim(3), "reconnect reclaims within grace");
        assert!(p.remote_can_control(3));
        assert!(!p.remote_can_control(2));
    }

    #[test]
    fn expired_grace_frees_the_slot() {
        let mut p = ControlPolicy::new(false, false);
        assert_eq!(p.request(1), RequestOutcome::Granted);
        p.disconnect(1);
        // Force the grace deadline into the past, then a new request succeeds
        // and a stale reclaim fails.
        p.grace = Some((
            1,
            std::time::Instant::now()
                .checked_sub(Duration::from_secs(1))
                .unwrap(),
        ));
        assert_eq!(p.request(2), RequestOutcome::Granted);
        assert!(!p.reclaim(3));
        assert!(p.remote_can_control(2));
    }

    #[test]
    fn non_controller_disconnect_just_releases() {
        let mut p = ControlPolicy::new(true, false);
        p.grant(1);
        // A session that is not the holder disconnecting must not reserve a slot.
        p.disconnect(2);
        assert!(p.remote_can_control(1));
        p.disconnect(1);
        // Holder drop reserves; local user reclaims after expiry.
        p.grace = Some((
            1,
            std::time::Instant::now()
                .checked_sub(Duration::from_secs(1))
                .unwrap(),
        ));
        p.request(2); // triggers expire_grace
        assert_eq!(p.holder(), Holder::Local);
    }

    // --- integration: a real loopback server + tungstenite client ---

    fn test_server(local_present: bool, allow_control: bool) -> RemoteServer {
        test_server_cfg(ServerConfig {
            token: "tok".to_owned(),
            local_present,
            allow_control,
            allowed_origins: Vec::new(),
            queue_max: 1 << 20,
        })
    }

    fn test_server_cfg(cfg: ServerConfig) -> RemoteServer {
        RemoteServer::start(
            "127.0.0.1:0",
            cfg,
            Arc::new(BroadcastBus::new()),
            Arc::new(TurnShared::default()),
        )
        .expect("server binds")
    }

    fn connect(addr: std::net::SocketAddr) -> tungstenite::WebSocket<std::net::TcpStream> {
        let url = format!("ws://{addr}/");
        let stream = TcpStream::connect(addr).unwrap();
        let (ws, _resp) =
            tungstenite::client(url.parse::<tungstenite::http::Uri>().unwrap(), stream)
                .expect("ws handshake");
        ws
    }

    fn send_client(ws: &mut tungstenite::WebSocket<std::net::TcpStream>, msg: ClientMsg) {
        ws.send(Message::Text(
            ClientFrame::new(msg).to_json().unwrap().into(),
        ))
        .unwrap();
        ws.flush().unwrap();
    }

    fn read_server(ws: &mut tungstenite::WebSocket<std::net::TcpStream>) -> Option<ServerFrame> {
        loop {
            match ws.read() {
                Ok(Message::Text(t)) => return Some(serde_json::from_str(&t).unwrap()),
                Ok(Message::Close(_)) | Err(_) => return None,
                Ok(_) => {}
            }
        }
    }

    #[test]
    fn auth_required_and_hello_snapshot_flow() {
        let server = test_server(false, false);
        let addr = server.local_addr;
        // Seed some scrollback so the snapshot is non-empty.
        server
            .state
            .bus
            .broadcast(UiEvent::Visible("earlier".into()));

        let mut ws = connect(addr);
        send_client(
            &mut ws,
            ClientMsg::Auth {
                token: "tok".into(),
                resume_from: None,
            },
        );
        let hello = read_server(&mut ws).expect("hello");
        match hello.msg {
            ServerMsg::Hello {
                controller,
                protocol_version,
                ..
            } => {
                assert_eq!(protocol_version, PROTOCOL_VERSION);
                assert!(controller, "headless first client auto-controls");
            }
            other => panic!("expected hello, got {other:?}"),
        }
        let snap = read_server(&mut ws).expect("snapshot");
        match snap.msg {
            ServerMsg::Snapshot { scrollback, .. } => {
                assert_eq!(scrollback.len(), 1);
                assert_eq!(
                    scrollback[0].msg,
                    ServerMsg::Visible {
                        text: "earlier".into()
                    }
                );
            }
            other => panic!("expected snapshot, got {other:?}"),
        }
        // A live event after connect is mirrored.
        server.state.bus.broadcast(UiEvent::Visible("live".into()));
        let live = read_server(&mut ws).expect("live frame");
        assert_eq!(
            live.msg,
            ServerMsg::Visible {
                text: "live".into()
            }
        );
    }

    /// `/clear` must reach the page two ways: the attached client is told to
    /// clear, and the scrollback a *later* client would be replayed no longer
    /// contains the cleared transcript. Before this, a `/clear` was invisible
    /// remotely — the page kept the old transcript and a fresh attach replayed
    /// it from the bus.
    #[test]
    fn session_reset_reaches_attached_and_later_clients() {
        let server = test_server(false, false);
        server
            .state
            .bus
            .broadcast(UiEvent::Visible("from the old session".into()));

        let mut ws = connect(server.local_addr);
        send_client(
            &mut ws,
            ClientMsg::Auth {
                token: "tok".into(),
                resume_from: None,
            },
        );
        read_server(&mut ws).expect("hello");
        read_server(&mut ws).expect("snapshot");

        server.state.bus.broadcast(UiEvent::SessionReset);
        let frame = read_server(&mut ws).expect("reset frame");
        assert_eq!(frame.msg, ServerMsg::Reset, "attached client is told");

        // A client attaching now must not be replayed the cleared transcript.
        let mut late = connect(server.local_addr);
        send_client(
            &mut late,
            ClientMsg::Auth {
                token: "tok".into(),
                resume_from: None,
            },
        );
        read_server(&mut late).expect("hello");
        let snap = read_server(&mut late).expect("snapshot");
        match snap.msg {
            ServerMsg::Snapshot { scrollback, .. } => {
                assert!(
                    !scrollback.iter().any(|e| matches!(
                        &e.msg,
                        ServerMsg::Visible { text } if text == "from the old session"
                    )),
                    "cleared transcript replayed to a late joiner: {scrollback:?}"
                );
            }
            other => panic!("expected snapshot, got {other:?}"),
        }
    }

    /// End of turn reaches an attached client as a `notify` frame. The local
    /// desktop notification is invisible to whoever is driving from a browser,
    /// so this is the only signal they get — worth pinning end to end rather
    /// than only at the `from_event` mapping.
    #[test]
    fn end_of_turn_notification_reaches_an_attached_client() {
        let server = test_server(false, false);
        let mut ws = connect(server.local_addr);
        send_client(
            &mut ws,
            ClientMsg::Auth {
                token: "tok".into(),
                resume_from: None,
            },
        );
        read_server(&mut ws).expect("hello");
        read_server(&mut ws).expect("snapshot");

        server.state.bus.broadcast(UiEvent::Notify {
            title: "finished: add the button".into(),
            body: "…and wired it up.".into(),
        });
        let frame = read_server(&mut ws).expect("notify frame");
        assert_eq!(
            frame.msg,
            ServerMsg::Notify {
                title: "finished: add the button".into(),
                body: "…and wired it up.".into(),
            }
        );
    }

    #[test]
    fn auth_rejects_bad_token() {
        let server = test_server(false, false);
        let mut ws = connect(server.local_addr);
        send_client(
            &mut ws,
            ClientMsg::Auth {
                token: "wrong".into(),
                resume_from: None,
            },
        );
        // Server closes the connection without a hello.
        assert!(read_server(&mut ws).is_none());
    }

    #[test]
    fn remote_prompt_queues_and_interrupt_sets_flag() {
        let server = test_server(false, false);
        let mut ws = connect(server.local_addr);
        send_client(
            &mut ws,
            ClientMsg::Auth {
                token: "tok".into(),
                resume_from: None,
            },
        );
        let _ = read_server(&mut ws); // hello
        let _ = read_server(&mut ws); // snapshot

        send_client(
            &mut ws,
            ClientMsg::Prompt {
                text: "do it".into(),
            },
        );
        send_client(&mut ws, ClientMsg::Ping);
        // Wait for pong to guarantee the prompt was processed first.
        assert_eq!(read_server(&mut ws).map(|f| f.msg), Some(ServerMsg::Pong));
        assert_eq!(
            server
                .state
                .shared
                .take_prompt_requests()
                .into_iter()
                .map(|(_, text)| text)
                .collect::<Vec<_>>(),
            vec!["do it"]
        );

        send_client(&mut ws, ClientMsg::Interrupt);
        send_client(&mut ws, ClientMsg::Ping);
        assert_eq!(read_server(&mut ws).map(|f| f.msg), Some(ServerMsg::Pong));
        assert!(server.state.shared.interrupt.load(Ordering::Relaxed));
    }

    #[test]
    fn non_controller_prompt_is_denied_but_btw_allowed() {
        let server = test_server(false, false);
        let addr = server.local_addr;
        // First client grabs control.
        let mut c1 = connect(addr);
        send_client(
            &mut c1,
            ClientMsg::Auth {
                token: "tok".into(),
                resume_from: None,
            },
        );
        let _ = read_server(&mut c1);
        let _ = read_server(&mut c1);

        // Second client is a mirror.
        let mut c2 = connect(addr);
        send_client(
            &mut c2,
            ClientMsg::Auth {
                token: "tok".into(),
                resume_from: None,
            },
        );
        let _ = read_server(&mut c2);
        let _ = read_server(&mut c2);

        send_client(
            &mut c2,
            ClientMsg::Prompt {
                text: "nope".into(),
            },
        );
        let denied = read_server(&mut c2).expect("denied frame");
        assert!(matches!(denied.msg, ServerMsg::ControlDenied { .. }));

        // But a mirror's /btw is accepted.
        send_client(
            &mut c2,
            ClientMsg::Btw {
                text: "why?".into(),
            },
        );
        send_client(&mut c2, ClientMsg::Ping);
        assert_eq!(read_server(&mut c2).map(|f| f.msg), Some(ServerMsg::Pong));
        assert_eq!(server.state.shared.take_btw(), vec!["why?"]);
    }

    // --- Origin allow-list ---

    #[test]
    fn origin_policy_unit() {
        let allow = vec!["https://app.example.com".to_owned()];
        // Missing Origin (native clients): allowed.
        assert!(origin_allowed(None, &allow));
        // Loopback Origins: always allowed.
        assert!(origin_allowed(Some("http://localhost:31415"), &allow));
        assert!(origin_allowed(Some("http://127.0.0.1"), &allow));
        assert!(origin_allowed(Some("http://[::1]:8080"), &allow));
        // Allow-listed browser Origin (case-insensitive).
        assert!(origin_allowed(Some("https://app.example.com"), &allow));
        assert!(origin_allowed(Some("HTTPS://APP.EXAMPLE.COM"), &allow));
        // A non-allow-listed browser Origin is refused.
        assert!(!origin_allowed(Some("https://evil.example.com"), &allow));
        assert!(!origin_allowed(Some("null"), &allow));
        // With no allow-list, only missing / loopback pass.
        assert!(origin_allowed(Some("http://localhost"), &[]));
        assert!(!origin_allowed(Some("https://app.example.com"), &[]));
    }

    /// Builds a WS handshake request carrying an optional `Origin` and attempts
    /// the upgrade. Returns whether the handshake succeeded.
    fn try_ws_with_origin(addr: std::net::SocketAddr, origin: Option<&str>) -> bool {
        let stream = TcpStream::connect(addr).unwrap();
        let key = tungstenite::handshake::client::generate_key();
        let mut builder = tungstenite::http::Request::builder()
            .uri(format!("ws://{addr}/"))
            .header("Host", addr.to_string())
            .header("Connection", "Upgrade")
            .header("Upgrade", "websocket")
            .header("Sec-WebSocket-Version", "13")
            .header("Sec-WebSocket-Key", key);
        if let Some(o) = origin {
            builder = builder.header("Origin", o);
        }
        let req = builder.body(()).unwrap();
        tungstenite::client(req, stream).is_ok()
    }

    #[test]
    fn origin_allowlist_accepts_and_rejects_on_upgrade() {
        let server = test_server_cfg(ServerConfig {
            token: "tok".to_owned(),
            local_present: false,
            allow_control: false,
            allowed_origins: vec!["https://app.example.com".to_owned()],
            queue_max: 1 << 20,
        });
        let addr = server.local_addr;
        // No Origin (native client): accepted.
        assert!(try_ws_with_origin(addr, None));
        // Loopback Origin: accepted.
        assert!(try_ws_with_origin(addr, Some("http://127.0.0.1:31415")));
        // Allow-listed browser Origin: accepted.
        assert!(try_ws_with_origin(addr, Some("https://app.example.com")));
        // Non-allow-listed browser Origin: rejected (HTTP 403, no upgrade).
        assert!(!try_ws_with_origin(addr, Some("https://evil.example.com")));
    }

    // --- static web client ---

    /// Minimal blocking HTTP GET over a raw socket; returns (status line,
    /// headers-block, body).
    fn http_get(addr: std::net::SocketAddr, path: &str) -> (String, String, String) {
        use std::io::{Read, Write};
        let mut stream = TcpStream::connect(addr).unwrap();
        stream
            .write_all(
                format!("GET {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n")
                    .as_bytes(),
            )
            .unwrap();
        stream.flush().unwrap();
        let mut raw = String::new();
        stream.read_to_string(&mut raw).unwrap();
        let (head, body) = raw.split_once("\r\n\r\n").unwrap_or((&raw, ""));
        let (status, headers) = head.split_once("\r\n").unwrap_or((head, ""));
        (status.to_owned(), headers.to_owned(), body.to_owned())
    }

    /// Byte-oriented sibling of [`http_get`]: the logo body is PNG, so it
    /// cannot go through `read_to_string`.
    fn http_get_bytes(addr: std::net::SocketAddr, path: &str) -> (String, Vec<u8>) {
        use std::io::{Read, Write};
        let mut stream = TcpStream::connect(addr).unwrap();
        stream
            .write_all(
                format!("GET {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n")
                    .as_bytes(),
            )
            .unwrap();
        stream.flush().unwrap();
        let mut raw = Vec::new();
        stream.read_to_end(&mut raw).unwrap();
        let split = raw
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .expect("response has a header/body boundary");
        let head = String::from_utf8_lossy(&raw[..split]).into_owned();
        (head, raw[split + 4..].to_vec())
    }

    /// The web client's header image is served from the binary, so the page
    /// stays self-contained: no request ever leaves the loopback listener.
    #[test]
    fn serves_the_logo_as_png() {
        let server = test_server(false, false);
        let (head, body) = http_get_bytes(server.local_addr, "/logo.png");
        assert!(head.contains("200"), "head: {head}");
        assert!(
            head.to_ascii_lowercase()
                .contains("content-type: image/png"),
            "head: {head}"
        );
        assert_eq!(body.len(), WEB_CLIENT_LOGO.len(), "whole file served");
        assert_eq!(&body[..8], b"\x89PNG\r\n\x1a\n", "PNG magic intact");
    }

    #[test]
    fn serves_web_client_at_root() {
        let server = test_server(false, false);
        let addr = server.local_addr;
        let (status, headers, body) = http_get(addr, "/");
        assert!(status.contains("200"), "status: {status}");
        assert!(
            headers
                .to_ascii_lowercase()
                .contains("content-type: text/html"),
            "headers: {headers}"
        );
        assert!(body.contains("plank remote"), "body missing marker");
        // A bogus path 404s.
        let (status, _, _) = http_get(addr, "/nope");
        assert!(status.contains("404"), "status: {status}");
    }

    /// The `/remote-control` link carries the token in the query string; the
    /// router strips it, so the page is served as it is for a bare `/`.
    #[test]
    fn serves_web_client_when_the_path_carries_a_token() {
        let server = test_server(false, false);
        let addr = server.local_addr;
        let (status, _, body) = http_get(addr, "/?t=tok");
        assert!(status.contains("200"), "status: {status}");
        assert!(body.contains("plank remote"), "body missing marker");
    }

    // --- bounded outbound queue / slow-consumer eviction ---

    #[test]
    fn slow_client_is_evicted_while_others_keep_receiving() {
        // Small per-client cap so a stalled client trips it quickly.
        let server = test_server_cfg(ServerConfig {
            token: "tok".to_owned(),
            local_present: false,
            allow_control: false,
            allowed_origins: Vec::new(),
            queue_max: 64 * 1024,
        });
        let addr = server.local_addr;

        // Healthy client: drained continuously on a background thread.
        let mut healthy = connect(addr);
        send_client(
            &mut healthy,
            ClientMsg::Auth {
                token: "tok".into(),
                resume_from: None,
            },
        );
        let _ = read_server(&mut healthy); // hello
        let _ = read_server(&mut healthy); // snapshot
        let healthy_count = Arc::new(AtomicU64::new(0));
        let hc = Arc::clone(&healthy_count);
        let drain = std::thread::spawn(move || {
            while let Some(f) = read_server(&mut healthy) {
                if matches!(f.msg, ServerMsg::Visible { .. }) {
                    hc.fetch_add(1, Ordering::Relaxed);
                }
                if matches!(f.msg, ServerMsg::Bye { .. }) {
                    break;
                }
            }
        });

        // Slow client: authenticates, then stops reading entirely.
        let mut slow = connect(addr);
        send_client(
            &mut slow,
            ClientMsg::Auth {
                token: "tok".into(),
                resume_from: None,
            },
        );
        // Do NOT read hello/snapshot: its socket buffer fills and the server's
        // bounded outbound queue for this client overflows.

        // Wait until both connections are registered as subscribers.
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while server.state.bus.subscriber_count() < 2 {
            assert!(
                std::time::Instant::now() < deadline,
                "both never subscribed"
            );
            std::thread::sleep(Duration::from_millis(10));
        }

        // Flood enough data to overflow OS buffers plus the 64 KiB cap.
        let big = "x".repeat(2048);
        for _ in 0..2000 {
            server.state.bus.broadcast(UiEvent::Visible(big.clone()));
        }

        // The slow client is evicted: subscriber count drops back to just the
        // healthy one (the bus prunes the dropped subscriber on broadcast).
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        loop {
            // Keep broadcasting so the prune (which happens on send) fires.
            server.state.bus.broadcast(UiEvent::Visible("tick".into()));
            if server.state.bus.subscriber_count() <= 1 {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "slow client was not evicted"
            );
            std::thread::sleep(Duration::from_millis(20));
        }

        // The healthy client kept receiving throughout.
        assert!(
            healthy_count.load(Ordering::Relaxed) > 0,
            "healthy client received nothing"
        );

        drop(server); // sends Bye, unblocks the drain thread
        let _ = drain.join();
    }
}
