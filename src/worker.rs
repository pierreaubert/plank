// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Worker/UI split for the TUI (issue #12), mirroring the C's "Model Worker
//! Thread" and "Worker/UI Synchronization Helpers" sections.
//!
//! During a turn the engine, session, and tool dispatch run on a scoped
//! worker thread while the UI thread keeps its real event loop: the next
//! prompt stays editable (and is queued for the worker to drain between tool
//! rounds, like the C's `queued_user_drain`), scrolling and interrupts work
//! directly, and redraws happen at the UI's own cadence instead of inside an
//! engine callback.
//!
//! The channel payload is [`UiEvent`]: the [`crate::viz::RenderSink`] calls
//! made by the worker's stream renderer, plus log lines and status snapshots.
//! [`RenderSink`] is the natural serialization boundary — the UI replays the
//! same calls against the real [`crate::tui::OutputLog`].

use std::sync::atomic::AtomicBool;
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};

use crate::status::Status;
use crate::tui::OutputLog;
use crate::viz::RenderSink;

/// A shared editor original excluded from diagnostic text.
#[derive(Clone)]
pub struct EditorRequest(Arc<serde_json::Value>);

impl From<serde_json::Value> for EditorRequest {
    fn from(value: serde_json::Value) -> Self {
        Self(Arc::new(value))
    }
}

impl AsRef<serde_json::Value> for EditorRequest {
    fn as_ref(&self) -> &serde_json::Value {
        &self.0
    }
}

impl std::fmt::Debug for EditorRequest {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("EditorRequest")
            .field("bytes", &self.0["text"].as_str().map(str::len))
            .finish_non_exhaustive()
    }
}

/// One worker→UI message during a turn.
#[derive(Debug, Clone)]
pub enum UiEvent {
    /// Rendered assistant text ([`RenderSink::visible_text`]).
    Visible(String),
    /// Thinking text ([`RenderSink::think_text`]).
    Think(String),
    /// Tool banner text ([`RenderSink::tool_text`]).
    Tool(String),
    /// Stream error text ([`RenderSink::error_text`]).
    Error(String),
    /// A dim log line (tool observations, notices, hook warnings).
    Dim(String),
    /// The transient live `… N lines` line of a collapsed `write` preview.
    /// `Some` sets/replaces it in place; `None` removes it (the permanent
    /// summary then arrives as ordinary [`Self::Think`] text). See
    /// [`crate::viz::RenderSink::preview_status`].
    PreviewStatus(Option<String>),
    /// An agent system status line (`agent_publish_system_status`): carries the
    /// bare message, styled by the UI with [`crate::status::system_line`].
    SystemStatus(String),
    /// A git-style diff card for a file an `edit`/`write` call changed.
    EditCard(crate::tools::diff::EditPreview),
    /// A complete markdown document a tool wants shown in the transcript —
    /// the plan `ExitPlanMode` puts up for approval. Rendered like assistant
    /// text rather than as a status line: it is a document, not a notice.
    Markdown(String),
    /// A plain log line.
    Plain(String),
    /// A user-echo line (queued prompts, `/btw` questions).
    UserEcho(String),
    /// A queued prompt joined the conversation: the UI moves the oldest
    /// indented prompt out of the log's pending region and into the scrollback
    /// as a committed user echo.
    ///
    /// Payload-free on purpose. The queue is FIFO on both sides and the text
    /// already sits in the pending region, so sending it along would create a
    /// second copy free to disagree with the first.
    ///
    /// "FIFO on both sides" is not quite literal: the remote control path
    /// (`src/remote/control.rs`) pushes `Prompt`/`Command` frames straight
    /// into `shared.queued` without a matching `push_pending`, so a remote
    /// push can make `QueuedJoined` events outnumber `pending` entries. That
    /// can transiently commit the wrong row — a local prompt typed between a
    /// remote drain and this event's delivery gets committed in the remote
    /// prompt's place — but it self-corrects on the local prompt's own drain
    /// (the second commit is a no-op), so the end state is always right and
    /// only the ordering is briefly wrong. Tolerated rather than fixed: making
    /// the remote path push a pending row is a behaviour change beyond this
    /// feature's scope.
    QueuedJoined,
    /// The transcript was replaced — `/clear`, `/new`, `/switch`, `/resume`.
    /// Everything above it belongs to a session that no longer exists, so a
    /// remote front-end clears its log, and [`BroadcastBus::broadcast`] drops
    /// the scrollback rather than replaying a cleared transcript to whoever
    /// attaches next. The local UI clears its own log at the call site.
    SessionReset,
    /// The turn finished: the same headline and body the local desktop
    /// notification carries, so an attached remote front-end can raise its own
    /// (a browser notification, a bell, a title flash) instead of the operator
    /// having to watch the page. Carries no on-screen text — the local UI
    /// already showed the turn — so the TUI ignores it.
    Notify {
        /// Notification headline (the prompt, or that the turn was interrupted).
        title: String,
        /// Notification body (the tail of the answer).
        body: String,
        /// Optional desktop policy for rebuilt graphical clients.
        mode: Option<String>,
    },
    /// A remote slash command returned to the idle dispatcher.
    CommandDone {
        /// The submitted command, including its arguments.
        text: String,
    },
    /// A stamped remote prompt returned from its own turn.
    PromptDone { request_id: String },
    /// An identified remote slash command returned from its dispatcher.
    CommandRequestDone { request_id: String },
    /// Runtime command catalog for remote front ends.
    Controls(serde_json::Value),
    /// An interactive question available to the remote controller.
    Question(serde_json::Value),
    /// The question closed after local input, remote input, or interruption.
    QuestionDone {
        /// Identity of the closed question.
        id: String,
    },
    /// A document awaiting an explicit remote Save or Cancel.
    Editor(EditorRequest),
    /// The pending document editor closed.
    EditorDone {
        /// Identity of the closed editor.
        id: String,
    },
    /// Terminates the in-progress rendered line.
    EndLine,
    /// Worker progress snapshot for the status footer.
    Status(Status),
    /// Task list snapshot for the status-bar counter and the contextual strip
    /// (issue #35), sent whenever a `task` tool call changes the list.
    Tasks(crate::tui::TaskView),
    /// Background job snapshot for graphical and remote monitors.
    Jobs(crate::remote::control::JobsWire),
    /// Requested page of engine-host saved conversations.
    Sessions(crate::remote::control::SessionsWire),
    /// Requested page of engine settings.
    Config(crate::remote::control::ConfigWire),
    /// Requested checkpoint or fork-point metadata.
    History(crate::remote::control::HistoryWire),
    /// Transient complete-file export transfer, never retained in scrollback.
    Artifact(crate::export::transfer::Transfer),
    /// Requested inventory of running extensions.
    Extensions(crate::remote::control::ExtensionsWire),
    /// Requested page of KV cache lineage.
    Cache(crate::remote::control::CacheWire),
    /// Detached model-update helper status.
    Downloads(crate::remote::control::DownloadsWire),
    /// Requested measurements, independently retained by kind.
    Metrics(crate::remote::control::MetricsWire),
    /// Live remote-host policy and pending grants, without credentials.
    Host(crate::remote::control::HostWire),
    /// A `/btw` side answer is starting: the UI opens the side panel (if not
    /// already visible) and routes subsequent render events into it.
    BtwBegin,
    /// The `/btw` answer finished (or was cancelled): the UI stops routing to
    /// the panel and resumes rendering into the main log, but leaves the panel
    /// on screen (frozen) so the answer stays readable while the main task
    /// continues. The panel is dismissed by the user with Esc, not here.
    BtwEnd,
    /// Marks the start of a main generation pass: the UI snapshots the main
    /// log length so a later [`UiEvent::MainRollback`] can discard a
    /// preempted pass's partial output.
    MainCheckpoint,
    /// A main pass was preempted by a priority `/btw`: the UI truncates the
    /// main log back to the last [`UiEvent::MainCheckpoint`] so the re-run
    /// does not duplicate the discarded partial output.
    MainRollback,
    /// A sub-agent run began: the UI opens a roster row for it and starts
    /// filling its buffer.
    SubStart {
        /// Display label — the configured agent name, or a generic one for an
        /// unnamed sub-agent.
        label: String,
        /// What the agent was asked to do, as the roster row reports it beside
        /// the name. The delegated task, not the agent's output: a row summarises
        /// the job, and a tail of streamed text lands mid-token on whatever the
        /// model happens to be writing (`vals =`).
        task: String,
    },
    /// The sub-agent run finished, successfully or not. The buffer is kept so
    /// the user can still read it, and its roster row freezes its elapsed time.
    SubEnd,
    /// One *completed* pass's token spend for a running sub-agent, so its roster
    /// row can report what it is costing. Emitted per pass rather than once at
    /// the end: a long run's row would otherwise read as free right up to the
    /// moment it finishes. While a pass is still in flight the row tracks the
    /// worker's [`UiEvent::Status`] snapshots instead — this arrives only once
    /// the pass is done, so the two never double-count.
    ///
    /// `label` names the row when the emitter knows it — a fan-out has several
    /// rows open at once and folds their passes in together, so "the current
    /// run" would credit them all to whichever started last. `None` means the
    /// current run, which is what the serial path has.
    SubTokens {
        label: Option<String>,
        /// Tokens the pass ingested (prompt / prefill).
        prefill: u64,
        /// Tokens the pass generated.
        generated: u64,
    },
    /// One render payload destined for the sub-agent buffer rather than the
    /// main log. Boxed so the enum does not grow by a whole `UiEvent`.
    Sub(Box<UiEvent>),
    /// One render payload destined for the `/btw` panel, addressed per event
    /// rather than by the [`BtwBegin`](Self::BtwBegin)/[`BtwEnd`](Self::BtwEnd)
    /// mode.
    ///
    /// The mode cannot serve a *multiplexed* aside: the main task and the aside
    /// interleave, so their events arrive mixed and a mode flag would misroute
    /// whichever stream it is not currently set to. Boxed for the same reason
    /// as [`Sub`](Self::Sub).
    Btw(Box<UiEvent>),
}

impl UiEvent {
    /// Whether this event belongs to the local pane only and must not
    /// go onto the [`BroadcastBus`]. A remote-side pane is out of scope, so
    /// `ServerMsg::from_event` maps these to `None` — broadcasting them
    /// anyway would burn scrollback ring slots (one `Sub` per streamed chunk)
    /// and evict the real transcript a reconnecting client replays. The remote
    /// protocol does not require contiguous sequence ids, so simply not
    /// broadcasting is safe.
    #[must_use]
    pub fn is_local_pane_only(&self) -> bool {
        matches!(
            self,
            Self::SubStart { .. }
                | Self::SubEnd
                | Self::SubTokens { .. }
                | Self::Sub(_)
                | Self::QueuedJoined
        )
    }
}

/// [`RenderSink`] forwarding render calls over the worker→UI channel.
///
/// A receiver that hangs up mid-turn just drops the text: the worker keeps
/// running and the transcript (owned by the worker) stays authoritative.
#[derive(Debug)]
pub struct ChannelSink(pub Sender<UiEvent>);

impl RenderSink for ChannelSink {
    fn visible_text(&mut self, text: &str) {
        let _ = self.0.send(UiEvent::Visible(text.to_owned()));
    }
    fn think_text(&mut self, text: &str) {
        let _ = self.0.send(UiEvent::Think(text.to_owned()));
    }
    fn tool_text(&mut self, text: &str) {
        let _ = self.0.send(UiEvent::Tool(text.to_owned()));
    }
    fn error_text(&mut self, text: &str) {
        let _ = self.0.send(UiEvent::Error(text.to_owned()));
    }
    fn preview_status(&mut self, text: Option<&str>) {
        let _ = self.0.send(UiEvent::PreviewStatus(text.map(str::to_owned)));
    }
}

/// [`RenderSink`] forwarding a sub-agent's render calls over the worker→UI
/// channel, wrapped in [`UiEvent::Sub`] so the UI routes them to the sub-agent
/// buffer instead of the main log. Same hang-up tolerance as [`ChannelSink`].
#[derive(Debug)]
pub struct SubAgentSink(pub Sender<UiEvent>);

impl RenderSink for SubAgentSink {
    fn visible_text(&mut self, text: &str) {
        let _ = self
            .0
            .send(UiEvent::Sub(Box::new(UiEvent::Visible(text.to_owned()))));
    }
    fn think_text(&mut self, text: &str) {
        let _ = self
            .0
            .send(UiEvent::Sub(Box::new(UiEvent::Think(text.to_owned()))));
    }
    fn tool_text(&mut self, text: &str) {
        let _ = self
            .0
            .send(UiEvent::Sub(Box::new(UiEvent::Tool(text.to_owned()))));
    }
    fn error_text(&mut self, text: &str) {
        let _ = self
            .0
            .send(UiEvent::Sub(Box::new(UiEvent::Error(text.to_owned()))));
    }
}

/// [`RenderSink`] forwarding a `/btw` aside's render calls, wrapped in
/// [`UiEvent::Btw`] so they reach the side panel even while the main task is
/// streaming into the main log beside them. Same hang-up tolerance as
/// [`ChannelSink`].
///
/// A multiplexed aside needs its own renderer, not just its own sink: without
/// one its raw model output reaches the panel with the `<think>` tags still in
/// it.
#[derive(Debug)]
pub struct BtwSink(pub Sender<UiEvent>);

impl RenderSink for BtwSink {
    fn visible_text(&mut self, text: &str) {
        let _ = self
            .0
            .send(UiEvent::Btw(Box::new(UiEvent::Visible(text.to_owned()))));
    }
    fn think_text(&mut self, text: &str) {
        let _ = self
            .0
            .send(UiEvent::Btw(Box::new(UiEvent::Think(text.to_owned()))));
    }
    fn tool_text(&mut self, text: &str) {
        let _ = self
            .0
            .send(UiEvent::Btw(Box::new(UiEvent::Tool(text.to_owned()))));
    }
    fn error_text(&mut self, text: &str) {
        let _ = self
            .0
            .send(UiEvent::Btw(Box::new(UiEvent::Error(text.to_owned()))));
    }
}

/// A sequenced worker event: a monotonic `id` plus the [`UiEvent`]. The id lets
/// a reconnecting remote client resume with `resume_from` (see
/// `docs/REMOTE-CONTROL-DESIGN.md` §4.8) without re-seeing frames it already has.
#[derive(Debug, Clone)]
pub struct SeqEvent {
    /// Monotonic sequence id, unique and increasing per [`BroadcastBus`].
    pub id: u64,
    /// The event payload.
    pub event: UiEvent,
}

/// Default cap on the scrollback ring, in events. Bounds memory while giving a
/// late-joining remote client enough recent context to replay a `snapshot`.
pub const SCROLLBACK_CAP: usize = 4096;

/// Fan-out of [`UiEvent`]s to any number of consumers (the local TUI plus each
/// remote session), the single structural change remote control needs
/// (`docs/REMOTE-CONTROL-DESIGN.md` §4.2).
///
/// `broadcast` clones each event to every live subscriber and prunes hung-up
/// ones; a slow or vanished subscriber never blocks the worker, inheriting the
/// [`ChannelSink`] resilience contract. A bounded scrollback ring with
/// monotonic sequence ids backs late-join replay and `resume_from`.
#[derive(Debug, Default)]
pub struct BroadcastBus {
    inner: Mutex<BusInner>,
}

#[derive(Debug, Default)]
struct BusInner {
    subscribers: Vec<Subscriber>,
    scrollback: std::collections::VecDeque<SeqEvent>,
    next_id: u64,
}

#[derive(Debug)]
struct Subscriber {
    ordinary: Sender<SeqEvent>,
    exports: Option<std::sync::mpsc::SyncSender<SeqEvent>>,
    failed_export: Arc<Mutex<Option<String>>>,
}

/// An export-capable subscriber merging bounded file events with ordinary events.
#[derive(Debug)]
pub struct ExportReceiver {
    ordinary: std::sync::mpsc::Receiver<SeqEvent>,
    exports: std::sync::mpsc::Receiver<SeqEvent>,
    heads: Mutex<[Option<SeqEvent>; 2]>,
}

/// A file event could not be delivered safely.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportSendError {
    /// The owning request was cancelled.
    Cancelled,
    /// A caller supplied an oversized or malformed file event.
    InvalidFrame,
}

impl std::fmt::Display for ExportSendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Cancelled => "Export transfer cancelled",
            Self::InvalidFrame => "Invalid export event",
        })
    }
}

impl std::error::Error for ExportSendError {}

impl ExportReceiver {
    /// Reads the lowest available sequence ID without reordering export boundaries.
    ///
    /// # Errors
    /// Returns `Empty` with no ready event, or `Disconnected` after both queues close.
    ///
    /// # Panics
    /// Panics if the internal populated-head invariant is violated.
    pub fn try_recv(&self) -> Result<SeqEvent, std::sync::mpsc::TryRecvError> {
        let mut heads = BroadcastBus::lock(&self.heads);
        let mut disconnected = 0;
        for (index, receiver) in [&self.ordinary, &self.exports].into_iter().enumerate() {
            if heads[index].is_none() {
                match receiver.try_recv() {
                    Ok(event) => heads[index] = Some(event),
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => disconnected += 1,
                    Err(std::sync::mpsc::TryRecvError::Empty) => {}
                }
            }
        }
        let index = match (&heads[0], &heads[1]) {
            (Some(a), Some(b)) => usize::from(b.id < a.id),
            (Some(_), None) => 0,
            (None, Some(_)) => 1,
            (None, None) => {
                return Err(if disconnected == 2 {
                    std::sync::mpsc::TryRecvError::Disconnected
                } else {
                    std::sync::mpsc::TryRecvError::Empty
                });
            }
        };
        Ok(heads[index]
            .take()
            .expect("selected a populated queue head"))
    }

    /// Iterates over events currently available without blocking.
    pub fn try_iter(&self) -> impl Iterator<Item = SeqEvent> + '_ {
        std::iter::from_fn(|| self.try_recv().ok())
    }
}

// Eight maximum-sized chunks retain at most 256 KiB of raw file text per subscriber.
const EXPORT_QUEUE_EVENTS: usize = 8;
// A stalled reader loses only its current export, rather than blocking the agent indefinitely.
const EXPORT_STALL_LIMIT: std::time::Duration = std::time::Duration::from_secs(2);

impl BroadcastBus {
    /// A new, empty bus.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a new consumer, returning the receiver end. The subscriber is
    /// pruned automatically on the first `broadcast` after it hangs up.
    pub fn subscribe(&self) -> std::sync::mpsc::Receiver<SeqEvent> {
        let (tx, rx) = std::sync::mpsc::channel();
        Self::lock(&self.inner).subscribers.push(Subscriber {
            ordinary: tx,
            exports: None,
            failed_export: Arc::new(Mutex::new(None)),
        });
        rx
    }

    /// Registers an export-capable consumer with bounded file-event storage.
    pub fn subscribe_exports(&self) -> ExportReceiver {
        let (ordinary, ordinary_rx) = std::sync::mpsc::channel();
        let (exports, exports_rx) = std::sync::mpsc::sync_channel(EXPORT_QUEUE_EVENTS);
        Self::lock(&self.inner).subscribers.push(Subscriber {
            ordinary,
            exports: Some(exports),
            failed_export: Arc::new(Mutex::new(None)),
        });
        ExportReceiver {
            ordinary: ordinary_rx,
            exports: exports_rx,
            heads: Mutex::new([None, None]),
        }
    }

    /// Reports whether any subscriber opted into complete-file transfers.
    #[must_use]
    pub fn has_export_subscribers(&self) -> bool {
        Self::lock(&self.inner)
            .subscribers
            .iter()
            .any(|subscriber| subscriber.exports.is_some())
    }

    /// Atomically captures replay and subscribes to every subsequent live event.
    ///
    /// File events use bounded storage only when `exports` is enabled. Events
    /// published after the snapshot boundary are queued while replay is sent.
    #[must_use]
    pub fn subscribe_replay(
        &self,
        exports: bool,
        resume_from: Option<u64>,
    ) -> (ExportReceiver, Vec<SeqEvent>, Option<u64>) {
        let (ordinary, ordinary_rx) = std::sync::mpsc::channel();
        let (files, files_rx) = std::sync::mpsc::sync_channel(EXPORT_QUEUE_EVENTS);
        let mut inner = Self::lock(&self.inner);
        let tail = inner
            .scrollback
            .iter()
            .filter(|event| resume_from.is_none_or(|from| event.id > from))
            .cloned()
            .collect();
        let highest = inner.next_id.checked_sub(1);
        inner.subscribers.push(Subscriber {
            ordinary,
            exports: exports.then_some(files),
            failed_export: Arc::new(Mutex::new(None)),
        });
        (
            ExportReceiver {
                ordinary: ordinary_rx,
                exports: files_rx,
                heads: Mutex::new([None, None]),
            },
            tail,
            highest,
        )
    }

    /// Delivers file events with cancellable backpressure and explicit partial retirement.
    ///
    /// # Errors
    /// Returns cancellation after retirement, or `InvalidFrame` without enqueueing malformed data.
    pub fn broadcast_artifact(
        &self,
        event: &crate::export::transfer::Transfer,
        cancelled: impl Fn() -> bool,
    ) -> Result<(), ExportSendError> {
        use crate::export::transfer::Transfer;
        if !event.is_valid() {
            return Err(ExportSendError::InvalidFrame);
        }
        let (id, begin, abort) = match event {
            Transfer::Begin { id, .. } => (id, true, false),
            Transfer::Chunk { id, .. } | Transfer::End { id, .. } => (id, false, false),
            Transfer::Abort { id, .. } => (id, false, true),
        };
        let subscribers = Self::lock(&self.inner)
            .subscribers
            .iter()
            .filter_map(|subscriber| {
                subscriber.exports.as_ref().map(|exports| {
                    (
                        subscriber.ordinary.clone(),
                        exports.clone(),
                        Arc::clone(&subscriber.failed_export),
                    )
                })
            })
            .collect::<Vec<_>>();
        let mut was_cancelled = false;
        for (ordinary, exports, failed) in subscribers {
            if begin {
                *Self::lock(&failed) = None;
            }
            if !abort && Self::lock(&failed).as_deref() == Some(id.as_str()) {
                continue;
            }
            let started = std::time::Instant::now();
            loop {
                let stopping = cancelled();
                if abort || stopping || started.elapsed() >= EXPORT_STALL_LIMIT {
                    *Self::lock(&failed) = Some(id.clone());
                    let reason = if stopping {
                        "Export transfer cancelled"
                    } else {
                        "Export receiver stalled"
                    };
                    let retired = if abort {
                        event.clone()
                    } else {
                        Transfer::Abort {
                            id: id.clone(),
                            reason: reason.into(),
                        }
                    };
                    let mut inner = Self::lock(&self.inner);
                    let sequence = inner.next_id;
                    inner.next_id += 1;
                    let _ = ordinary.send(SeqEvent {
                        id: sequence,
                        event: UiEvent::Artifact(retired),
                    });
                    was_cancelled |= stopping;
                    break;
                }
                let result = {
                    let mut inner = Self::lock(&self.inner);
                    let sequence = inner.next_id;
                    inner.next_id += 1;
                    exports.try_send(SeqEvent {
                        id: sequence,
                        event: UiEvent::Artifact(event.clone()),
                    })
                };
                match result {
                    Ok(()) | Err(std::sync::mpsc::TrySendError::Disconnected(_)) => break,
                    Err(std::sync::mpsc::TrySendError::Full(_)) => {
                        // Never hold the bus lock while waiting; ordinary output stays live.
                        std::thread::sleep(std::time::Duration::from_millis(10));
                    }
                }
            }
        }
        if was_cancelled {
            Err(ExportSendError::Cancelled)
        } else {
            Ok(())
        }
    }

    /// Assigns the next sequence id, appends to the scrollback ring (evicting
    /// the oldest beyond [`SCROLLBACK_CAP`]), and fans the event out to every
    /// live subscriber, dropping any that have hung up.
    pub fn broadcast(&self, event: UiEvent) {
        if let UiEvent::Artifact(event) = event {
            let _ = self.broadcast_artifact(&event, || false);
            return;
        }
        let mut inner = Self::lock(&self.inner);
        // A reset invalidates everything before it: drop the retained tail so a
        // client attaching afterwards is not replayed the transcript the
        // operator just cleared. Ids keep climbing, so an already-connected
        // client's `resume_from` stays meaningful across the reset.
        if matches!(event, UiEvent::SessionReset) {
            inner.scrollback.clear();
        }
        // Original documents are current interactive state, not transcript
        // history. Keep only the pending one and retire its body on closure.
        match &event {
            UiEvent::Editor(_) => inner.scrollback.retain(|old| !matches!(old.event, UiEvent::Editor(_))),
            UiEvent::EditorDone { id } => inner.scrollback.retain(|old| {
                !matches!(&old.event, UiEvent::Editor(request) if request.as_ref()["id"].as_str() == Some(id.as_str()))
            }),
            _ => {}
        }
        // Snapshots replace older readings so idle polling preserves transcript
        // history and does not retain thousands of copies of large reports.
        if matches!(
            event,
            UiEvent::Jobs(_)
                | UiEvent::Tasks(_)
                | UiEvent::Status(_)
                | UiEvent::Controls(_)
                | UiEvent::Sessions(_)
                | UiEvent::Config(_)
                | UiEvent::History(_)
                | UiEvent::Extensions(_)
                | UiEvent::Cache(_)
                | UiEvent::Downloads(_)
                | UiEvent::Metrics(_)
                | UiEvent::Host(_)
        ) && let Some(index) = inner.scrollback.iter().rposition(|old| {
            matches!(
                (&event, &old.event),
                (UiEvent::Jobs(_), UiEvent::Jobs(_))
                    | (UiEvent::Tasks(_), UiEvent::Tasks(_))
                    | (UiEvent::Status(_), UiEvent::Status(_))
                    | (UiEvent::Controls(_), UiEvent::Controls(_))
                    | (UiEvent::Sessions(_), UiEvent::Sessions(_))
                    | (UiEvent::Config(_), UiEvent::Config(_))
                    | (UiEvent::Cache(_), UiEvent::Cache(_))
                    | (UiEvent::Downloads(_), UiEvent::Downloads(_))
                    | (UiEvent::Host(_), UiEvent::Host(_))
            ) || matches!((&event, &old.event), (UiEvent::History(a), UiEvent::History(b)) if a.kind == b.kind)
                || matches!((&event, &old.event), (UiEvent::Extensions(a), UiEvent::Extensions(b)) if a.kind == b.kind)
                || matches!((&event, &old.event), (UiEvent::Metrics(a), UiEvent::Metrics(b)) if a.kind() == b.kind())
        }) {
            inner.scrollback.remove(index);
        }
        let id = inner.next_id;
        inner.next_id += 1;
        let seq = SeqEvent { id, event };
        inner.scrollback.push_back(seq.clone());
        while inner.scrollback.len() > SCROLLBACK_CAP {
            // A pending editor remains available to late joiners even when
            // unrelated output fills the replay ring. At most one is pinned.
            let evict = inner
                .scrollback
                .iter()
                .position(|old| !matches!(old.event, UiEvent::Editor(_)))
                .unwrap_or(0);
            inner.scrollback.remove(evict);
        }
        inner
            .subscribers
            .retain(|subscriber| subscriber.ordinary.send(seq.clone()).is_ok());
    }

    /// Returns scrollback events with `id > resume_from` for late-join replay.
    /// A `resume_from` older than the retained tail yields the whole tail
    /// (best-effort resume, §4.8). Also returns the highest id assigned so far,
    /// or `None` when nothing has been broadcast yet.
    #[must_use]
    pub fn scrollback_since(&self, resume_from: Option<u64>) -> (Vec<SeqEvent>, Option<u64>) {
        let inner = Self::lock(&self.inner);
        let highest = inner.next_id.checked_sub(1);
        let tail = inner
            .scrollback
            .iter()
            .filter(|s| resume_from.is_none_or(|from| s.id > from))
            .cloned()
            .collect();
        (tail, highest)
    }

    /// Number of currently registered subscribers (test/introspection helper).
    #[must_use]
    pub fn subscriber_count(&self) -> usize {
        Self::lock(&self.inner).subscribers.len()
    }

    fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
        m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// A [`RenderSink`] that forwards render calls into a [`BroadcastBus`]. Mirrors
/// [`ChannelSink`] but fans out to every subscriber instead of one channel.
#[derive(Debug)]
pub struct BusSink<'a>(pub &'a BroadcastBus);

impl RenderSink for BusSink<'_> {
    fn visible_text(&mut self, text: &str) {
        self.0.broadcast(UiEvent::Visible(text.to_owned()));
    }
    fn think_text(&mut self, text: &str) {
        self.0.broadcast(UiEvent::Think(text.to_owned()));
    }
    fn tool_text(&mut self, text: &str) {
        self.0.broadcast(UiEvent::Tool(text.to_owned()));
    }
    fn error_text(&mut self, text: &str) {
        self.0.broadcast(UiEvent::Error(text.to_owned()));
    }
}

/// Mirrors rendered output while preserving the primary sink's behavior.
#[derive(Debug)]
pub(crate) struct MirrorSink<S> {
    pub inner: S,
    pub bus: Option<Arc<BroadcastBus>>,
    pub activity: Option<crate::console_input::output::SubagentOutput>,
    pub suppress_primary: bool,
    pub aside: bool,
    pub gui: bool,
}

/// Publishes already-rendered side output through typed console and remote frames.
pub(crate) fn publish_aside(gui: bool, bus: Option<&BroadcastBus>, event: UiEvent) {
    let event = UiEvent::Btw(Box::new(event));
    let Some(crate::remote::control::ServerMsg::Aside { role, text }) =
        crate::remote::control::ServerMsg::from_event(&event)
    else {
        return;
    };
    let mut offset = 0;
    while offset < text.len() {
        // Bound worst-case JSON escaping within the client's frame ceiling.
        let mut end = offset.saturating_add(32 * 1024).min(text.len());
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        let chunk = &text[offset..end];
        if gui {
            use std::io::Write as _;
            let _ = writeln!(
                std::io::stderr().lock(),
                "+PLANK_TELEMETRY {}",
                serde_json::json!({"type":"aside", "role":role, "text":chunk})
            );
        }
        if let Some(bus) = bus {
            let inner = match role.as_str() {
                "answer" => UiEvent::Visible(chunk.into()),
                "thinking" => UiEvent::Think(chunk.into()),
                "tool" => UiEvent::Tool(chunk.into()),
                "error" => UiEvent::Error(chunk.into()),
                _ => UiEvent::Dim(chunk.into()),
            };
            bus.broadcast(UiEvent::Btw(Box::new(inner)));
        }
        offset = end;
    }
}

impl<S: RenderSink> RenderSink for MirrorSink<S> {
    fn visible_text(&mut self, text: &str) {
        if !self.suppress_primary {
            self.inner.visible_text(text);
        }
        if let Some(activity) = &mut self.activity {
            activity.visible_text(text);
        }
        if self.aside {
            publish_aside(self.gui, self.bus.as_deref(), UiEvent::Visible(text.into()));
            return;
        }
        if let Some(bus) = &self.bus
            && self.activity.is_none()
        {
            BusSink(bus).visible_text(text);
        }
    }
    fn think_text(&mut self, text: &str) {
        if !self.suppress_primary {
            self.inner.think_text(text);
        }
        if let Some(activity) = &mut self.activity {
            activity.think_text(text);
        }
        if self.aside {
            publish_aside(self.gui, self.bus.as_deref(), UiEvent::Think(text.into()));
            return;
        }
        if let Some(bus) = &self.bus
            && self.activity.is_none()
        {
            BusSink(bus).think_text(text);
        }
    }
    fn tool_text(&mut self, text: &str) {
        if !self.suppress_primary {
            self.inner.tool_text(text);
        }
        if let Some(activity) = &mut self.activity {
            activity.tool_text(text);
        }
        if self.aside {
            publish_aside(self.gui, self.bus.as_deref(), UiEvent::Tool(text.into()));
            return;
        }
        if let Some(bus) = &self.bus
            && self.activity.is_none()
        {
            BusSink(bus).tool_text(text);
        }
    }
    fn error_text(&mut self, text: &str) {
        if !self.suppress_primary {
            self.inner.error_text(text);
        }
        if let Some(activity) = &mut self.activity {
            activity.error_text(text);
        }
        if self.aside {
            publish_aside(self.gui, self.bus.as_deref(), UiEvent::Error(text.into()));
            return;
        }
        if let Some(bus) = &self.bus
            && self.activity.is_none()
        {
            BusSink(bus).error_text(text);
        }
    }
    fn preview_status(&mut self, text: Option<&str>) {
        // Rewriting a transient line belongs only to the owning console.
        if !self.suppress_primary {
            self.inner.preview_status(text);
        }
    }
}

/// State shared by the UI and worker threads for one turn.
#[derive(Debug, Default)]
pub struct TurnShared {
    /// Explicit document replies, separate from model prompts and tool questions.
    pub(crate) editors: crate::remote::editors::Editors,
    /// Matches remote replies without putting them into the model's prompt queue.
    pub(crate) questions: crate::remote::questions::Questions,
    /// Remote controls wait for the idle dispatcher, never the model's prompt drain.
    commands: Mutex<Vec<String>>,
    /// Set by the UI (Esc / Ctrl-C / SIGINT) to stop the worker at the next
    /// sampling or prefill checkpoint.
    pub interrupt: AtomicBool,
    /// Set by the front end while the worker is running a *quiet background
    /// pass* rather than a turn — the memory pass, or a prompt suggestion.
    /// (The name predates the second one; the flag has never meant that a
    /// memory job exists, and nothing that reads it assumes one does. The
    /// footer's notes mark comes from `extract_state`, not from here.) The
    /// busy loop then treats a submitted prompt as an interrupt as well as a
    /// queued line: the user wants the model, and the pass goes back where it
    /// came from for the next idle moment.
    pub memory_pass: AtomicBool,
    /// Set by the front end for the whole duration of a quiet background
    /// pass (the memory pass, a suggestion). Where `memory_pass` tells the
    /// busy loop that a submitted prompt may interrupt the pass, this tells
    /// it that a pass is running at all, so a mid-pass `/btw` is rejected
    /// (`btw_gate`).
    pub background_pass: AtomicBool,
    /// Set by the busy loop's Ctrl-D arm while a quiet background pass is
    /// running: the user wants out, and work they never asked for must not
    /// hold the door shut. `tui_quiet_pass` reads it once the worker has
    /// stopped and hands the quit to the idle loop, which takes its normal
    /// exit path.
    /// Never set during a real turn — see the arm's `memory_pass` guard.
    ///
    /// Distinct from the file-scope `QUIT_REQUESTED` static in `ui.rs` and
    /// its same-named reader `quit_requested()`, which belong to the
    /// mid-turn `/exit` confirmation pane. The two quit channels are
    /// disjoint; only the name is shared.
    pub quit_requested: AtomicBool,
    /// Set by the UI when a `/btw` is submitted mid-turn: the current main
    /// generation pass stops immediately so the side question is answered
    /// with priority, then the interrupted pass is re-run (nothing is
    /// committed until a pass completes, so the restart is transcript-safe).
    pub preempt: AtomicBool,
    /// User lines typed while the worker is busy. The worker drains them into
    /// the transcript between tool rounds (the C's `queued_user_drain`);
    /// lines still queued when the turn ends start a fresh turn.
    pub queued: Mutex<Vec<String>>,
    /// Idle-dispatched remote prompts and a bounded recent-ID deduplication window.
    prompt_requests: Mutex<PromptRequests>,
    /// `/btw` side questions queued while the worker is busy, answered FIFO
    /// at generation boundaries (modeled on `OpenClaw`'s side-question queue).
    pub btw: Mutex<Vec<String>>,
    /// Snapshot of the background bash job table, refreshed by the worker at
    /// each pass start and tool boundary, so the UI thread can show `/jobs`
    /// and the footer's jobs panel while the worker owns the live table.
    pub jobs: Mutex<Vec<crate::tools::bash::JobRow>>,
    /// Token counts behind the `/context` report, refreshed by the worker
    /// wherever the transcript changes, so the UI thread can draw the panel
    /// mid-turn against the live fill while the worker owns the agent.
    pub context: Mutex<crate::ctxreport::Breakdown>,
    /// The `/repro` dump's agent-side half, published by the worker at the
    /// start of every main pass, so the UI thread can write a dump mid-turn
    /// while the worker owns the agent. `None` before the first pass.
    pub repro: Mutex<Option<crate::repro::ReproBase>>,
    /// The text the running main pass has generated so far, appended token by
    /// token by the worker and reset at each pass start; a mid-turn `/repro`
    /// writes it below the transcript, which is what makes the dump the
    /// model's state *now* rather than at the last boundary.
    pub live_pass: Mutex<String>,
    /// The path of the last mid-turn `/repro`, handed to the agent when the
    /// turn ends so a bare `/open` aims at it, exactly as an idle `/repro`
    /// sets `last_edited` directly.
    pub repro_written: Mutex<Option<std::path::PathBuf>>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct CompletedInput {
    pub(crate) command: bool,
    pub(crate) legacy: bool,
}

#[derive(Debug, Default)]
struct PromptRequests {
    pending: std::collections::HashSet<String>,
    queued: Vec<(String, String)>,
    recent: std::collections::VecDeque<String>,
    dispatched: std::collections::HashSet<String>,
    running: std::collections::HashSet<String>,
    cancelled: std::collections::HashSet<String>,
    commands: std::collections::HashSet<String>,
    owners: std::collections::HashMap<String, crate::remote::control::ControlStamp>,
    legacy: std::collections::HashSet<String>,
    legacy_sequence: u64,
}

impl PromptRequests {
    fn enqueue(
        &mut self,
        id: String,
        text: String,
        command: bool,
        owner: Option<crate::remote::control::ControlStamp>,
        legacy: bool,
    ) -> Result<(), String> {
        if self.recent.contains(&id) || self.pending.contains(&id) {
            return Err("input request ID already used".into());
        }
        if self.queued.len() >= 64 {
            return Err("identified input queue is full".into());
        }
        self.recent.push_back(id.clone());
        self.pending.insert(id.clone());
        while self.recent.len() > 128 {
            self.recent.pop_front();
        }
        if command {
            self.commands.insert(id.clone());
        }
        if let Some(owner) = owner {
            self.owners.insert(id.clone(), owner);
        }
        if legacy {
            self.legacy.insert(id.clone());
        }
        self.queued.push((id, text));
        Ok(())
    }
}

/// Cap on queued `/btw` questions; a push beyond it drops the oldest entry
/// (`OpenClaw`'s bounded-buffer `drop-oldest` overflow policy).
pub const BTW_QUEUE_CAP: usize = 20;

/// Dim marker shown when a mid-generation `/btw` freezes the main task
/// (BTW-SUSPEND-DESIGN §4.4). Emitted into the main log before the aside's
/// side panel opens.
pub const BTW_SUSPEND_MARKER: &str = "[btw — main task paused]";

/// Dim marker shown when the frozen main task resumes after the aside(s)
/// (BTW-SUSPEND-DESIGN §4.4).
pub const BTW_RESUME_MARKER: &str = "[btw — resuming]";

/// Whether a `/btw` submitted right now has anything to run beside, and if
/// not, what to tell the user. Pure so it can be unit-tested without a
/// worker: `background_pass` is `TurnShared::background_pass`. During a real
/// turn (`background_pass` false) there is always a main pass to preempt.
/// During a quiet background pass (the memory pass, a suggestion) a `/btw`
/// has nothing to preempt and nothing would resume, so it is rejected.
#[must_use]
pub fn btw_gate(background_pass: bool) -> Option<&'static str> {
    background_pass.then_some(
        "[/btw has nothing to run beside right now — \
         just type your prompt; it starts at once]",
    )
}

#[cfg(test)]
mod btw_gate_tests {
    use super::btw_gate;

    #[test]
    fn no_background_pass_allows_btw() {
        assert_eq!(btw_gate(false), None);
    }

    #[test]
    fn a_background_pass_rejects_with_nothing_to_run_beside() {
        let msg = btw_gate(true).expect("rejected");
        assert!(msg.contains("nothing to run beside"));
    }
}

impl TurnShared {
    /// Queues an identified prompt for the idle dispatcher, without mid-turn injection.
    ///
    /// # Errors
    /// Returns an error for invalid input, a reused ID, or a full queue.
    pub fn push_prompt_request(&self, id: String, text: String) -> Result<(), String> {
        self.push_input_request(id, text, false, None)
    }

    /// Queues an identified slash command for the same idle dispatcher.
    ///
    /// # Errors
    /// Returns an error for invalid input, a reused ID, or a full queue.
    pub fn push_command_request(&self, id: String, text: String) -> Result<(), String> {
        self.push_input_request(id, text, true, None)
    }

    fn push_input_request(
        &self,
        id: String,
        text: String,
        command: bool,
        owner: Option<crate::remote::control::ControlStamp>,
    ) -> Result<(), String> {
        if id.is_empty()
            || id.len() > 128
            || !id
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
            || text.trim().is_empty()
            || text.trim_start().starts_with('/') != command
            || text.len() > 256 * 1024
        {
            return Err("invalid identified input".into());
        }
        let mut requests = self
            .prompt_requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        requests.enqueue(id, text, command, owner, false)
    }

    /// Associates an identified network input with its accepting controller lease.
    pub(crate) fn push_remote_request(
        &self,
        owner: crate::remote::control::ControlStamp,
        id: String,
        text: String,
        command: bool,
    ) -> Result<(), String> {
        self.push_input_request(id, text, command, Some(owner))
    }

    /// Gives legacy network input an internal identity and the same idle fence.
    pub(crate) fn push_legacy_request(
        &self,
        owner: crate::remote::control::ControlStamp,
        text: String,
    ) -> Result<(), String> {
        if text.trim().is_empty() || text.len() > 256 * 1024 {
            return Err("invalid legacy input".into());
        }
        let mut requests = self
            .prompt_requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let id = loop {
            requests.legacy_sequence = requests
                .legacy_sequence
                .checked_add(1)
                .ok_or("legacy input identity exhausted")?;
            let id = format!("legacy-{}-{}", owner.session, requests.legacy_sequence);
            if !requests.recent.contains(&id)
                && !requests.dispatched.contains(&id)
                && !requests.running.contains(&id)
            {
                break id;
            }
        };
        let command = text.trim_start().starts_with('/');
        requests.enqueue(id, text, command, Some(owner), true)
    }

    pub(crate) fn request_owner(&self, id: &str) -> Option<crate::remote::control::ControlStamp> {
        self.prompt_requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .owners
            .get(id)
            .copied()
    }

    pub(crate) fn request_is_legacy(&self, id: &str) -> bool {
        self.prompt_requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .legacy
            .contains(id)
    }

    pub(crate) fn request_is_pending(&self, id: &str) -> bool {
        let requests = self
            .prompt_requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        requests.dispatched.contains(id) || requests.running.contains(id)
    }

    /// Detaches network-owned inputs when the local operator reclaims control.
    pub(crate) fn take_remote_requests(&self) -> Vec<(String, String)> {
        let mut requests = self
            .prompt_requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let queued = std::mem::take(&mut requests.queued);
        let (remote, local): (Vec<_>, Vec<_>) = queued
            .into_iter()
            .partition(|(id, _)| requests.owners.contains_key(id));
        requests.queued = local;
        requests
            .dispatched
            .extend(remote.iter().map(|(id, _)| id.clone()));
        remote
    }

    /// Takes identified prompts and controls only for the idle dispatcher.
    pub fn take_prompt_requests(&self) -> Vec<(String, String)> {
        let mut requests = self
            .prompt_requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let queued = std::mem::take(&mut requests.queued);
        requests
            .dispatched
            .extend(queued.iter().map(|(id, _)| id.clone()));
        queued
    }

    /// Claims a dispatched input unless a cancellation reached it first.
    #[must_use]
    pub fn begin_prompt_request(&self, id: &str) -> bool {
        let mut requests = self
            .prompt_requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !requests.dispatched.remove(id) {
            return false;
        }
        if requests.cancelled.remove(id) {
            return false;
        }
        requests.running.insert(id.into());
        true
    }

    /// Cancels queued/dispatched input, or interrupts the batch already running it.
    #[must_use]
    pub fn cancel_prompt_request(&self, id: &str) -> bool {
        let mut requests = self
            .prompt_requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(index) = requests
            .queued
            .iter()
            .position(|(request, _)| request == id)
        {
            requests.queued.remove(index);
            return true;
        }
        if requests.dispatched.contains(id) {
            requests.cancelled.insert(id.into());
            return true;
        }
        if requests.running.contains(id) {
            requests.cancelled.insert(id.into());
            self.interrupt
                .store(true, std::sync::atomic::Ordering::Relaxed);
        }
        false
    }

    /// Retains cancellation across the busy UI's transient interrupt reset.
    #[must_use]
    pub fn prompt_batch_cancelled(&self) -> bool {
        let requests = self
            .prompt_requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        requests
            .running
            .iter()
            .any(|id| requests.cancelled.contains(id))
    }

    /// Retires a finished request so late cancellation cannot interrupt another turn.
    pub fn finish_prompt_request(&self, id: &str) {
        let _ = self.retire_prompt_request(id);
    }

    /// Captures completion identity and retires it exactly once under one lock.
    pub(crate) fn retire_prompt_request(&self, id: &str) -> Option<CompletedInput> {
        let mut requests = self
            .prompt_requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !requests.pending.remove(id) {
            return None;
        }
        let completed = CompletedInput {
            command: requests.commands.contains(id),
            legacy: requests.legacy.contains(id),
        };
        requests.queued.retain(|(request, _)| request != id);
        requests.dispatched.remove(id);
        requests.running.remove(id);
        requests.cancelled.remove(id);
        requests.commands.remove(id);
        requests.owners.remove(id);
        requests.legacy.remove(id);
        Some(completed)
    }

    /// Identifies completion kind before retiring a cancelled request.
    #[must_use]
    pub fn is_command_request(&self, id: &str) -> bool {
        self.prompt_requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .commands
            .contains(id)
    }

    /// Reports whether identified remote inputs are waiting for the idle dispatcher.
    #[must_use]
    pub fn has_prompt_requests(&self) -> bool {
        !self
            .prompt_requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .queued
            .is_empty()
    }

    /// Queues a remote slash command independently of model prompts.
    pub fn push_command(&self, line: String) {
        self.commands
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(line);
    }

    /// Takes controls only when the front end can dispatch them.
    pub fn take_commands(&self) -> Vec<String> {
        std::mem::take(
            &mut *self
                .commands
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        )
    }

    /// Whether a control is waiting for the idle dispatcher.
    #[must_use]
    pub fn has_commands(&self) -> bool {
        !self
            .commands
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_empty()
    }
    /// Replaces the shared job snapshot.
    pub fn set_jobs(&self, rows: Vec<crate::tools::bash::JobRow>) {
        if let Ok(mut g) = self.jobs.lock() {
            *g = rows;
        }
    }

    /// Replaces the shared `/context` breakdown.
    pub fn set_context(&self, breakdown: crate::ctxreport::Breakdown) {
        if let Ok(mut g) = self.context.lock() {
            *g = breakdown;
        }
    }

    /// Publishes the `/repro` base for the pass about to run and empties the
    /// live-pass buffer, so a dump taken during it pairs this base with only
    /// this pass's output.
    pub fn begin_repro_pass(&self, base: crate::repro::ReproBase) {
        if let Ok(mut g) = self.repro.lock() {
            *g = Some(base);
        }
        if let Ok(mut g) = self.live_pass.lock() {
            g.clear();
        }
    }

    /// Appends a generated chunk to the live-pass buffer.
    pub fn push_live_pass(&self, text: &str) {
        if let Ok(mut g) = self.live_pass.lock() {
            g.push_str(text);
        }
    }

    /// Drops the published `/repro` base at turn end, so a `TurnShared` that
    /// outlives the turn (the remote bridge's) cannot serve a stale one.
    pub fn end_repro(&self) {
        if let Ok(mut g) = self.repro.lock() {
            *g = None;
        }
        if let Ok(mut g) = self.live_pass.lock() {
            g.clear();
        }
    }

    /// Writes a `/repro` dump from the UI thread, mid-turn: the base the
    /// worker published at pass start plus the pass's output so far, under
    /// `prefix`. `Err` names the reason, including the no-base-yet case
    /// (the turn has not reached its first main pass).
    ///
    /// # Errors
    /// Returns a message when no base has been published or a file cannot be
    /// written.
    pub fn write_repro(
        &self,
        note: &str,
        prefix: &str,
        secs: u64,
    ) -> Result<(std::path::PathBuf, usize), String> {
        let base = self
            .repro
            .lock()
            .ok()
            .and_then(|g| g.clone())
            .ok_or_else(|| {
                "the turn has not started generating yet; try again in a moment".to_owned()
            })?;
        let partial = self
            .live_pass
            .lock()
            .map_or_else(|_| String::new(), |g| g.clone());
        let note = match note.trim() {
            "" => crate::repro::MID_TURN_NOTE,
            n => n,
        };
        // The base was rendered with an empty note; stamp the real one onto
        // the header's line (the first `note:` in the file) so the dump reads
        // as what it is from the top, not only from the in-progress section.
        let report = base
            .report
            .replacen("- note: (none)\n", &format!("- note: {note}\n"), 1);
        let mut report = report;
        crate::repro::append_live_pass(&mut report, note, &partial);
        let out = base.save(prefix, secs, &report)?;
        if let Ok(mut g) = self.repro_written.lock() {
            *g = Some(out.0.clone());
        }
        Ok(out)
    }

    /// Takes the path of the last mid-turn `/repro`, if one was written.
    pub fn take_repro_written(&self) -> Option<std::path::PathBuf> {
        self.repro_written.lock().ok().and_then(|mut g| g.take())
    }

    /// The `/context` breakdown the worker last published.
    #[must_use]
    pub fn context(&self) -> crate::ctxreport::Breakdown {
        self.context
            .lock()
            .map_or_else(|_| crate::ctxreport::Breakdown::default(), |g| g.clone())
    }

    /// The `/jobs` text for the shared snapshot, elapsed times as of now.
    #[must_use]
    pub fn jobs_report(&self) -> String {
        self.jobs
            .lock()
            .map_or_else(|_| String::new(), |g| crate::tools::bash::render_rows(&g))
    }

    /// Takes all queued user lines.
    pub fn take_queued(&self) -> Vec<String> {
        self.queued.lock().map_or_else(
            |e| std::mem::take(&mut *e.into_inner()),
            |mut q| std::mem::take(&mut *q),
        )
    }

    /// Whether any user line worth running is queued, without taking it.
    /// A blank line does not count: the idle loop skips it when the queue
    /// runs, so it must not be what takes a screensaver down.
    #[must_use]
    pub fn has_queued(&self) -> bool {
        let any = |q: &Vec<String>| q.iter().any(|l| !l.trim().is_empty());
        self.queued
            .lock()
            .map_or_else(|e| any(&e.into_inner()), |q| any(&q))
    }

    /// Queues one user line for the worker.
    pub fn push_queued(&self, line: String) {
        match self.queued.lock() {
            Ok(mut q) => q.push(line),
            Err(e) => e.into_inner().push(line),
        }
    }

    /// Queues one `/btw` question (FIFO, capped at [`BTW_QUEUE_CAP`]).
    /// Returns the oldest entry when the cap forced it out, so the caller can
    /// surface a visible drop notice instead of losing it silently.
    pub fn push_btw(&self, question: String) -> Option<String> {
        let mut q = match self.btw.lock() {
            Ok(q) => q,
            Err(e) => e.into_inner(),
        };
        q.push(question);
        if q.len() > BTW_QUEUE_CAP {
            Some(q.remove(0))
        } else {
            None
        }
    }

    /// Takes the oldest queued `/btw` question.
    pub fn pop_btw(&self) -> Option<String> {
        let mut q = match self.btw.lock() {
            Ok(q) => q,
            Err(e) => e.into_inner(),
        };
        if q.is_empty() {
            None
        } else {
            Some(q.remove(0))
        }
    }

    /// Drops all queued `/btw` questions, returning how many were cleared.
    pub fn clear_btw(&self) -> usize {
        let mut q = match self.btw.lock() {
            Ok(q) => q,
            Err(e) => e.into_inner(),
        };
        let n = q.len();
        q.clear();
        n
    }

    /// Takes all queued `/btw` questions.
    pub fn take_btw(&self) -> Vec<String> {
        match self.btw.lock() {
            Ok(mut q) => std::mem::take(&mut *q),
            Err(e) => std::mem::take(&mut *e.into_inner()),
        }
    }
}

/// Applies one non-status event to the TUI output log.
pub fn apply(log: &mut OutputLog, ev: UiEvent) {
    match ev {
        UiEvent::Visible(t) => log.visible_text(&t),
        UiEvent::Think(t) => log.think_text(&t),
        UiEvent::Tool(t) => log.tool_text(&t),
        UiEvent::Error(t) => log.error_text(&t),
        UiEvent::Dim(t) => log.push_dim(t),
        UiEvent::PreviewStatus(t) => log.apply_preview_status(t.as_deref()),
        UiEvent::SystemStatus(t) => log.push_ansi(&crate::status::system_line(&t, true)),
        UiEvent::EditCard(p) => crate::tui::render_diff_card(log, &p),
        UiEvent::Markdown(t) => log.push_markdown(&t),
        UiEvent::Plain(t) => log.push_plain(t),
        UiEvent::UserEcho(t) => log.push_user_echo(&t),
        UiEvent::QueuedJoined => log.commit_pending(),
        UiEvent::EndLine => log.end_line(),
        // Both are for remote front-ends only: locally the desktop notification
        // has already fired, and the log was cleared at the `/clear` call site.
        UiEvent::Notify { .. }
        | UiEvent::CommandDone { .. }
        | UiEvent::Controls(_)
        | UiEvent::Question(_)
        | UiEvent::QuestionDone { .. }
        | UiEvent::Editor(_)
        | UiEvent::EditorDone { .. }
        | UiEvent::SessionReset
        | UiEvent::Status(_)
        | UiEvent::Tasks(_)
        | UiEvent::Jobs(_)
        | UiEvent::Sessions(_)
        | UiEvent::Config(_)
        | UiEvent::History(_)
        | UiEvent::Artifact(_)
        | UiEvent::Extensions(_)
        | UiEvent::Cache(_)
        | UiEvent::Downloads(_)
        | UiEvent::PromptDone { .. }
        | UiEvent::CommandRequestDone { .. }
        | UiEvent::Metrics(_)
        | UiEvent::Host(_)
        | UiEvent::BtwBegin
        | UiEvent::BtwEnd
        | UiEvent::MainCheckpoint
        | UiEvent::MainRollback
        | UiEvent::SubStart { .. }
        | UiEvent::SubEnd
        | UiEvent::SubTokens { .. }
        | UiEvent::Sub(_)
        | UiEvent::Btw(_) => {}
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn editor_originals_are_shared_pinned_and_retired_without_entering_closed_replay() {
        let bus = BroadcastBus::new();
        let a = bus.subscribe();
        let b = bus.subscribe();
        let original =
            EditorRequest::from(serde_json::json!({"id":"editor-1","text":"private original"}));
        let weak = Arc::downgrade(&original.0);
        assert!(!format!("{original:?}").contains("private original"));
        bus.broadcast(UiEvent::Editor(original));
        let UiEvent::Editor(first) = a.try_recv().unwrap().event else {
            panic!("expected editor");
        };
        let UiEvent::Editor(second) = b.try_recv().unwrap().event else {
            panic!("expected editor");
        };
        assert!(Arc::ptr_eq(&first.0, &second.0));
        drop(first);
        drop(second);
        for _ in 0..SCROLLBACK_CAP + 2 {
            bus.broadcast(UiEvent::Visible("output".into()));
        }
        {
            let (tail, _) = bus.scrollback_since(None);
            assert_eq!(tail.len(), SCROLLBACK_CAP);
            assert!(
                tail.iter()
                    .any(|event| matches!(event.event, UiEvent::Editor(_)))
            );
            assert!(tail.windows(2).all(|pair| pair[0].id < pair[1].id));
        }
        bus.broadcast(UiEvent::EditorDone { id: "other".into() });
        assert!(weak.upgrade().is_some());
        bus.broadcast(UiEvent::EditorDone {
            id: "editor-1".into(),
        });
        assert!(weak.upgrade().is_none());
        assert!(
            !bus.scrollback_since(None)
                .0
                .iter()
                .any(|event| matches!(event.event, UiEvent::Editor(_)))
        );
        let bus = BroadcastBus::new();
        let original = EditorRequest::from(serde_json::json!({"id":"old","text":"first"}));
        let weak = Arc::downgrade(&original.0);
        bus.broadcast(UiEvent::Editor(original));
        bus.broadcast(UiEvent::Editor(
            serde_json::json!({"id":"new","text":"second"}).into(),
        ));
        assert!(weak.upgrade().is_none());
        bus.broadcast(UiEvent::SessionReset);
        // The reset marker itself is retained (see
        // `session_reset_drops_the_scrollback_it_invalidates`); only the
        // pre-reset editor and transcript are gone.
        let (tail, _) = bus.scrollback_since(None);
        assert_eq!(tail.len(), 1, "only the reset survives: {tail:?}");
        assert!(matches!(tail[0].event, UiEvent::SessionReset));
        assert!(
            !tail
                .iter()
                .any(|event| matches!(event.event, UiEvent::Editor(_)))
        );
    }

    #[test]
    fn replay_subscription_keeps_snapshot_and_live_events_on_opposite_sides() {
        for exports in [false, true] {
            let bus = super::BroadcastBus::new();
            bus.broadcast(super::UiEvent::Visible("Before capture".into()));
            let (receiver, tail, highest) = bus.subscribe_replay(exports, None);
            assert_eq!(tail.len(), 1);
            assert_eq!(Some(tail[0].id), highest);
            assert_eq!(bus.has_export_subscribers(), exports);
            // These events occur while the connection is still sending replay.
            bus.broadcast(super::UiEvent::Visible("During replay".into()));
            bus.broadcast(super::UiEvent::EditorDone {
                id: "closed-during-replay".into(),
            });
            let live: Vec<_> = receiver.try_iter().collect();
            assert_eq!(live.len(), 2);
            assert!(live.iter().all(|event| Some(event.id) > highest));
            let (next, resumed, latest) = bus.subscribe_replay(exports, highest);
            assert_eq!(resumed.len(), 2);
            assert_eq!(latest, Some(live[1].id));
            assert!(next.try_iter().next().is_none());
        }
    }

    #[test]
    fn console_aside_output_is_separate_and_does_not_duplicate_primary_text() {
        use crate::viz::{CollectSink, RenderSink};
        let primary = CollectSink::default();
        let bus = Arc::new(BroadcastBus::new());
        let mut sink = MirrorSink {
            inner: primary.clone(),
            bus: Some(Arc::clone(&bus)),
            activity: None,
            suppress_primary: true,
            aside: true,
            gui: false,
        };
        sink.visible_text("side answer");
        sink.think_text("side reasoning");
        sink.tool_text("side tool");
        sink.error_text("side error");
        assert!(primary.0.lock().unwrap().is_empty());
        let (events, _) = bus.scrollback_since(None);
        assert_eq!(events.len(), 4);
        for (event, expected) in events.iter().zip(["answer", "thinking", "tool", "error"]) {
            assert!(
                matches!(crate::remote::control::ServerMsg::from_event(&event.event),
                Some(crate::remote::control::ServerMsg::Aside { role, .. }) if role == expected)
            );
        }
        sink.aside = false;
        sink.suppress_primary = false;
        sink.visible_text("main answer");
        assert_eq!(*primary.0.lock().unwrap(), "main answer");
    }

    #[test]
    fn console_aside_frames_preserve_large_unicode_output_without_oversized_frames() {
        let bus = BroadcastBus::new();
        let body = "🦀\u{0001}".repeat(50_000);
        publish_aside(false, Some(&bus), UiEvent::Visible(body.clone()));
        let (events, _) = bus.scrollback_since(None);
        let mut received = String::new();
        for event in events {
            let message = crate::remote::control::ServerMsg::from_event(&event.event).unwrap();
            assert!(serde_json::to_vec(&message).unwrap().len() < 256 * 1024);
            if let crate::remote::control::ServerMsg::Aside { text, .. } = message {
                received.push_str(&text);
            }
        }
        assert_eq!(received, body);
    }
    #[test]
    fn export_transfer_backpressure_retires_only_the_stalled_reader() {
        use crate::export::transfer::Transfer;
        use std::sync::atomic::{AtomicUsize, Ordering};
        let bus = Arc::new(BroadcastBus::new());
        let slow = bus.subscribe_exports();
        let fast = bus.subscribe_exports();
        let legacy = bus.subscribe();
        let progress = Arc::new(AtomicUsize::new(0));
        let reader = std::thread::spawn(move || {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            let mut events = Vec::new();
            loop {
                if let Ok(event) = fast.try_recv() {
                    let ended = matches!(event.event, UiEvent::Artifact(Transfer::End { .. }));
                    events.push(event);
                    if ended {
                        break events;
                    }
                } else {
                    assert!(
                        std::time::Instant::now() < deadline,
                        "fast reader never completed"
                    );
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
            }
        });
        let body = "x".repeat(crate::export::transfer::CHUNK_BYTES * 20);
        let publisher = {
            let bus = Arc::clone(&bus);
            let progress = Arc::clone(&progress);
            let body = body.clone();
            std::thread::spawn(move || {
                crate::export::transfer::emit(
                    crate::export::transfer::Purpose::Conversation,
                    "bounded-1",
                    "file.md",
                    crate::export::Format::Markdown,
                    &body,
                    |event| {
                        bus.broadcast_artifact(&event, || false)?;
                        progress.fetch_add(1, Ordering::Relaxed);
                        Ok::<_, ExportSendError>(())
                    },
                )
                .unwrap();
            })
        };
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while progress.load(Ordering::Relaxed) < EXPORT_QUEUE_EVENTS {
            assert!(std::time::Instant::now() < deadline);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let ordinary = {
            let bus = Arc::clone(&bus);
            std::thread::spawn(move || {
                bus.broadcast(UiEvent::Visible("ordinary output remains live".into()));
            })
        };
        assert!(matches!(
            legacy
                .recv_timeout(std::time::Duration::from_millis(500))
                .unwrap()
                .event,
            UiEvent::Visible(_)
        ));
        ordinary.join().unwrap();
        publisher.join().unwrap();
        let events = reader.join().unwrap();
        assert!(events.windows(2).all(|pair| pair[0].id < pair[1].id));
        let received = events
            .iter()
            .filter_map(|event| match &event.event {
                UiEvent::Artifact(Transfer::Chunk { text, .. }) => Some(text.as_str()),
                _ => None,
            })
            .collect::<String>();
        assert_eq!(received, body);
        let slow = slow.try_iter().collect::<Vec<_>>();
        assert!(slow.windows(2).all(|pair| pair[0].id < pair[1].id));
        assert!(
            slow.iter()
                .any(|event| matches!(event.event, UiEvent::Artifact(Transfer::Abort { .. })))
        );
        assert!(
            !slow
                .iter()
                .any(|event| matches!(event.event, UiEvent::Artifact(Transfer::End { .. })))
        );
        assert!(
            slow.iter()
                .filter(|event| matches!(event.event, UiEvent::Artifact(Transfer::Chunk { .. })))
                .count()
                < EXPORT_QUEUE_EVENTS
        );
        assert!(
            legacy.try_recv().is_err(),
            "legacy reader must receive no file payloads"
        );
    }

    #[test]
    fn export_transfer_cancellation_retires_all_readers_and_rejects_unbounded_events() {
        use crate::export::transfer::Transfer;
        let bus = BroadcastBus::new();
        let a = bus.subscribe_exports();
        let b = bus.subscribe_exports();
        let event = Transfer::Chunk {
            id: "export-1".into(),
            offset: 0,
            text: "x".repeat(crate::export::transfer::CHUNK_BYTES + 1),
        };
        assert_eq!(
            bus.broadcast_artifact(&event, || false),
            Err(ExportSendError::InvalidFrame)
        );
        assert!(a.try_recv().is_err());
        assert!(b.try_recv().is_err());
        let event = Transfer::End {
            id: "export-1".into(),
            bytes: 0,
        };
        assert_eq!(
            bus.broadcast_artifact(&event, || true),
            Err(ExportSendError::Cancelled)
        );
        for receiver in [a, b] {
            assert!(matches!(
                receiver.try_recv().unwrap().event,
                UiEvent::Artifact(Transfer::Abort { .. })
            ));
            assert!(receiver.try_recv().is_err());
        }
    }

    #[test]
    fn export_transfer_is_delivered_live_without_replay_or_transcript_mutation() {
        use crate::export::transfer::Transfer;
        let bus = BroadcastBus::new();
        let receiver = bus.subscribe_exports();
        bus.broadcast(UiEvent::Visible("conversation".into()));
        receiver.try_iter().for_each(drop);
        crate::export::transfer::emit(
            crate::export::transfer::Purpose::Conversation,
            "export-test",
            "session.md",
            crate::export::Format::Markdown,
            "complete file 🦀",
            |event| {
                bus.broadcast(UiEvent::Artifact(event));
                Ok::<_, ()>(())
            },
        )
        .unwrap();
        let captured = receiver.try_iter().collect::<Vec<_>>();
        assert_eq!(captured.len(), 3);
        let mut log = crate::tui::OutputLog::new();
        let before = log.line_count();
        for event in captured {
            let message = crate::remote::control::ServerMsg::from_event(&event.event).unwrap();
            let frame = crate::remote::control::ServerFrame::seq(event.id, message);
            let encoded = serde_json::to_vec(&frame).unwrap();
            assert_eq!(
                serde_json::from_slice::<crate::remote::control::ServerFrame>(&encoded).unwrap(),
                frame
            );
            super::apply(&mut log, event.event);
        }
        assert_eq!(log.line_count(), before);
        let (tail, _) = bus.scrollback_since(None);
        assert_eq!(tail.len(), 1);
        assert!(matches!(&tail[0].event, UiEvent::Visible(text) if text == "conversation"));
        bus.broadcast(UiEvent::Artifact(Transfer::End {
            id: "other".into(),
            bytes: 0,
        }));
        bus.broadcast(UiEvent::Visible("after export".into()));
        let (tail, highest) = bus.scrollback_since(None);
        assert_eq!(tail.len(), 2);
        assert_eq!(highest, Some(5));
    }

    #[test]
    fn console_mirroring_preserves_primary_bytes_and_channel_identity() {
        use crate::viz::RenderSink;
        for color in [false, true] {
            let make_sink = || {
                trace_stream::TerminalSink::new(crate::render::TokenRenderer::new(
                    Vec::<u8>::new(),
                    crate::render::RenderOptions {
                        use_color: color,
                        format_thinking: true,
                        format_markdown: true,
                    },
                ))
            };
            let mut baseline = make_sink();
            let bus = std::sync::Arc::new(super::BroadcastBus::new());
            let mut mirrored = super::MirrorSink {
                inner: make_sink(),
                bus: Some(std::sync::Arc::clone(&bus)),
                activity: None,
                suppress_primary: false,
                aside: false,
                gui: false,
            };
            for sink in [
                &mut baseline as &mut dyn RenderSink,
                &mut mirrored as &mut dyn RenderSink,
            ] {
                sink.visible_text("**Answer** · 🦀\n");
                sink.think_text("Reasoning\n");
                sink.tool_text("pattern=**/src.rs a_b `code`\n");
                sink.error_text("invalid tool\n");
                sink.preview_status(Some("Writing…"));
                sink.preview_status(None);
            }
            assert_eq!(
                mirrored.inner.into_renderer().into_sink(),
                baseline.into_renderer().into_sink()
            );
            let (events, _) = bus.scrollback_since(None);
            assert_eq!(events.len(), 4, "transient preview ticks stay local");
            assert!(
                matches!(&events[0].event, super::UiEvent::Visible(text) if text == "**Answer** · 🦀\n")
            );
            assert!(
                matches!(&events[1].event, super::UiEvent::Think(text) if text == "Reasoning\n")
            );
            assert!(
                matches!(&events[2].event, super::UiEvent::Tool(text) if text == "pattern=**/src.rs a_b `code`\n")
            );
            assert!(
                matches!(&events[3].event, super::UiEvent::Error(text) if text == "invalid tool\n")
            );
        }
    }

    #[test]
    fn identified_commands_keep_completion_kind_and_skip_retired_cancellations() {
        use std::sync::atomic::Ordering;
        let shared = super::TurnShared::default();
        shared
            .push_command_request("command-1".into(), "/command".into())
            .unwrap();
        assert!(shared.is_command_request("command-1"));
        assert!(
            shared
                .push_command_request("bad".into(), "ordinary prompt".into())
                .is_err()
        );
        assert!(
            shared
                .push_prompt_request("command-1".into(), "duplicate ID".into())
                .is_err()
        );
        assert_eq!(
            shared.take_prompt_requests(),
            vec![("command-1".into(), "/command".into())]
        );
        assert!(shared.cancel_prompt_request("command-1"));
        assert!(shared.is_command_request("command-1"));
        shared.finish_prompt_request("command-1");
        assert!(
            !shared.begin_prompt_request("command-1"),
            "dispatcher cannot revive a cancellation acknowledged before startup"
        );
        assert!(!shared.is_command_request("command-1"));
        assert!(!shared.interrupt.load(Ordering::Relaxed));
        let frame =
            crate::remote::control::ServerMsg::from_event(&super::UiEvent::CommandRequestDone {
                request_id: "command-1".into(),
            })
            .unwrap();
        let value = serde_json::to_value(frame).unwrap();
        assert_eq!(value["type"], "command_request_done");
        assert_eq!(value["request_id"], "command-1");
    }

    #[test]
    fn identified_prompts_cancel_without_interrupting_another_turn() {
        use std::sync::atomic::Ordering;
        let shared = super::TurnShared::default();
        shared
            .push_prompt_request("queued".into(), "Queued task".into())
            .unwrap();
        assert!(shared.cancel_prompt_request("queued"));
        assert!(!shared.interrupt.load(Ordering::Relaxed));
        assert!(shared.take_prompt_requests().is_empty());
        shared
            .push_prompt_request("dispatched".into(), "Waiting for dispatcher".into())
            .unwrap();
        let _ = shared.take_prompt_requests();
        assert!(shared.cancel_prompt_request("dispatched"));
        assert!(!shared.begin_prompt_request("dispatched"));
        assert!(!shared.interrupt.load(Ordering::Relaxed));
        shared.finish_prompt_request("dispatched");
        shared
            .push_prompt_request("running".into(), "Running task".into())
            .unwrap();
        let _ = shared.take_prompt_requests();
        assert!(shared.begin_prompt_request("running"));
        assert!(!shared.cancel_prompt_request("running"));
        assert!(shared.interrupt.load(Ordering::Relaxed));
        shared.interrupt.store(false, Ordering::Relaxed);
        assert!(
            shared.prompt_batch_cancelled(),
            "busy UI reset cannot lose cancellation before worker startup"
        );
        shared.finish_prompt_request("running");
        assert!(!shared.prompt_batch_cancelled());
        assert!(!shared.cancel_prompt_request("running"));
        assert!(
            !shared.interrupt.load(Ordering::Relaxed),
            "a late cancellation cannot interrupt the next task"
        );
    }

    #[test]
    fn identified_prompts_wait_for_idle_and_reject_duplicate_ids() {
        let shared = super::TurnShared::default();
        shared
            .push_prompt_request("request-1".into(), "First task".into())
            .unwrap();
        assert!(shared.has_prompt_requests());
        assert!(
            shared.take_queued().is_empty(),
            "a running pass cannot inject identified prompts"
        );
        let first = shared.take_prompt_requests();
        assert_eq!(first, vec![("request-1".into(), "First task".into())]);
        assert!(!shared.has_prompt_requests());
        assert!(
            shared
                .push_prompt_request("request-1".into(), "Duplicate task".into())
                .is_err()
        );
        shared
            .push_prompt_request("request-2".into(), "Next task".into())
            .unwrap();
        assert_eq!(
            first.len(),
            1,
            "later requests are not part of the already-running batch"
        );
        assert_eq!(shared.take_prompt_requests()[0].0, "request-2");
        assert!(
            shared
                .push_prompt_request("bad id".into(), "Task".into())
                .is_err()
        );
        assert!(
            shared
                .push_prompt_request("request-3".into(), "/quit".into())
                .is_err()
        );
        for index in 0..64 {
            shared
                .push_prompt_request(format!("queued-{index}"), "Task".into())
                .unwrap();
        }
        assert!(
            shared
                .push_prompt_request("overflow".into(), "Task".into())
                .is_err()
        );
        let frame = crate::remote::control::ServerMsg::from_event(&super::UiEvent::PromptDone {
            request_id: "request-1".into(),
        })
        .unwrap();
        assert_eq!(serde_json::to_value(frame).unwrap()["type"], "prompt_done");
    }
    #[test]
    fn download_wire_tracks_detached_state_and_cancellation_without_starting_work() {
        use crate::downloader::{self, Phase};
        use crate::remote::control::DownloadsWire;
        let root = std::env::temp_dir().join(format!("plank-download-wire-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        assert!(DownloadsWire::capture_in(&root).state.is_none());
        assert!(!DownloadsWire::capture_in(&root).pending);
        let id = crate::manifest::EngineId::new("wire-fixture").unwrap();
        let manifest = crate::manifest::parse(&format!(
            r#"{{"version":1,"files":{{"main":{{"name":"weights.gguf","url":"https://fixture.invalid/weights","bytes":100,"sha256":"{}"}}}}}}"#,
            "a".repeat(64)
        ))
        .unwrap();
        downloader::write_job_in(&root, id, &manifest).unwrap();
        assert!(DownloadsWire::capture_in(&root).pending);
        let mut state = downloader::tests::a_state();
        state.pid = std::process::id();
        state.updated = downloader::now_epoch();
        state.done_bytes = 41;
        state.total_bytes = 100;
        downloader::write_state_in(&root, &state).unwrap();
        let snapshot = DownloadsWire::capture_in(&root);
        assert_eq!(snapshot.state.as_ref().unwrap().done_bytes, 41);
        assert!(snapshot.cancel_requested.is_none());
        assert!(downloader::cancel_command_in(&root, false).contains("partial files are kept"));
        assert_eq!(
            DownloadsWire::capture_in(&root).cancel_requested.as_deref(),
            Some("keep")
        );
        assert!(
            downloader::cancel_command_in(&root, true).contains("partial files will be deleted")
        );
        assert_eq!(
            DownloadsWire::capture_in(&root).cancel_requested.as_deref(),
            Some("delete")
        );
        state.phase = Phase::Cancelled;
        state.error = Some("keep".into());
        state.pid = 0;
        state.updated = 0;
        downloader::write_state_in(&root, &state).unwrap();
        let cancelled = DownloadsWire::capture_in(&root);
        assert!(cancelled.cancel_requested.is_none());
        assert_eq!(cancelled.state.as_ref().unwrap().partials_kept, Some(true));
        state.phase = Phase::Failed;
        state.error = Some("checksum mismatch\nretry".into());
        downloader::write_state_in(&root, &state).unwrap();
        let failed = DownloadsWire::capture_in(&root);
        assert_eq!(failed.state.as_ref().unwrap().phase, "failed");
        assert_eq!(
            failed.state.as_ref().unwrap().error.as_deref(),
            Some("checksum mismatchretry")
        );
        state.phase = Phase::Downloading;
        state.error = None;
        downloader::write_state_in(&root, &state).unwrap();
        assert!(
            DownloadsWire::capture_in(&root).state.is_none(),
            "a dead, stale helper is not live"
        );
        state.pid = std::process::id();
        state.done_bytes = u64::MAX;
        state.total_bytes = u64::MAX;
        downloader::write_state_in(&root, &state).unwrap();
        assert!(downloader::segment_text(&state).contains("100%"));
        assert!(downloader::status_report_in(&root).contains("100%"));
        let bus = super::BroadcastBus::new();
        bus.broadcast(super::UiEvent::Visible("Keep answer".into()));
        for _ in 0..100 {
            bus.broadcast(super::UiEvent::Downloads(DownloadsWire::capture_in(&root)));
        }
        bus.broadcast(super::UiEvent::Downloads(failed.clone()));
        let (tail, _) = bus.scrollback_since(None);
        assert_eq!(tail.len(), 2);
        assert!(
            matches!(&tail[1].event, super::UiEvent::Downloads(snapshot) if snapshot == &failed)
        );
        let frame =
            crate::remote::control::ServerMsg::from_event(&super::UiEvent::Downloads(failed))
                .unwrap();
        assert_eq!(serde_json::to_value(frame).unwrap()["type"], "downloads");
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn cache_wire_pages_are_bounded_and_ambiguous_fingerprints_are_read_only() {
        let nodes = (0..34)
            .map(|index| crate::kvmeta::KvMeta {
                version: crate::kvmeta::META_VERSION,
                role: crate::kvmeta::KvRole::System,
                fingerprint: format!("cache-{:02}", if index == 33 { 0 } else { index }),
                parent: None,
                model: "🦀".repeat(1000),
                created: 0,
                last_used: 0,
                hits: 3,
                bytes: 100,
                pinned: index == 0,
                label: crate::kvmeta::KvLabel::Unknown,
            })
            .collect();
        let view = crate::kvpane::KvPane::new(
            crate::kvtree::build(nodes),
            crate::kvgc::SweepPolicy::from_settings(&crate::settings::Settings::default().kvcache),
            Vec::new(),
            0,
        );
        let first = crate::remote::control::CacheWire::capture(&view, 0);
        let last = crate::remote::control::CacheWire::capture(&view, 99);
        assert_eq!((first.page, first.total, first.rows.len()), (1, 34, 32));
        assert_eq!((last.page, last.rows.len()), (2, 2));
        assert_eq!(
            first
                .rows
                .iter()
                .chain(&last.rows)
                .filter(|row| row.fingerprint.is_none())
                .count(),
            2
        );
        assert!(
            first
                .rows
                .iter()
                .chain(&last.rows)
                .all(|row| row.detail.len() <= 1024
                    && row.fingerprint.as_deref() != Some("cache-00"))
        );
        let bus = super::BroadcastBus::new();
        bus.broadcast(super::UiEvent::Visible("Keep this answer".into()));
        for _ in 0..100 {
            bus.broadcast(super::UiEvent::Cache(first.clone()));
        }
        bus.broadcast(super::UiEvent::Cache(last.clone()));
        let (tail, _) = bus.scrollback_since(None);
        assert_eq!(tail.len(), 2);
        assert!(
            matches!(&tail[0].event, super::UiEvent::Visible(text) if text == "Keep this answer")
        );
        assert!(matches!(&tail[1].event, super::UiEvent::Cache(snapshot) if snapshot.page == 2));
        let message =
            crate::remote::control::ServerMsg::from_event(&super::UiEvent::Cache(last)).unwrap();
        let json = serde_json::to_value(message).unwrap();
        assert_eq!(json["type"], "cache");
    }

    #[test]
    fn extension_replay_coalesces_by_category_and_preserves_conversation() {
        let bus = super::BroadcastBus::new();
        bus.broadcast(super::UiEvent::Visible("Keep this answer".into()));
        for index in 0..100 {
            for kind in ["plugins", "skills", "templates", "hooks", "mcp", "agents"] {
                bus.broadcast(super::UiEvent::Extensions(
                    crate::remote::control::ExtensionsWire::capture(
                        kind,
                        &format!("Update {index}"),
                        None,
                    ),
                ));
            }
        }
        let (tail, highest) = bus.scrollback_since(None);
        assert_eq!(tail.len(), 7);
        assert_eq!(highest, Some(600));
        assert!(
            matches!(&tail[0].event, super::UiEvent::Visible(text) if text == "Keep this answer")
        );
        for event in &tail[1..] {
            assert!(
                matches!(&event.event, super::UiEvent::Extensions(snapshot) if snapshot.report == "Update 99")
            );
        }
        let snapshot = crate::remote::control::ExtensionsWire::capture(
            "skills",
            &"🦀".repeat(40_000),
            Some(true),
        );
        assert!(snapshot.report.len() <= 65 * 1024);
        assert!(snapshot.report.contains("Inventory truncated"));
    }

    #[test]
    fn history_replay_keeps_the_latest_checkpoint_and_fork_pages_separately() {
        let bus = super::BroadcastBus::new();
        bus.broadcast(super::UiEvent::Visible("Keep this answer".into()));
        for _ in 0..100 {
            for kind in ["checkpoints", "forks"] {
                bus.broadcast(super::UiEvent::History(
                    crate::remote::control::HistoryWire {
                        kind: kind.into(),
                        total: 0,
                        page: 1,
                        rows: Vec::new(),
                        report: String::new(),
                    },
                ));
            }
        }
        let (tail, highest) = bus.scrollback_since(None);
        assert_eq!(tail.len(), 3);
        assert_eq!(highest, Some(200));
        assert!(
            matches!(&tail[0].event, super::UiEvent::Visible(text) if text == "Keep this answer")
        );
        assert!(
            matches!(&tail[1].event, super::UiEvent::History(snapshot) if snapshot.kind == "checkpoints")
        );
        assert!(
            matches!(&tail[2].event, super::UiEvent::History(snapshot) if snapshot.kind == "forks")
        );
    }

    #[test]
    fn gui_snapshot_updates_preserve_scrollback_and_resume_sequence_ids() {
        let bus = super::BroadcastBus::new();
        let subscriber = bus.subscribe();
        bus.broadcast(super::UiEvent::Visible("Keep this answer".into()));
        for generated in 0..5000 {
            bus.broadcast(super::UiEvent::Status(crate::status::Status {
                generated,
                ..crate::status::Status::default()
            }));
            bus.broadcast(super::UiEvent::Jobs(
                crate::remote::control::JobsWire::capture(&[]),
            ));
            bus.broadcast(super::UiEvent::Tasks(crate::tui::TaskView::default()));
            bus.broadcast(super::UiEvent::Controls(serde_json::json!({
                "settings": {"temperature": generated}
            })));
            bus.broadcast(super::UiEvent::Sessions(
                crate::remote::control::SessionsWire::capture(&[], 1),
            ));
            bus.broadcast(super::UiEvent::Config(crate::remote::control::ConfigWire {
                scope: None,
                components: Vec::new(),
                components_total: 0,
                total: 0,
                page: 1,
                fields: Vec::new(),
            }));
        }
        let (tail, highest) = bus.scrollback_since(None);
        assert_eq!(tail.len(), 7);
        assert!(
            matches!(&tail[0].event, super::UiEvent::Visible(text) if text == "Keep this answer")
        );
        assert_eq!(highest, Some(30_000));
        assert!(
            matches!(&tail[1].event, super::UiEvent::Status(status) if status.generated == 4999)
        );
        assert!(matches!(&tail[4].event, super::UiEvent::Controls(controls)
            if controls["settings"]["temperature"] == 4999));
        assert!(
            matches!(&tail[5].event, super::UiEvent::Sessions(snapshot) if snapshot.total == 0)
        );
        assert!(matches!(&tail[6].event, super::UiEvent::Config(config) if config.total == 0));
        assert_eq!(subscriber.try_iter().count(), 30_001);
        assert_eq!(bus.scrollback_since(Some(29_998)).0.len(), 2);
    }
    use super::*;

    #[test]
    fn channel_sink_forwards_render_calls_in_order() {
        let (tx, rx) = std::sync::mpsc::channel();
        let mut sink = ChannelSink(tx);
        sink.visible_text("a");
        sink.think_text("b");
        sink.tool_text("c");
        sink.error_text("d");
        let got: Vec<UiEvent> = rx.try_iter().collect();
        assert!(matches!(&got[0], UiEvent::Visible(t) if t == "a"));
        assert!(matches!(&got[1], UiEvent::Think(t) if t == "b"));
        assert!(matches!(&got[2], UiEvent::Tool(t) if t == "c"));
        assert!(matches!(&got[3], UiEvent::Error(t) if t == "d"));
    }

    #[test]
    fn sub_agent_sink_wraps_render_calls_in_sub_events() {
        let (tx, rx) = std::sync::mpsc::channel();
        let mut sink = SubAgentSink(tx);
        sink.visible_text("a");
        sink.think_text("b");
        sink.tool_text("c");
        sink.error_text("d");
        drop(sink);
        let got: Vec<UiEvent> = rx.iter().collect();
        assert!(
            matches!(got.as_slice(), [
                UiEvent::Sub(a),
                UiEvent::Sub(b),
                UiEvent::Sub(c),
                UiEvent::Sub(d),
            ] if matches!(**a, UiEvent::Visible(ref s) if s == "a")
                && matches!(**b, UiEvent::Think(ref s) if s == "b")
                && matches!(**c, UiEvent::Tool(ref s) if s == "c")
                && matches!(**d, UiEvent::Error(ref s) if s == "d")),
            "unexpected events: {got:?}"
        );
    }

    #[test]
    fn sub_agent_events_are_local_pane_only_and_never_reach_the_bus() {
        assert!(
            UiEvent::SubStart {
                label: "agent".to_string(),
                task: "t".to_string(),
            }
            .is_local_pane_only()
        );
        assert!(UiEvent::SubEnd.is_local_pane_only());
        assert!(UiEvent::Sub(Box::new(UiEvent::Visible("x".to_string()))).is_local_pane_only());
        assert!(!UiEvent::Visible("v".to_string()).is_local_pane_only());
        assert!(!UiEvent::Dim("d".to_string()).is_local_pane_only());

        // The mirror sites filter on that predicate, so a chatty sub-agent run
        // cannot evict the real transcript from the bounded scrollback ring.
        let bus = BroadcastBus::new();
        let rx = bus.subscribe();
        let stream = [
            UiEvent::Visible("before".to_string()),
            UiEvent::SubStart {
                label: "agent".to_string(),
                task: "t".to_string(),
            },
            UiEvent::Sub(Box::new(UiEvent::Visible("noise".to_string()))),
            UiEvent::SubEnd,
            UiEvent::Visible("after".to_string()),
        ];
        for ev in stream {
            if !ev.is_local_pane_only() {
                bus.broadcast(ev);
            }
        }
        let got: Vec<UiEvent> = rx.try_iter().map(|s| s.event).collect();
        assert!(
            matches!(got.as_slice(), [UiEvent::Visible(a), UiEvent::Visible(b)]
                if a == "before" && b == "after"),
            "unexpected events: {got:?}"
        );
        // And the ring a reconnecting client replays holds only those two.
        assert_eq!(bus.scrollback_since(None).0.len(), 2);
    }

    #[test]
    fn apply_ignores_sub_events_on_the_main_log() {
        // The main log must never absorb sub-agent output: the UI unwraps `Sub`
        // and applies the inner event to the sub-agent buffer instead.
        let mut log = OutputLog::new();
        let before = log.line_count();
        apply(
            &mut log,
            UiEvent::SubStart {
                label: "research".to_string(),
                task: "t".to_string(),
            },
        );
        apply(
            &mut log,
            UiEvent::Sub(Box::new(UiEvent::Visible("x".to_string()))),
        );
        apply(&mut log, UiEvent::SubEnd);
        assert_eq!(log.line_count(), before);
    }

    /// Pins the single line joining `QueuedJoined` to `commit_pending` — the
    /// worker thread's join site. Nothing else in the test suite exercises
    /// this arm: `src/ui.rs` proves the event is *sent*, `src/tui.rs` proves
    /// `commit_pending` works, but a refactor that moved this arm out from
    /// under `apply` (e.g. into the no-op tail list right below it) would
    /// leave every queued prompt stuck indented forever with the whole suite
    /// still green.
    #[test]
    fn apply_queued_joined_commits_the_pending_prompt() {
        let mut log = OutputLog::new();
        log.push_pending("fix the parser");

        apply(&mut log, UiEvent::QueuedJoined);

        // Committed into the scrollback (unindented), not still pending.
        let rows: Vec<String> = log
            .to_text()
            .lines
            .iter()
            .map(|l| {
                l.spans
                    .iter()
                    .map(|s| s.content.as_ref())
                    .collect::<String>()
            })
            .collect();
        assert_eq!(rows, ["* fix the parser"], "{rows:?}");
    }

    #[test]
    fn btw_queue_is_fifo_and_drops_oldest_beyond_cap() {
        let shared = TurnShared::default();
        for i in 0..(BTW_QUEUE_CAP + 2) {
            let dropped = shared.push_btw(format!("q{i}"));
            match i {
                i if i < BTW_QUEUE_CAP => assert!(dropped.is_none()),
                i if i == BTW_QUEUE_CAP => assert_eq!(dropped.as_deref(), Some("q0")),
                _ => assert_eq!(dropped.as_deref(), Some("q1")),
            }
        }
        assert_eq!(shared.pop_btw().as_deref(), Some("q2"));
        assert_eq!(shared.take_btw().len(), BTW_QUEUE_CAP - 1);
        shared.push_btw("late".to_owned());
        assert_eq!(shared.clear_btw(), 1);
        assert!(shared.pop_btw().is_none());
    }

    #[test]
    fn bus_fans_out_to_multiple_subscribers() {
        let bus = BroadcastBus::new();
        let a = bus.subscribe();
        let b = bus.subscribe();
        bus.broadcast(UiEvent::Visible("hi".into()));
        for rx in [&a, &b] {
            let got = rx.try_recv().expect("subscriber received");
            assert_eq!(got.id, 0);
            assert!(matches!(got.event, UiEvent::Visible(ref t) if t == "hi"));
        }
        // A dropped subscriber is pruned on the next broadcast and does not
        // stall the survivor.
        drop(b);
        bus.broadcast(UiEvent::Visible("again".into()));
        assert_eq!(bus.subscriber_count(), 1);
        let got = a.try_recv().expect("survivor still receives");
        assert_eq!(got.id, 1);
    }

    #[test]
    fn bus_scrollback_replays_on_late_join() {
        let bus = BroadcastBus::new();
        bus.broadcast(UiEvent::Visible("one".into()));
        bus.broadcast(UiEvent::Visible("two".into()));
        // Late joiner replays the full tail...
        let (tail, highest) = bus.scrollback_since(None);
        assert_eq!(tail.len(), 2);
        assert_eq!(highest, Some(1));
        // ...and a resume_from only sees newer events.
        let (since, _) = bus.scrollback_since(Some(0));
        assert_eq!(since.len(), 1);
        assert_eq!(since[0].id, 1);
        // Empty bus reports no highest id.
        assert_eq!(BroadcastBus::new().scrollback_since(None).1, None);
    }

    #[test]
    fn bus_scrollback_is_bounded() {
        let bus = BroadcastBus::new();
        for _ in 0..(SCROLLBACK_CAP + 10) {
            bus.broadcast(UiEvent::EndLine);
        }
        let (tail, highest) = bus.scrollback_since(None);
        assert_eq!(tail.len(), SCROLLBACK_CAP);
        assert_eq!(highest, Some((SCROLLBACK_CAP + 10 - 1) as u64));
        // Oldest ids were evicted; resume from a rolled-past id yields the tail.
        assert_eq!(tail.first().unwrap().id, 10);
    }

    /// A session reset drops the scrollback with it. Otherwise a client that
    /// attaches after `/clear` is replayed the transcript the operator just
    /// cleared — the reset would only be visible to whoever was already
    /// connected to see the frame go past.
    #[test]
    fn session_reset_drops_the_scrollback_it_invalidates() {
        let bus = BroadcastBus::new();
        bus.broadcast(UiEvent::Visible("stale answer".into()));
        bus.broadcast(UiEvent::UserEcho("stale prompt".into()));
        let before = bus.scrollback_since(None).0.len();
        assert_eq!(before, 2);

        bus.broadcast(UiEvent::SessionReset);
        let (tail, highest) = bus.scrollback_since(None);
        assert_eq!(tail.len(), 1, "only the reset survives: {tail:?}");
        assert!(matches!(tail[0].event, UiEvent::SessionReset));
        // Ids stay monotonic across the reset, so an already-connected client's
        // `resume_from` is still meaningful and does not replay the reset twice.
        assert_eq!(highest, Some(2));
        assert_eq!(tail[0].id, 2);

        // Output after the reset accumulates normally again.
        bus.broadcast(UiEvent::Visible("fresh".into()));
        assert_eq!(bus.scrollback_since(None).0.len(), 2);
    }

    #[test]
    fn queued_lines_round_trip() {
        let shared = TurnShared::default();
        shared.push_queued("one".into());
        shared.push_queued("two".into());
        shared.push_command("/usage".into());
        assert_eq!(shared.take_queued(), vec!["one", "two"]);
        assert!(shared.has_commands());
        assert_eq!(shared.take_commands(), vec!["/usage"]);
        assert!(!shared.has_commands());
        assert!(!shared.has_queued());
        assert!(shared.take_queued().is_empty());

        // Blank lines are skipped when the queue runs, so they are not
        // reason enough to take a screensaver down.
        shared.push_queued("  \n".into());
        assert!(!shared.has_queued());
        shared.push_queued("go".into());
        assert!(shared.has_queued());
    }
}
