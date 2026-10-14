//! Owned console input and shared GUI/remote dialog routing.

// Rust guideline compliant 2026-02-21
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

use crate::{
    remote::{
        control::RemoteState,
        editors::{EditReply, Editors},
        questions::Questions,
    },
    tools::ask::{AskOutcome, AskRequest},
    worker::UiEvent,
};
use serde_json::Value;

// Match the daemon's escaped-document frame bound; cap preserved input separately.
const FRAME_BYTES: usize = 2 * 1024 * 1024;
pub(crate) const BATCH_BYTES: usize = 4 * 1024 * 1024;
const PENDING_BYTES: usize = BATCH_BYTES;
const PENDING_COUNT: usize = 64;
pub(crate) mod output;

#[derive(Debug, Default)]
struct Inbox {
    prompts: VecDeque<String>,
    bytes: usize,
    fragment: String,
    eof: bool,
    dialogs: VecDeque<String>,
    utf8: Vec<u8>,
}

impl Inbox {
    fn feed(&mut self, bytes: &[u8], id: Option<&str>) -> Result<Vec<Value>, String> {
        if self.utf8.len().saturating_add(bytes.len()) > BATCH_BYTES {
            return Err("Console input batch exceeds 4 MiB".into());
        }
        self.utf8.extend_from_slice(bytes);
        if let Err(error) = std::str::from_utf8(&self.utf8) {
            if error.error_len().is_none() && !self.eof {
                return Ok(Vec::new());
            }
            return Err("Console input contains invalid or incomplete UTF-8".into());
        }
        let batch =
            String::from_utf8(std::mem::take(&mut self.utf8)).map_err(|error| error.to_string())?;
        self.route(&batch, id)
    }
    fn push(&mut self, prompt: String) -> Result<(), String> {
        if prompt.trim().is_empty() {
            return Ok(());
        }
        if self.prompts.len() >= PENDING_COUNT
            || self.bytes.saturating_add(prompt.len()) > PENDING_BYTES
        {
            return Err("Console pending input limit exceeded".into());
        }
        self.bytes += prompt.len();
        self.prompts.push_back(prompt);
        Ok(())
    }

    fn route(&mut self, batch: &str, id: Option<&str>) -> Result<Vec<Value>, String> {
        if batch.len() > PENDING_BYTES {
            return Err("Console input batch exceeds 4 MiB".into());
        }
        let mut replies = Vec::new();
        let mut prompt = String::new();
        for line in batch.split_inclusive('\n') {
            if self.fragment.is_empty() && !line.trim_start().starts_with('{') {
                prompt.push_str(line);
                continue;
            }
            if self.fragment.len().saturating_add(line.len()) > FRAME_BYTES {
                return Err("Console input frame exceeds 2 MiB".into());
            }
            self.fragment.push_str(line);
            if !line.ends_with('\n') && !self.eof {
                continue;
            }
            let frame = std::mem::take(&mut self.fragment);
            self.frame(&frame, id, &mut prompt, &mut replies)?;
        }
        if self.eof && !self.fragment.is_empty() {
            let frame = std::mem::take(&mut self.fragment);
            self.frame(&frame, id, &mut prompt, &mut replies)?;
        }
        self.push(prompt)?;
        Ok(replies)
    }

    fn frame(
        &mut self,
        frame: &str,
        id: Option<&str>,
        prompt: &mut String,
        replies: &mut Vec<Value>,
    ) -> Result<(), String> {
        let value = serde_json::from_str::<Value>(frame).ok();
        let known_id = id
            .into_iter()
            .chain(self.dialogs.iter().map(String::as_str))
            .any(|id| frame.contains(&format!("\"{id}\"")));
        let malformed_reply = value.is_none()
            && (frame.contains("\"dialog_reply\"")
                || (known_id
                    && frame.contains("\"id\"")
                    && (frame.contains("\"answer\"") || frame.contains("\"text\""))));
        let reply = malformed_reply
            || value.as_ref().is_some_and(|value| {
                value["type"] == "dialog_reply"
                    || (value["id"].as_str().is_some_and(|received| {
                        Some(received) == id || self.dialogs.iter().any(|known| known == received)
                    }) && (value.get("answer").is_some() || value.get("text").is_some()))
            });
        if reply {
            self.push(std::mem::take(prompt))?;
            if let Some(value) = value.filter(|value| id.is_some() && value["id"].as_str() == id) {
                replies.push(value);
            }
        } else {
            prompt.push_str(frame);
        }
        Ok(())
    }

    fn pop(&mut self) -> Option<String> {
        let prompt = self.prompts.pop_front()?;
        self.bytes -= prompt.len();
        Some(prompt)
    }
}

/// Owns pending stdin and the optional bridge for one console agent.
#[derive(Default)]
pub(crate) struct ConsoleInput {
    inbox: Mutex<Inbox>,
    remote: Mutex<Option<Arc<RemoteState>>>,
    questions: Questions,
    editors: Editors,
    subagents: Mutex<crate::tui::SubPane>,
    saved_page: std::sync::atomic::AtomicUsize,
    // Zero means unopened. Fork and checkpoint pages are remembered independently.
    history_pages: [std::sync::atomic::AtomicUsize; 2],
    next_export: std::sync::atomic::AtomicU64,
    aside: std::sync::atomic::AtomicBool,
}

/// Restores the owning channel's previous aside scope on every return path.
#[derive(Debug)]
pub(crate) struct AsideGuard {
    channel: Arc<ConsoleInput>,
    previous: bool,
}

impl Drop for AsideGuard {
    fn drop(&mut self) {
        self.channel
            .aside
            .store(self.previous, std::sync::atomic::Ordering::Relaxed);
    }
}

/// One pass's live counters, addressed to a stable delegated-run identity.
pub(crate) struct SubagentProgress {
    channel: Arc<ConsoleInput>,
    run_id: u64,
    remote: Option<Arc<RemoteState>>,
    gui: bool,
    live: crate::tui::LiveTokens,
    last: Option<std::time::Instant>,
}

impl std::fmt::Debug for SubagentProgress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SubagentProgress")
            .field("run_id", &self.run_id)
            .finish_non_exhaustive()
    }
}

impl SubagentProgress {
    pub(crate) fn output(&self) -> output::SubagentOutput {
        output::SubagentOutput::new(self)
    }
    pub(crate) fn event(&mut self, event: &crate::engine::EngineEvent) {
        let prefilling = self.live.prefilling;
        match event {
            crate::engine::EngineEvent::Text(_) => {
                self.live.prefilling = false;
                self.live.generated = self.live.generated.saturating_add(1);
            }
            crate::engine::EngineEvent::Prefill(progress) => {
                self.live.prefilling = !progress.is_complete();
                self.live.prefill_done = progress.done;
                self.live.prefill_total = progress.total;
            }
            crate::engine::EngineEvent::Notice(_) | crate::engine::EngineEvent::Spec(_) => return,
        }
        // Publish phase changes immediately, otherwise at most once per second.
        if prefilling == self.live.prefilling
            && self
                .last
                .is_some_and(|last| last.elapsed() < std::time::Duration::from_secs(1))
        {
            return;
        }
        self.last = Some(std::time::Instant::now());
        let snapshot = {
            let mut pane = self
                .channel
                .subagents
                .lock()
                .expect("subagent roster poisoned");
            let Some(run) = pane
                .runs
                .iter_mut()
                .find(|run| run.id == self.run_id && run.running)
            else {
                return;
            };
            run.live = Some(self.live);
            crate::remote::control::SubagentWire::capture(&pane, crate::tui::roster_clock_ms())
        };
        if self.gui {
            eprintln!(
                "+PLANK_TELEMETRY {}",
                serde_json::json!({"type":"metrics", "snapshot":snapshot})
            );
        }
        if let Some(remote) = &self.remote {
            remote.bus.broadcast(UiEvent::Metrics(snapshot));
        }
    }
}

pub(crate) fn note_subagent_progress(
    progress: &mut Option<SubagentProgress>,
    event: &crate::engine::EngineEvent,
) {
    if let Some(progress) = progress {
        progress.event(event);
    }
}

impl std::fmt::Debug for ConsoleInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Do not expose remote credentials or retained prompt contents.
        let inbox = self.inbox.lock().expect("console inbox poisoned");
        f.debug_struct("ConsoleInput")
            .field("pending", &inbox.prompts.len())
            .field("eof", &inbox.eof)
            .finish_non_exhaustive()
    }
}

impl ConsoleInput {
    pub(crate) fn aside_scope(self: &Arc<Self>) -> AsideGuard {
        let previous = self.aside.swap(true, std::sync::atomic::Ordering::Relaxed);
        AsideGuard {
            channel: Arc::clone(self),
            previous,
        }
    }

    pub(crate) fn is_aside(&self) -> bool {
        self.aside.load(std::sync::atomic::Ordering::Relaxed)
    }
    /// Creates an export identity owned by this agent channel across context resets.
    pub(crate) fn new_export_id(&self) -> String {
        let id = self
            .next_export
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        format!("export-{id}")
    }

    /// Remembers only a successfully published saved-session page.
    pub(crate) fn set_saved_page(&self, page: usize) {
        self.saved_page
            .store(page, std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn saved_page(&self) -> Option<usize> {
        let page = self.saved_page.load(std::sync::atomic::Ordering::Relaxed);
        (page > 0).then_some(page)
    }

    /// Remembers the clamped page published for an opened history category.
    pub(crate) fn set_history_page(&self, checkpoints: bool, page: usize) {
        self.history_pages[usize::from(checkpoints)]
            .store(page, std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn history_page(&self, checkpoints: bool) -> Option<usize> {
        let page =
            self.history_pages[usize::from(checkpoints)].load(std::sync::atomic::Ordering::Relaxed);
        (page > 0).then_some(page)
    }

    /// Binds a pass to its active run; late callbacks cannot target a later run.
    pub(crate) fn progress(
        self: &Arc<Self>,
        label: &str,
        gui: bool,
        remote: Option<Arc<RemoteState>>,
    ) -> Option<SubagentProgress> {
        if !gui && remote.is_none() {
            return None;
        }
        let run_id = self
            .subagents
            .lock()
            .expect("subagent roster poisoned")
            .runs
            .iter()
            .find(|run| run.running && run.label == label)?
            .id;
        Some(SubagentProgress {
            channel: Arc::clone(self),
            run_id,
            remote,
            gui,
            live: crate::tui::LiveTokens::default(),
            last: None,
        })
    }

    /// Updates delegated-run metadata without retaining its streamed output.
    pub(crate) fn subagent_event(
        &self,
        event: &crate::worker::UiEvent,
    ) -> Option<crate::remote::control::MetricsWire> {
        let mut pane = self.subagents.lock().expect("subagent roster poisoned");
        let now = crate::tui::roster_clock_ms();
        match event {
            crate::worker::UiEvent::SubStart { label, task } => {
                pane.begin(label.clone(), task, now);
            }
            crate::worker::UiEvent::SubEnd => pane.end(now),
            crate::worker::UiEvent::SubTokens {
                label,
                prefill,
                generated,
            } => pane.add_tokens(label.as_deref(), *prefill, *generated),
            _ => return None,
        }
        Some(crate::remote::control::SubagentWire::capture(&pane, now))
    }

    /// Stores a bounded final report on its latest named run.
    pub(crate) fn subagent_report(
        &self,
        label: &str,
        report: &str,
    ) -> Option<crate::remote::control::MetricsWire> {
        let mut pane = self.subagents.lock().expect("subagent roster poisoned");
        let run = pane.runs.iter_mut().rev().find(|run| run.label == label)?;
        run.report = Some(crate::remote::control::bounded_subagent_report(report));
        Some(crate::remote::control::SubagentWire::capture(
            &pane,
            crate::tui::roster_clock_ms(),
        ))
    }

    pub(crate) fn reset_subagents(&self) {
        self.subagents
            .lock()
            .expect("subagent roster poisoned")
            .reset();
    }

    fn register(&self, id: &str) {
        let mut inbox = self.inbox.lock().expect("console inbox poisoned");
        // Bound late-reply recognition across successive dialogs without globals.
        while inbox.dialogs.len() >= 128 {
            inbox.dialogs.pop_front();
        }
        inbox.dialogs.push_back(id.into());
    }
    pub(crate) fn attach(&self, remote: Option<Arc<RemoteState>>) {
        *self.remote.lock().expect("console bridge poisoned") = remote;
    }

    /// Reads cancellation without consuming the owning dispatcher's stop.
    pub(crate) fn interrupted(&self) -> bool {
        crate::interrupt::pending()
            || self.remote().is_some_and(|remote| {
                remote
                    .shared
                    .interrupt
                    .load(std::sync::atomic::Ordering::Relaxed)
                    || remote.shared.prompt_batch_cancelled()
            })
    }

    fn remote(&self) -> Option<Arc<RemoteState>> {
        self.remote.lock().expect("console bridge poisoned").clone()
    }

    pub(crate) fn eof(&self) -> bool {
        self.inbox.lock().expect("console inbox poisoned").eof
    }

    pub(crate) fn has_pending(&self) -> bool {
        let inbox = self.inbox.lock().expect("console inbox poisoned");
        !inbox.prompts.is_empty() || !inbox.fragment.is_empty() || !inbox.utf8.is_empty()
    }

    fn read(
        &self,
        id: Option<&str>,
        read: &mut impl FnMut(&mut bool) -> Result<Option<Vec<u8>>, String>,
    ) -> Result<Vec<Value>, String> {
        let mut eof = false;
        let batch = read(&mut eof)?;
        let mut inbox = self.inbox.lock().expect("console inbox poisoned");
        inbox.eof |= eof;
        inbox.feed(batch.as_deref().unwrap_or(&[]), id)
    }

    /// Returns preserved input before reading; stale replies never become prompts.
    pub(crate) fn prompt(
        &self,
        mut read: impl FnMut(&mut bool) -> Result<Option<Vec<u8>>, String>,
    ) -> Result<Option<String>, String> {
        if let Some(prompt) = self.inbox.lock().expect("console inbox poisoned").pop() {
            return Ok(Some(prompt));
        }
        if !self.eof() {
            self.read(None, &mut read)?;
        }
        Ok(self.inbox.lock().expect("console inbox poisoned").pop())
    }

    /// Opens one question identity for local and authenticated remote replies.
    pub(crate) fn question(
        &self,
        id: &str,
        request: &AskRequest,
        gui: bool,
        mut read: impl FnMut(&mut bool) -> Result<Option<Vec<u8>>, String>,
    ) -> Result<AskOutcome, String> {
        let remote = self.remote();
        let questions = remote
            .as_ref()
            .map_or(&self.questions, |remote| &remote.shared.questions);
        self.register(id);
        questions.open(id, request.clone());
        let dialog = Dialog {
            id,
            slot: Slot::Question(questions),
            remote: remote.as_deref(),
            gui,
        };
        let options = crate::tools::ask::rows(request)
            .iter()
            .map(
                |option| serde_json::json!({"label":option.label,"description":option.description}),
            )
            .collect::<Vec<_>>();
        dialog.publish(serde_json::json!({"id":id,"header":request.header,"question":request.question,"multi":request.multi,"options":options}));
        Ok(match self.wait(&dialog, &mut read)? {
            Wait::Reply(Reply::Question(answer)) => answer,
            Wait::Interrupted => AskOutcome::Interrupted,
            _ => match questions.close(id) {
                Some(answer) => answer,
                None => AskOutcome::Declined,
            },
        })
    }

    /// Opens a bounded editor; only explicit Save or Cancel resolves it.
    pub(crate) fn edit(
        &self,
        id: &str,
        title: &str,
        text: &str,
        gui: bool,
        mut read: impl FnMut(&mut bool) -> Result<Option<Vec<u8>>, String>,
    ) -> Result<Option<String>, String> {
        if text.len() > crate::remote::editors::DOCUMENT_BYTES {
            return Err("Document exceeds the 256 KiB editing limit".into());
        }
        let remote = self.remote();
        let editors = remote
            .as_ref()
            .map_or(&self.editors, |remote| &remote.shared.editors);
        self.register(id);
        editors.open(id);
        let dialog = Dialog {
            id,
            slot: Slot::Editor(editors),
            remote: remote.as_deref(),
            gui,
        };
        dialog.publish(serde_json::json!({"id":id,"title":title,"text":text}));
        Ok(match self.wait(&dialog, &mut read)? {
            Wait::Reply(Reply::Editor(reply)) => reply.into_text(),
            Wait::Closed => editors.finish(id).and_then(EditReply::into_text),
            _ => None,
        })
    }

    fn wait(
        &self,
        dialog: &Dialog<'_>,
        read: &mut impl FnMut(&mut bool) -> Result<Option<Vec<u8>>, String>,
    ) -> Result<Wait, String> {
        loop {
            if crate::interrupt::pending()
                || dialog.remote.is_some_and(|remote| {
                    remote
                        .shared
                        .interrupt
                        .load(std::sync::atomic::Ordering::Relaxed)
                        || remote.shared.prompt_batch_cancelled()
                })
            {
                return Ok(Wait::Interrupted);
            }
            if let Some(reply) = dialog.slot.take(dialog.id) {
                return Ok(Wait::Reply(reply));
            }
            if self.eof() {
                return Ok(Wait::Closed);
            }
            for reply in self.read(Some(dialog.id), read)? {
                if !dialog.slot.accept(dialog.id, &reply) {
                    eprintln!("Dialog reply rejected; use the current request's explicit choices");
                }
            }
        }
    }
}

enum Reply {
    Question(AskOutcome),
    Editor(EditReply),
}
enum Wait {
    Reply(Reply),
    Interrupted,
    Closed,
}
enum Slot<'a> {
    Question(&'a Questions),
    Editor(&'a Editors),
}

impl Slot<'_> {
    fn accept(&self, id: &str, value: &Value) -> bool {
        match self {
            Self::Question(questions) => value["answer"]
                .as_str()
                .is_some_and(|answer| questions.answer(id, answer)),
            Self::Editor(editors) => value
                .get("text")
                .is_some_and(|text| editors.reply(id, text)),
        }
    }
    fn take(&self, id: &str) -> Option<Reply> {
        match self {
            Self::Question(questions) => questions.take_answer(id).map(Reply::Question),
            Self::Editor(editors) => editors.take_reply(id).map(Reply::Editor),
        }
    }
    fn close(&self, id: &str) {
        match self {
            Self::Question(questions) => {
                questions.close(id);
            }
            Self::Editor(editors) => editors.close(id),
        }
    }
}

struct Dialog<'a> {
    id: &'a str,
    slot: Slot<'a>,
    remote: Option<&'a RemoteState>,
    gui: bool,
}
impl Dialog<'_> {
    fn publish(&self, value: Value) {
        let marker = match self.slot {
            Slot::Question(_) => "QUESTION",
            Slot::Editor(_) => "EDITOR",
        };
        if self.gui {
            eprintln!("+PLANK_{marker} {value}");
        }
        if let Some(remote) = self.remote {
            remote.bus.broadcast(match self.slot {
                Slot::Question(_) => UiEvent::Question(value),
                Slot::Editor(_) => UiEvent::Editor(value),
            });
        }
    }
}
impl Drop for Dialog<'_> {
    fn drop(&mut self) {
        self.slot.close(self.id);
        let (marker, event) = match self.slot {
            Slot::Question(_) => ("QUESTION", UiEvent::QuestionDone { id: self.id.into() }),
            Slot::Editor(_) => ("EDITOR", UiEvent::EditorDone { id: self.id.into() }),
        };
        if self.gui {
            eprintln!("+PLANK_{marker}_DONE {}", serde_json::json!({"id":self.id}));
        }
        if let Some(remote) = self.remote {
            remote.bus.broadcast(event);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::ask::AskOption;

    #[test]
    fn console_subagent_progress_is_coalesced_and_cannot_update_a_reused_name() {
        use crate::{
            engine::{EngineEvent, PrefillProgress},
            remote::control::MetricsWire,
        };
        let channel = Arc::new(ConsoleInput::default());
        let remote = crate::remote::control::RemoteState::test_bridge();
        let receiver = remote.bus.subscribe();
        let start = UiEvent::SubStart {
            label: "review".into(),
            task: "Inspect change".into(),
        };
        channel.subagent_event(&start);
        let mut progress = channel
            .progress("review", false, Some(Arc::clone(&remote)))
            .unwrap();
        progress.event(&EngineEvent::Prefill(PrefillProgress {
            done: 1,
            total: 10,
            tps: 10.0,
        }));
        progress.event(&EngineEvent::Prefill(PrefillProgress {
            done: 2,
            total: 10,
            tps: 10.0,
        }));
        progress.event(&EngineEvent::Text("first".into()));
        progress.event(&EngineEvent::Text("second".into()));
        assert_eq!(
            receiver.try_iter().count(),
            2,
            "phase transitions publish; same-phase callbacks coalesce"
        );
        progress.last = Some(std::time::Instant::now() - std::time::Duration::from_secs(2));
        progress.event(&EngineEvent::Text("third".into()));
        assert!(receiver.try_iter().any(|event| matches!(event.event,
            UiEvent::Metrics(MetricsWire::Subagents { rows, .. })
                if rows[0].live.as_ref().is_some_and(|live| !live.prefilling && live.generated == 3)
                    && rows[0].generated == 0)));
        channel.subagent_event(&UiEvent::SubTokens {
            label: None,
            prefill: 100,
            generated: 20,
        });
        channel.subagent_event(&UiEvent::SubEnd);
        channel.subagent_event(&start);
        progress.last = None;
        progress.event(&EngineEvent::Text("late".into()));
        assert!(receiver.try_recv().is_err());
        let Some(MetricsWire::Subagents { rows, .. }) =
            channel.subagent_event(&UiEvent::SubTokens {
                label: None,
                prefill: 0,
                generated: 0,
            })
        else {
            panic!("snapshot");
        };
        assert!(rows[0].live.is_none());
        assert_eq!(rows[1].generated, 20);
        assert!(rows[1].live.is_none());
    }

    #[test]
    fn console_subagent_roster_keeps_fanout_labels_and_resets_with_context() {
        use crate::{remote::control::MetricsWire, worker::UiEvent};
        let channel = ConsoleInput::default();
        for label in ["first", "second"] {
            channel.subagent_event(&UiEvent::SubStart {
                label: label.into(),
                task: "Do work\nwith context".into(),
            });
        }
        channel.subagent_event(&UiEvent::SubTokens {
            label: Some("first".into()),
            prefill: 100,
            generated: 20,
        });
        channel.subagent_event(&UiEvent::SubStart {
            label: "first".into(),
            task: String::new(),
        });
        let Some(MetricsWire::Subagents {
            rows,
            total,
            running,
        }) = channel.subagent_event(&UiEvent::SubEnd)
        else {
            panic!("roster snapshot");
        };
        assert_eq!((total, running), (2, 1));
        assert_eq!(rows[0].label, "second");
        assert_eq!((rows[1].prefill, rows[1].generated), (100, 20));
        assert!(!rows[1].task.contains('\n'));
        channel.reset_subagents();
        let Some(MetricsWire::Subagents { total, running, .. }) =
            channel.subagent_event(&UiEvent::SubStart {
                label: "fresh".into(),
                task: "new context".into(),
            })
        else {
            panic!("new snapshot");
        };
        assert_eq!((total, running), (1, 1));
    }

    #[test]
    fn console_subagent_roster_bounds_wire_rows_and_unicode_labels() {
        use crate::{remote::control::MetricsWire, worker::UiEvent};
        let channel = ConsoleInput::default();
        let mut snapshot = None;
        for index in 0..80 {
            snapshot = channel.subagent_event(&UiEvent::SubStart {
                label: format!("{index}{}", "🦀".repeat(100)),
                task: "é".repeat(1000),
            });
        }
        let Some(MetricsWire::Subagents {
            rows,
            total,
            running,
        }) = snapshot
        else {
            panic!("snapshot");
        };
        assert_eq!((rows.len(), total, running), (64, 80, 80));
        assert!(
            rows.iter()
                .all(|row| row.label.len() <= 128 && row.task.len() <= 1024)
        );
    }

    #[test]
    fn console_subagent_reports_keep_identity_on_repeated_names_and_clip_unicode() {
        use crate::{remote::control::MetricsWire, worker::UiEvent};
        let channel = ConsoleInput::default();
        let start = UiEvent::SubStart {
            label: "review".into(),
            task: "Inspect change".into(),
        };
        channel.subagent_event(&start);
        channel.subagent_event(&UiEvent::SubEnd);
        let Some(MetricsWire::Subagents { rows, .. }) =
            channel.subagent_report("review", "first report")
        else {
            panic!("first report");
        };
        let first = rows[0].id;
        channel.subagent_event(&start);
        channel.subagent_event(&UiEvent::SubEnd);
        let Some(MetricsWire::Subagents { rows, .. }) =
            channel.subagent_report("review", &format!("\u{1b}{}", "🦀".repeat(5000)))
        else {
            panic!("second report");
        };
        assert_ne!(rows[1].id, first);
        assert_eq!(rows[0].report.as_deref(), Some("first report"));
        let report = rows[1].report.as_ref().unwrap();
        assert!(report.len() <= 4096);
        assert!(report.contains("Report truncated"));
        assert!(!report.contains('\u{1b}'));
        let second = rows[1].id;
        channel.reset_subagents();
        let Some(MetricsWire::Subagents { rows, .. }) = channel.subagent_event(&start) else {
            panic!("fresh run");
        };
        assert!(rows[0].id > second);
        assert!(rows[0].report.is_none());
        assert!(channel.subagent_report("missing", "late report").is_none());
    }

    fn request() -> AskRequest {
        AskRequest {
            question: "Choose?".into(),
            header: "Choice".into(),
            multi: false,
            allow_chat: false,
            options: vec![
                AskOption {
                    label: "One".into(),
                    description: String::new(),
                },
                AskOption {
                    label: "Two".into(),
                    description: String::new(),
                },
            ],
        }
    }

    fn wire(id: &str, key: &str, value: Value) -> Vec<u8> {
        let mut object = serde_json::json!({"type":"dialog_reply","id":id});
        object[key] = value;
        let mut bytes = serde_json::to_vec(&object).unwrap();
        bytes.push(b'\n');
        bytes
    }

    #[test]
    fn console_inbox_preserves_json_prompts_and_discards_late_or_malformed_replies() {
        let mut inbox = Inbox::default();
        let json = "{\"id\":\"object-17\",\"text\":\"ordinary JSON prompt\"}\n";
        inbox.feed(json.as_bytes(), None).unwrap();
        assert_eq!(inbox.pop().as_deref(), Some(json));
        inbox
            .feed(&wire("old", "answer", serde_json::json!("1")), None)
            .unwrap();
        inbox
            .feed(b"{\"type\":\"dialog_reply\", invalid}\n", None)
            .unwrap();
        assert!(inbox.pop().is_none());
        let mut bytes = b"before\n".to_vec();
        bytes.extend(wire("q1", "answer", serde_json::json!("2")));
        bytes.extend(b"after\n");
        let replies = inbox.feed(&bytes, Some("q1")).unwrap();
        assert_eq!(replies[0]["answer"], "2");
        assert_eq!(inbox.pop().as_deref(), Some("before\n"));
        assert_eq!(inbox.pop().as_deref(), Some("after\n"));
    }

    #[test]
    fn console_editor_retains_utf8_fragments_across_reader_timeouts() {
        let input = ConsoleInput::default();
        let text = "🦀 café\n日本語";
        let bytes = wire("e1", "text", serde_json::json!(text));
        let split = bytes
            .windows(4)
            .position(|bytes| bytes == "🦀".as_bytes())
            .unwrap()
            + 1;
        let mut reads = VecDeque::from([
            Some(bytes[..split].to_vec()),
            None,
            Some(bytes[split..].to_vec()),
        ]);
        let edited = input
            .edit("e1", "Document", "initial", false, |_| {
                Ok(reads
                    .pop_front()
                    .expect("editor must finish on its complete reply"))
            })
            .unwrap();
        assert_eq!(edited.as_deref(), Some(text));
        assert!(!input.has_pending());
        assert!(!input.editors.reply("e1", &serde_json::json!("late")));
    }

    #[test]
    fn console_inbox_eof_preserves_json_bytes_and_rejects_truncated_utf8() {
        let mut inbox = Inbox::default();
        let json = "{\"id\":\"data\",\"text\":\"value\"}";
        inbox.feed(json.as_bytes(), None).unwrap();
        assert!(inbox.pop().is_none());
        inbox.eof = true;
        inbox.feed(&[], None).unwrap();
        assert_eq!(inbox.pop().as_deref(), Some(json));
        let mut truncated = Inbox::default();
        truncated.feed(&[0xe2], None).unwrap();
        truncated.eof = true;
        assert!(truncated.feed(&[], None).is_err());
    }

    #[test]
    fn console_question_shares_identity_and_honors_the_first_accepted_reply() {
        let remote = RemoteState::test_bridge();
        let receiver = remote.bus.subscribe();
        let input = ConsoleInput::default();
        input.attach(Some(Arc::clone(&remote)));
        let answer = input
            .question("q1", &request(), false, |_| {
                let UiEvent::Question(value) = receiver.try_recv().unwrap().event else {
                    panic!("question event");
                };
                assert_eq!(value["id"], "q1");
                assert_eq!(value["options"][0]["label"], "One");
                assert!(remote.shared.questions.answer("q1", "1"));
                let mut bytes = b"A later local prompt\n".to_vec();
                bytes.extend(wire("q1", "answer", serde_json::json!("2")));
                Ok(Some(bytes))
            })
            .unwrap();
        assert_eq!(answer, AskOutcome::Answered(vec!["One".into()]));
        assert!(
            matches!(receiver.try_recv().unwrap().event, UiEvent::QuestionDone {id} if id == "q1")
        );
        assert_eq!(
            input
                .prompt(|_| panic!("preserved input must not read stdin"))
                .unwrap()
                .as_deref(),
            Some("A later local prompt\n")
        );
        assert!(!remote.shared.questions.answer("q1", "1"));
    }

    #[test]
    fn console_question_rejects_invalid_answers_and_accepts_explicit_decline() {
        let input = ConsoleInput::default();
        let mut reads = VecDeque::from([
            wire("old", "answer", serde_json::json!("1")),
            wire("q1", "answer", serde_json::json!("99")),
            wire("q1", "answer", serde_json::json!("0")),
        ]);
        let outcome = input
            .question("q1", &request(), false, |_| {
                Ok(Some(reads.pop_front().unwrap()))
            })
            .unwrap();
        assert_eq!(outcome, AskOutcome::Declined);
        assert!(!input.has_pending());
        assert!(!input.questions.answer("q1", "1"));
    }

    #[test]
    fn console_dialogs_close_on_reader_error_and_remote_interruption() {
        let remote = RemoteState::test_bridge();
        let input = ConsoleInput::default();
        input.attach(Some(Arc::clone(&remote)));
        assert!(
            input
                .edit("e1", "Document", "initial", false, |_| Err(
                    "reader failed".into()
                ))
                .is_err()
        );
        assert!(
            !remote
                .shared
                .editors
                .reply("e1", &serde_json::json!("late"))
        );
        let outcome = input
            .question("q1", &request(), false, |_| {
                remote
                    .shared
                    .interrupt
                    .store(true, std::sync::atomic::Ordering::Relaxed);
                Ok(None)
            })
            .unwrap();
        assert_eq!(outcome, AskOutcome::Interrupted);
        assert!(!remote.shared.questions.answer("q1", "1"));
        let (events, _) = remote.bus.scrollback_since(None);
        assert!(
            events
                .iter()
                .any(|event| matches!(&event.event, UiEvent::EditorDone {id} if id == "e1"))
        );
        assert!(
            events
                .iter()
                .any(|event| matches!(&event.event, UiEvent::QuestionDone {id} if id == "q1"))
        );
    }

    #[test]
    fn console_remote_editor_requires_explicit_save_or_cancel() {
        let remote = RemoteState::test_bridge();
        let input = ConsoleInput::default();
        input.attach(Some(Arc::clone(&remote)));
        let edited = input
            .edit("e1", "Document", "initial", false, |_| {
                assert!(
                    remote
                        .shared
                        .editors
                        .reply("e1", &serde_json::json!("remote edit 🦀"))
                );
                Ok(None)
            })
            .unwrap();
        assert_eq!(edited.as_deref(), Some("remote edit 🦀"));
        let cancelled = input
            .edit("e2", "Document", "initial", false, |_| {
                Ok(Some(wire("e2", "text", Value::Null)))
            })
            .unwrap();
        assert!(cancelled.is_none());
        assert!(
            !remote
                .shared
                .editors
                .reply("e2", &serde_json::json!("late"))
        );
        assert!(
            input
                .edit(
                    "large",
                    "Document",
                    &"x".repeat(crate::remote::editors::DOCUMENT_BYTES + 1),
                    false,
                    |_| panic!("oversized documents must not open")
                )
                .is_err()
        );
    }

    #[test]
    fn console_inbox_bounds_pending_input_and_debug_omits_content_and_credentials() {
        let input = ConsoleInput::default();
        input.attach(Some(RemoteState::test_bridge()));
        let mut inbox = input.inbox.lock().unwrap();
        for _ in 0..PENDING_COUNT {
            inbox.push("private prompt\n".into()).unwrap();
        }
        assert!(inbox.push("overflow".into()).is_err());
        assert_eq!(inbox.prompts.len(), PENDING_COUNT);
        drop(inbox);
        let debug = format!("{input:?}");
        assert!(!debug.contains("private prompt"));
        assert!(!debug.contains("fixture"));
    }
}
