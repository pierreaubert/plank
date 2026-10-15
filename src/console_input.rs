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
// An incomplete UTF-8 scalar retains at most three bytes between reads.
const UTF8_TAIL_BYTES: usize = 3;
const PENDING_BYTES: usize = BATCH_BYTES;
const PENDING_COUNT: usize = 64;
// Match the terminal aside FIFO depth, without displacing previously accepted questions.
const ASIDE_COUNT: usize = 20;
const ASIDE_BYTES: usize = 64 * 1024;
// Two crossed-cell pointer samples fit even on the largest supported grid.
const FRAME_INPUT_COUNT: usize = 512;
pub(crate) mod output;

#[derive(Debug, Default)]
struct Inbox {
    live: VecDeque<crate::livecontrol::LiveControl>,
    prompts: VecDeque<String>,
    bytes: usize,
    asides: VecDeque<String>,
    aside_bytes: usize,
    fragment: String,
    eof: bool,
    dialogs: VecDeque<String>,
    utf8: Vec<u8>,
}

impl Inbox {
    fn feed(&mut self, bytes: &[u8], id: Option<&str>) -> Result<Vec<Value>, String> {
        if bytes.len() > BATCH_BYTES || self.utf8.len() > UTF8_TAIL_BYTES {
            return Err("Console input batch exceeds 4 MiB".into());
        }
        self.utf8.extend_from_slice(bytes);
        let valid = match std::str::from_utf8(&self.utf8) {
            Ok(_) => self.utf8.len(),
            Err(error) if error.error_len().is_none() && !self.eof => error.valid_up_to(),
            Err(_) => return Err("Console input contains invalid or incomplete UTF-8".into()),
        };
        // Route the valid prefix now. Retaining an entire batch for one split
        // scalar would make the next bounded read exceed the scratch limit.
        let tail = self.utf8.split_off(valid);
        let batch = String::from_utf8(std::mem::replace(&mut self.utf8, tail))
            .map_err(|error| error.to_string())?;
        self.route(&batch, id)
    }
    fn push(&mut self, prompt: String) -> Result<(), String> {
        if prompt.trim().is_empty() {
            return Ok(());
        }
        if self.prompts.len() >= PENDING_COUNT
            || self
                .bytes
                .saturating_add(self.aside_bytes)
                .saturating_add(prompt.len())
                > PENDING_BYTES
        {
            return Err("Console pending input limit exceeded".into());
        }
        self.bytes += prompt.len();
        self.prompts.push_back(prompt);
        Ok(())
    }

    fn route(&mut self, batch: &str, id: Option<&str>) -> Result<Vec<Value>, String> {
        if batch.len() > BATCH_BYTES + UTF8_TAIL_BYTES {
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
        if let Some(value) = value
            .as_ref()
            .filter(|value| value["type"] == "live_control")
        {
            let control = serde_json::from_value(value["request"].clone())
                .map_err(|_| "Invalid live control")?;
            if self.live.len() >= PENDING_COUNT {
                return Err("Live control queue is full".into());
            }
            self.live.push_back(control);
            return Ok(());
        }
        if value.is_none() && frame.contains("\"type\"") && frame.contains("\"live_control\"") {
            return Err("Malformed live control frame".into());
        }
        if value
            .as_ref()
            .is_some_and(|value| value["type"] == "prompt")
        {
            self.push(std::mem::take(prompt))?;
            let text = value
                .as_ref()
                .and_then(|value| value["text"].as_str())
                .ok_or("Console prompt requires text")?;
            // Treat the body as opaque input, never recursively as a dialog,
            // frame control or side question, even when it resembles JSON.
            self.push(text.into())?;
            return Ok(());
        }
        if value.is_none() && frame.contains("\"type\"") && frame.contains("\"prompt\"") {
            return Err("Malformed console prompt frame".into());
        }
        if value
            .as_ref()
            .is_some_and(|value| value["type"] == "frame_input")
        {
            self.push(std::mem::take(prompt))?;
            if let Some(value) = value.filter(|value| id.is_some() && value["id"].as_str() == id) {
                if replies.len() >= FRAME_INPUT_COUNT {
                    return Err("Console frame input batch is full".into());
                }
                replies.push(value);
            }
            return Ok(());
        }
        if value.is_none() && frame.contains("\"type\"") && frame.contains("\"frame_input\"") {
            return Err("Malformed console frame input".into());
        }
        if value.as_ref().is_some_and(|value| value["type"] == "btw") {
            self.push(std::mem::take(prompt))?;
            let question = value
                .as_ref()
                .and_then(|value| value["text"].as_str())
                .ok_or("Console side question requires text")?;
            if question.trim().is_empty() || question.len() > ASIDE_BYTES {
                return Err("Console side question must contain 1–65536 bytes of text".into());
            }
            if self.asides.len() >= ASIDE_COUNT
                || self
                    .bytes
                    .saturating_add(self.aside_bytes)
                    .saturating_add(question.len())
                    > PENDING_BYTES
            {
                return Err("Console side question queue is full".into());
            }
            self.aside_bytes += question.len();
            self.asides.push_back(question.into());
            return Ok(());
        }
        if value.is_none() && frame.contains("\"type\"") && frame.contains("\"btw\"") {
            return Err("Malformed console side question frame".into());
        }
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

    fn pop_aside(&mut self) -> Option<String> {
        let question = self.asides.pop_front()?;
        self.aside_bytes -= question.len();
        Some(question)
    }

    fn pop_input(&mut self) -> Option<String> {
        self.pop_aside()
            .map(|question| format!("/btw {question}"))
            .or_else(|| self.pop())
    }
}

/// Owns pending stdin and the optional bridge for one console agent.
#[derive(Default)]
pub(crate) struct ConsoleInput {
    controls: Mutex<Option<Value>>,
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
            .field("pending_asides", &inbox.asides.len())
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

    pub(crate) fn remote(&self) -> Option<Arc<RemoteState>> {
        self.remote.lock().expect("console bridge poisoned").clone()
    }

    pub(crate) fn eof(&self) -> bool {
        self.inbox.lock().expect("console inbox poisoned").eof
    }

    pub(crate) fn has_pending(&self) -> bool {
        let inbox = self.inbox.lock().expect("console inbox poisoned");
        !inbox.prompts.is_empty()
            || !inbox.asides.is_empty()
            || !inbox.fragment.is_empty()
            || !inbox.utf8.is_empty()
    }

    /// Polls owned input without consuming ordinary prompts or dialog replies.
    pub(crate) fn poll_asides(
        &self,
        mut read: impl FnMut(&mut bool) -> Result<Option<Vec<u8>>, String>,
    ) -> Result<(), String> {
        if !self.eof() {
            self.read(None, &mut read)?;
        }
        Ok(())
    }

    pub(crate) fn has_asides(&self) -> bool {
        !self
            .inbox
            .lock()
            .expect("console inbox poisoned")
            .asides
            .is_empty()
    }

    pub(crate) fn pop_aside(&self) -> Option<String> {
        self.inbox
            .lock()
            .expect("console inbox poisoned")
            .pop_aside()
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
        let replies = inbox.feed(batch.as_deref().unwrap_or(&[]), id)?;
        let live = std::mem::take(&mut inbox.live);
        drop(inbox);
        for control in live {
            self.apply_live_control(control);
        }
        Ok(replies)
    }

    pub(crate) fn remember_controls(&self, value: &Value) {
        let mut controls = self.controls.lock().expect("console controls poisoned");
        if controls.as_ref() != Some(value) {
            *controls = Some(value.clone());
        }
    }

    fn apply_live_control(&self, control: crate::livecontrol::LiveControl) {
        let reply = crate::ui::apply_live_control(control);
        let mut cache = self.controls.lock().expect("console controls poisoned");
        if let Some(controls) = &mut *cache {
            crate::ui::refresh_live_settings(controls);
            if crate::ui::gui_protocol_enabled() {
                eprintln!("+PLANK_CONTROLS {controls}");
            }
            if let Some(remote) = self.remote() {
                remote
                    .bus
                    .broadcast(crate::worker::UiEvent::Controls(controls.clone()));
            }
        }
        drop(cache);
        eprintln!("{reply}");
        if let Some(remote) = self.remote() {
            remote.bus.broadcast(crate::worker::UiEvent::Dim(reply));
        }
    }

    /// Polls frame controls while preserving ordinary prompts and side questions.
    pub(crate) fn frame_inputs(
        &self,
        id: &str,
        mut read: impl FnMut(&mut bool) -> Result<Option<Vec<u8>>, String>,
    ) -> Result<Vec<crate::guiframe::Request>, String> {
        self.read(Some(id), &mut read)?
            .into_iter()
            .filter(|value| value["type"] == "frame_input")
            .map(|value| {
                let request: crate::guiframe::Request =
                    serde_json::from_value(value).map_err(|e| e.to_string())?;
                if request.is_valid() {
                    Ok(request)
                } else {
                    Err("Invalid frame input".into())
                }
            })
            .collect()
    }

    /// Returns preserved input before reading; stale replies never become prompts.
    pub(crate) fn prompt(
        &self,
        mut read: impl FnMut(&mut bool) -> Result<Option<Vec<u8>>, String>,
    ) -> Result<Option<String>, String> {
        if let Some(prompt) = self
            .inbox
            .lock()
            .expect("console inbox poisoned")
            .pop_input()
        {
            return Ok(Some(prompt));
        }
        if !self.eof() {
            self.read(None, &mut read)?;
        }
        Ok(self
            .inbox
            .lock()
            .expect("console inbox poisoned")
            .pop_input())
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
        let remote = self.remote();
        let streaming = gui
            && std::env::var_os("PLANK_GUI_EDITOR_STREAM").as_deref()
                == Some(std::ffi::OsStr::new("1"));
        let limit = if streaming {
            crate::documenttransfer::DOCUMENT_BYTES
        } else {
            remote
                .as_ref()
                .map_or(crate::remote::editors::DOCUMENT_BYTES, |remote| {
                    remote.editor_limit()
                })
        };
        if text.len() > limit {
            return Err(format!(
                "Document exceeds this client's {} KiB editing limit",
                limit / 1024
            ));
        }
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
                .is_some_and(|text| editors.reply_console(id, text)),
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
            let stream = matches!(self.slot, Slot::Editor(_))
                && std::env::var_os("PLANK_GUI_EDITOR_STREAM").as_deref()
                    == Some(std::ffi::OsStr::new("1"));
            if let Some(frames) = editor_stream(self.id, &value, stream) {
                for frame in frames {
                    eprintln!("+PLANK_EDITOR_STREAM {frame}");
                }
            } else {
                eprintln!("+PLANK_{marker} {value}");
            }
        }
        if let Some(remote) = self.remote {
            remote.bus.broadcast(match self.slot {
                Slot::Question(_) => UiEvent::Question(value),
                Slot::Editor(_) => UiEvent::Editor(value.into()),
            });
        }
    }
}

fn editor_stream<'a>(
    id: &'a str,
    value: &'a Value,
    enabled: bool,
) -> Option<impl Iterator<Item = Value> + 'a> {
    let text = value["text"].as_str()?;
    if !enabled || text.len() <= crate::documenttransfer::CHUNK_BYTES {
        return None;
    }
    let frames = crate::documenttransfer::Sender::new(id, text).ok()?;
    let title = value["title"].as_str().unwrap_or("Document");
    Some(frames.map(move |frame| serde_json::json!({"id":id,"title":title,"frame":frame})))
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
        progress.last = Some(
            std::time::Instant::now()
                .checked_sub(std::time::Duration::from_secs(2))
                .expect("fixture clock supports a two-second lookback"),
        );
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
    fn console_prompt_frames_preserve_boundaries_and_keep_protocol_text_opaque() {
        let mut inbox = Inbox::default();
        let first = "/remember first line\nsecond 🦀 line\n{\"type\":\"dialog_reply\",\"id\":\"q1\",\"answer\":\"1\"}";
        let second = "{\"type\":\"btw\",\"text\":\"literal question\"}";
        let mut batch = Vec::new();
        for text in [first, second] {
            batch.extend(
                serde_json::to_vec(&serde_json::json!({"type":"prompt","text":text})).unwrap(),
            );
            batch.push(b'\n');
        }
        // Split every byte, including inside UTF-8 scalars and JSON escapes.
        for byte in batch {
            assert!(inbox.feed(&[byte], Some("q1")).unwrap().is_empty());
        }
        assert_eq!(inbox.pop().as_deref(), Some(first));
        assert_eq!(inbox.pop().as_deref(), Some(second));
        assert!(inbox.pop().is_none());
        assert!(inbox.asides.is_empty());
        assert!(
            inbox
                .feed(b"{\"type\":\"prompt\",\"text\":null}\n", None)
                .is_err()
        );
        assert!(
            inbox
                .feed(b"{\"type\":\"prompt\", invalid}\n", None)
                .is_err()
        );
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
    fn console_live_frames_are_bounded_and_never_become_prompts_or_dialog_replies() {
        let mut inbox = Inbox::default();
        let frame = serde_json::to_vec(&serde_json::json!({"type":"live_control","request":{"control":"micro_compaction","enabled":false}})).unwrap();
        for byte in frame.iter().copied().chain(*b"\n") {
            assert!(inbox.feed(&[byte], Some("q1")).unwrap().is_empty());
        }
        assert_eq!(
            inbox.live.pop_front(),
            Some(crate::livecontrol::LiveControl::MicroCompaction(false))
        );
        assert!(inbox.pop().is_none());
        assert!(inbox.asides.is_empty());
        for invalid in [
            "{\"type\":\"live_control\", broken}\n",
            "{\"type\":\"live_control\",\"request\":{\"control\":\"quit\",\"enabled\":true}}\n",
            "{\"type\":\"live_control\",\"request\":{\"control\":\"loop_guard\",\"enabled\":\"off\"}}\n",
        ] {
            assert!(inbox.feed(invalid.as_bytes(), None).is_err());
        }
        assert!(inbox.pop().is_none());
        let mut batch = frame;
        batch.push(b'\n');
        for _ in 0..PENDING_COUNT {
            inbox.feed(&batch, None).unwrap();
        }
        assert!(inbox.feed(&batch, None).is_err());
        assert_eq!(inbox.live.len(), PENDING_COUNT);
    }

    #[test]
    fn console_side_questions_keep_fragmented_unicode_out_of_main_input() {
        let input = ConsoleInput::default();
        let question = "Explain 🦀.\nThen say why.";
        let frame = format!("{}\n", serde_json::json!({"type":"btw","text":question}));
        let split = frame
            .as_bytes()
            .windows(4)
            .position(|bytes| bytes == "🦀".as_bytes())
            .unwrap()
            + 1;
        let mut reads = VecDeque::from([
            Some(frame.as_bytes()[..split].to_vec()),
            None,
            Some(frame.as_bytes()[split..].to_vec()),
        ]);
        assert!(
            input
                .prompt(|_| Ok(reads.pop_front().unwrap()))
                .unwrap()
                .is_none()
        );
        assert!(
            input
                .prompt(|_| Ok(reads.pop_front().unwrap()))
                .unwrap()
                .is_none()
        );
        assert_eq!(
            input.prompt(|_| Ok(reads.pop_front().unwrap())).unwrap(),
            Some(format!("/btw {question}"))
        );
        assert!(!input.has_pending());
        assert!(input.inbox.lock().unwrap().prompts.is_empty());
    }

    #[test]
    fn graphical_frame_inputs_reject_stale_and_malformed_controls_without_creating_model_prompts() {
        let mut inbox = Inbox::default();
        let bytes = b"{\"type\":\"frame_input\",\"id\":\"old\",\"event\":\"close\"}\n{\"type\":\"frame_input\",\"id\":\"current\",\"event\":\"key\",\"code\":\"a\",\"text\":\"a\"}\n";
        let replies = inbox.feed(bytes, Some("current")).unwrap();
        assert_eq!(replies.len(), 1);
        assert!(inbox.pop().is_none());
        assert!(
            inbox
                .feed(b"{\"type\":\"frame_input\",broken}\n", None)
                .is_err()
        );
        assert!(inbox.pop().is_none());
        assert!(inbox.feed(bytes, None).unwrap().is_empty());
        assert!(inbox.pop().is_none());
    }

    #[test]
    fn console_side_questions_survive_editor_replies_without_answering_the_dialog() {
        let input = ConsoleInput::default();
        let batch = format!(
            "ordinary prompt\n{}\n{}\n",
            serde_json::json!({"type":"btw","text":"secret side question"}),
            serde_json::json!({"type":"dialog_reply","id":"e1","text":"edited"})
        );
        let mut reads = VecDeque::from([Some(batch.into_bytes())]);
        assert_eq!(
            input
                .edit("e1", "Editor", "original", false, |_| Ok(reads
                    .pop_front()
                    .expect("the matching reply must finish the editor")))
                .unwrap()
                .as_deref(),
            Some("edited")
        );
        let debug = format!("{input:?}");
        assert!(debug.contains("pending_asides: 1"));
        assert!(!debug.contains("secret side question"));
        assert_eq!(
            input
                .prompt(|_| panic!("aside is already buffered"))
                .unwrap()
                .as_deref(),
            Some("/btw secret side question")
        );
        assert_eq!(
            input
                .prompt(|_| panic!("main input must be preserved"))
                .unwrap()
                .as_deref(),
            Some("ordinary prompt\n")
        );
        assert!(!input.has_pending());
    }

    #[test]
    fn console_side_question_limits_refuse_overflow_without_evicting_accepted_questions() {
        let mut inbox = Inbox::default();
        for index in 0..ASIDE_COUNT {
            let frame = format!(
                "{}\n",
                serde_json::json!({"type":"btw","text":format!("question {index}")})
            );
            inbox.feed(frame.as_bytes(), None).unwrap();
        }
        let extra = b"{\"type\":\"btw\",\"text\":\"extra\"}\n";
        assert!(inbox.feed(extra, None).is_err());
        for index in 0..ASIDE_COUNT {
            assert_eq!(inbox.pop_aside(), Some(format!("question {index}")));
        }
        assert_eq!(inbox.aside_bytes, 0);
        assert!(inbox.prompts.is_empty());
        inbox.feed(extra, None).unwrap();
        assert_eq!(inbox.pop_aside().as_deref(), Some("extra"));
        for text in [String::new(), " ".to_owned(), "x".repeat(ASIDE_BYTES + 1)] {
            let frame = format!("{}\n", serde_json::json!({"type":"btw","text":text}));
            assert!(inbox.feed(frame.as_bytes(), None).is_err());
        }
        for frame in [
            b"{\"type\":\"btw\",\"text\":false}\n".as_slice(),
            b"{\"type\":\"btw\",bad}\n",
        ] {
            assert!(inbox.feed(frame, None).is_err());
        }
        assert!(inbox.prompts.is_empty());
        assert!(inbox.asides.is_empty());
        inbox.push("p".repeat(PENDING_BYTES)).unwrap();
        assert!(inbox.feed(extra, None).is_err());
        assert_eq!(inbox.pop().unwrap().len(), PENDING_BYTES);
    }

    #[test]
    fn console_editor_opening_protocol_requires_explicit_gui_client_opt_in() {
        for enabled in [false, true] {
            let mut command = std::process::Command::new(std::env::current_exe().unwrap());
            command
                .args([
                    "--exact",
                    "console_input::tests::console_editor_opening_wire_fixture",
                    "--ignored",
                    "--nocapture",
                ])
                .env("PLANK_EDITOR_OPEN_FIXTURE", "1");
            if enabled {
                command.env("PLANK_GUI_EDITOR_STREAM", "1");
            } else {
                command.env_remove("PLANK_GUI_EDITOR_STREAM");
            }
            let output = command.output().unwrap();
            assert!(output.status.success());
            let stderr = String::from_utf8(output.stderr).unwrap();
            let streamed = stderr
                .lines()
                .filter(|line| line.starts_with("+PLANK_EDITOR_STREAM "))
                .count();
            let inline = stderr
                .lines()
                .filter(|line| line.starts_with("+PLANK_EDITOR "))
                .count();
            assert!(if enabled {
                streamed > 3 && inline == 0
            } else {
                streamed == 0 && inline == 1
            });
            assert_eq!(
                stderr
                    .lines()
                    .filter(|line| line.starts_with("+PLANK_EDITOR_DONE "))
                    .count(),
                1
            );
        }
    }

    #[test]
    #[ignore = "isolated graphical protocol fixture invoked by its parent test"]
    fn console_editor_opening_wire_fixture() {
        if std::env::var_os("PLANK_EDITOR_OPEN_FIXTURE").is_none() {
            return;
        }
        let input = ConsoleInput::default();
        let text = "\u{0}🦀\\\r\n".repeat(20_000);
        assert!(
            input
                .edit("e1", "Document", &text, true, |eof| {
                    *eof = true;
                    Ok(None)
                })
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn console_editor_opening_streams_require_opt_in_and_preserve_bounded_text() {
        let text = "\u{0}🦀\\\r\n".repeat(20_000);
        let value = serde_json::json!({"id":"e1","title":"\u{0}".repeat(4096),"text":text});
        assert!(editor_stream("e1", &value, false).is_none());
        assert!(editor_stream("e1", &serde_json::json!({"text":"small"}), true).is_none());
        let mut receiver = crate::documenttransfer::Receiver::default();
        let mut complete = None;
        for value in editor_stream("e1", &value, true).unwrap() {
            assert_eq!(value["id"], "e1");
            assert!(value.to_string().len() < crate::documenttransfer::CHUNK_BYTES * 6 + 32 * 1024);
            let frame = serde_json::from_value(value["frame"].clone()).unwrap();
            complete = receiver.accept(&frame).unwrap();
        }
        assert!(
            complete.as_deref().is_some_and(|actual| actual == text),
            "Opening must preserve all document bytes"
        );
    }

    #[test]
    fn console_unicode_carry_does_not_retain_or_overflow_a_complete_batch() {
        let mut inbox = Inbox::default();
        let mut prefix = br#"{"type":"dialog_reply","id":"e1","text":""#.to_vec();
        prefix.push("🦀".as_bytes()[0]);
        assert!(inbox.feed(&prefix, Some("e1")).unwrap().is_empty());
        assert_eq!(inbox.utf8.len(), 1);
        let mut batch = "🦀".as_bytes()[1..].to_vec();
        batch.extend_from_slice(b"\"}\n");
        // Three legal prompt frames fill the next read exactly. A carried
        // scalar may make its decoded prefix a few bytes larger than the read.
        for index in 0..3 {
            let empty = serde_json::json!({"type":"prompt","text":""}).to_string();
            let text_bytes = if index < 2 {
                BATCH_BYTES / 3
            } else {
                BATCH_BYTES - batch.len() - empty.len() - 1
            };
            let frame = serde_json::json!({"type":"prompt","text":"x".repeat(text_bytes)});
            batch.extend_from_slice(frame.to_string().as_bytes());
            batch.push(b'\n');
        }
        assert_eq!(batch.len(), BATCH_BYTES);
        let replies = inbox.feed(&batch, Some("e1")).unwrap();
        assert_eq!(replies.len(), 1);
        assert_eq!(replies[0]["text"], "🦀");
        assert!(inbox.utf8.is_empty());
        assert_eq!(inbox.prompts.len(), 3);
    }

    #[test]
    fn console_editor_streams_preserve_fragmented_unicode_and_queued_prompts() {
        let input = ConsoleInput::default();
        let text = "🦀\r\n\"\\".repeat(100_000);
        let mut reads = VecDeque::new();
        for frame in crate::documenttransfer::Sender::new("save-1", &text).unwrap() {
            let mut bytes = wire("e1", "text", serde_json::to_value(frame).unwrap());
            if matches!(
                serde_json::from_slice::<Value>(&bytes).unwrap()["text"]["operation"].as_str(),
                Some("end")
            ) {
                bytes.extend_from_slice(b"queued prompt\n");
            }
            reads.extend(bytes.chunks(997).map(|bytes| Some(bytes.to_vec())));
        }
        let edited = input
            .edit("e1", "Document", "initial", false, |_| {
                Ok(reads
                    .pop_front()
                    .expect("only a complete End may resolve the editor"))
            })
            .unwrap();
        assert_eq!(edited.as_deref(), Some(text.as_str()));
        assert_eq!(
            input
                .prompt(|_| panic!("prompt must remain queued"))
                .unwrap()
                .as_deref(),
            Some("queued prompt\n")
        );
        assert!(!input.has_pending());
    }

    #[test]
    fn console_editor_streams_never_save_on_eof_before_completion() {
        let input = ConsoleInput::default();
        let mut frames =
            crate::documenttransfer::Sender::new("partial", "unsaved document").unwrap();
        let mut bytes = Vec::new();
        for frame in frames.by_ref().take(2) {
            bytes.extend_from_slice(&wire("e1", "text", serde_json::to_value(frame).unwrap()));
        }
        let mut batch = Some(bytes);
        let edited = input
            .edit("e1", "Document", "initial", false, |eof| {
                if batch.is_none() {
                    *eof = true;
                }
                Ok(batch.take())
            })
            .unwrap();
        assert!(edited.is_none());
        let end = serde_json::to_value(frames.next().unwrap()).unwrap();
        assert!(!input.editors.reply_console("e1", &end));
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
