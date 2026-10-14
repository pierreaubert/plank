//! Bounded delegated output mirrors for graphical clients.

// Rust guideline compliant 2026-02-21
use super::{ConsoleInput, SubagentProgress};
use crate::{remote::control::RemoteState, viz::RenderSink, worker::UiEvent};
use std::{
    io::Write as _,
    sync::Arc,
    time::{Duration, Instant},
};

/// Captures rendered child text without retaining another transcript.
pub(crate) struct SubagentOutput {
    channel: Arc<ConsoleInput>,
    run_id: u64,
    remote: Option<Arc<RemoteState>>,
    gui: bool,
    last: Option<Instant>,
    dirty: bool,
}

impl std::fmt::Debug for SubagentOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SubagentOutput")
            .field("run_id", &self.run_id)
            .finish_non_exhaustive()
    }
}

impl SubagentOutput {
    pub(super) fn new(progress: &SubagentProgress) -> Self {
        Self {
            channel: Arc::clone(&progress.channel),
            run_id: progress.run_id,
            remote: progress.remote.clone(),
            gui: progress.gui,
            last: None,
            dirty: false,
        }
    }

    fn append(&mut self, text: &str) {
        let Ok(mut pane) = self.channel.subagents.lock() else {
            return;
        };
        let Some(run) = pane
            .runs
            .iter_mut()
            .find(|run| run.id == self.run_id && run.running)
        else {
            return;
        };
        let (tail, clipped) = crate::remote::control::bounded_subagent_output(text);
        if tail.is_empty() {
            return;
        }
        if clipped {
            run.output = tail;
        } else {
            run.output.push_str(&tail);
        }
        let limit = crate::remote::control::SUBAGENT_OUTPUT_BYTES;
        let overflow = run.output.len() > limit;
        if overflow {
            let mut start = run.output.len() - limit;
            while !run.output.is_char_boundary(start) {
                start += 1;
            }
            run.output.drain(..start);
        }
        run.output_truncated |= clipped || overflow;
        self.dirty = true;
        drop(pane);
        if self
            .last
            .is_none_or(|last| last.elapsed() >= Duration::from_secs(1))
        {
            self.flush();
        }
    }

    fn flush(&mut self) {
        if !self.dirty {
            return;
        }
        let snapshot = {
            let Ok(pane) = self.channel.subagents.lock() else {
                return;
            };
            if !pane
                .runs
                .iter()
                .any(|run| run.id == self.run_id && run.running)
            {
                return;
            }
            crate::remote::control::SubagentWire::capture(&pane, crate::tui::roster_clock_ms())
        };
        if self.gui {
            let _ = writeln!(
                std::io::stderr(),
                "+PLANK_TELEMETRY {}",
                serde_json::json!({"type":"metrics", "snapshot":snapshot})
            );
        }
        if let Some(remote) = &self.remote {
            remote.bus.broadcast(UiEvent::Metrics(snapshot));
        }
        self.dirty = false;
        self.last = Some(Instant::now());
    }
}

impl Drop for SubagentOutput {
    fn drop(&mut self) {
        self.flush();
    }
}

impl RenderSink for SubagentOutput {
    fn visible_text(&mut self, text: &str) {
        self.append(text);
    }
    fn think_text(&mut self, text: &str) {
        self.append(text);
    }
    fn tool_text(&mut self, text: &str) {
        self.append(text);
    }
    fn error_text(&mut self, text: &str) {
        self.append(text);
    }
}

struct OutputMirror {
    inner: Box<dyn RenderSink + Send>,
    output: SubagentOutput,
}

impl RenderSink for OutputMirror {
    fn visible_text(&mut self, text: &str) {
        self.inner.visible_text(text);
        self.output.visible_text(text);
    }
    fn think_text(&mut self, text: &str) {
        self.inner.think_text(text);
        self.output.think_text(text);
    }
    fn tool_text(&mut self, text: &str) {
        self.inner.tool_text(text);
        self.output.tool_text(text);
    }
    fn error_text(&mut self, text: &str) {
        self.inner.error_text(text);
        self.output.error_text(text);
    }
    fn preview_status(&mut self, text: Option<&str>) {
        self.inner.preview_status(text);
    }
}

/// Adds an output mirror only for an observed delegated pass.
pub(crate) fn mirror_sink(
    inner: Box<dyn RenderSink + Send>,
    progress: Option<&SubagentProgress>,
) -> Box<dyn RenderSink + Send> {
    match progress {
        Some(progress) => Box::new(OutputMirror {
            inner,
            output: SubagentOutput::new(progress),
        }),
        None => inner,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::remote::control::MetricsWire;

    fn setup() -> (Arc<ConsoleInput>, Arc<RemoteState>, SubagentProgress) {
        let channel = Arc::new(ConsoleInput::default());
        let remote = RemoteState::test_bridge();
        channel.subagent_event(&UiEvent::SubStart {
            label: "worker".into(),
            task: "Inspect code".into(),
        });
        let progress = channel
            .progress("worker", false, Some(Arc::clone(&remote)))
            .unwrap();
        (channel, remote, progress)
    }

    #[test]
    fn subagent_output_preserves_primary_bytes_and_flushes_the_final_chunk() {
        let (_, remote, progress) = setup();
        let receiver = remote.bus.subscribe();
        let primary = crate::viz::CollectSink::default();
        let mut sink = mirror_sink(Box::new(primary.clone()), Some(&progress));
        sink.visible_text("first ");
        sink.tool_text("tool ");
        sink.error_text("error ");
        sink.visible_text("last 🦀");
        assert_eq!(receiver.try_iter().count(), 1, "output callbacks coalesce");
        drop(sink);
        assert_eq!(primary.take(), "first tool error last 🦀");
        assert!(receiver.try_iter().any(|event| matches!(event.event,
            UiEvent::Metrics(MetricsWire::Subagents { rows, .. })
                if rows[0].output.as_deref() == Some("first tool error last 🦀"))));
        assert_eq!(remote.bus.scrollback_since(None).0.len(), 1);
    }

    #[test]
    fn subagent_output_is_not_rebroadcast_as_main_transcript_text() {
        let (_, remote, progress) = setup();
        let primary = crate::viz::CollectSink::default();
        let mut sink = crate::worker::MirrorSink {
            aside: false,
            gui: false,
            inner: primary.clone(),
            bus: Some(Arc::clone(&remote.bus)),
            activity: Some(progress.output()),
            suppress_primary: true,
        };
        sink.visible_text("child answer");
        sink.tool_text("child tool");
        drop(sink);
        assert!(primary.take().is_empty());
        let (events, _) = remote.bus.scrollback_since(None);
        assert!(
            events
                .iter()
                .all(|event| matches!(event.event, UiEvent::Metrics(_)))
        );
        assert!(events.iter().any(|event| matches!(&event.event,
            UiEvent::Metrics(MetricsWire::Subagents { rows, .. })
                if rows[0].output.as_deref() == Some("child answerchild tool"))));
    }

    #[test]
    fn subagent_output_bounds_unicode_tails_and_refuses_retired_identities() {
        let (channel, remote, progress) = setup();
        let receiver = remote.bus.subscribe();
        let mut output = progress.output();
        output.visible_text(&format!("{}latest", "🦀".repeat(5000)));
        assert!(receiver.try_iter().any(|event| matches!(event.event,
            UiEvent::Metrics(MetricsWire::Subagents { rows, .. })
                if rows[0].output_truncated && rows[0].output.as_ref().is_some_and(|text| text.len() <= 4096 && text.ends_with("latest")))));
        channel.subagent_event(&UiEvent::SubEnd);
        channel.subagent_event(&UiEvent::SubStart {
            label: "worker".into(),
            task: "new work".into(),
        });
        output.visible_text("late old output");
        drop(output);
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
        assert!(rows[0].output.is_none());
    }

    #[test]
    fn subagent_output_obeys_renderer_thinking_visibility() {
        for visible in [false, true] {
            let (_, remote, progress) = setup();
            let sink = mirror_sink(
                Box::new(crate::viz::CollectSink::default()),
                Some(&progress),
            );
            let mut stream = crate::viz::StreamRenderer::with_syntax(
                sink,
                crate::sysprompt::ToolSyntax::for_model_name("echo"),
            );
            stream.set_show_thinking(visible);
            stream.push("<think>private reasoning</think>public answer");
            stream.finish();
            drop(stream);
            let (events, _) = remote.bus.scrollback_since(None);
            assert!(events.iter().any(|event| matches!(&event.event,
                UiEvent::Metrics(MetricsWire::Subagents { rows, .. })
                    if rows[0].output.as_ref().is_some_and(|text| text.contains("public answer") && text.contains("private reasoning") == visible))));
        }
    }
}
