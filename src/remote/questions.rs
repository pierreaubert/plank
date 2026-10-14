// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Matches remote answers to the currently open structured question.

// Rust guideline compliant 2026-02-21
use std::sync::Mutex;

use crate::tools::ask::{AskOutcome, AskRequest, parse_repl_answer, rows};

#[derive(Debug)]
struct Pending {
    id: String,
    request: AskRequest,
    answer: Option<AskOutcome>,
}

/// A single question shared by the modal UI and authenticated controller.
#[derive(Debug, Default)]
pub(crate) struct Questions(Mutex<Option<Pending>>);

impl Questions {
    pub(crate) fn open(&self, id: &str, request: AskRequest) {
        *self.0.lock().expect("question slot poisoned") = Some(Pending {
            id: id.into(),
            request,
            answer: None,
        });
    }

    pub(crate) fn answer(&self, id: &str, text: &str) -> bool {
        let mut slot = self.0.lock().expect("question slot poisoned");
        let Some(pending) = slot
            .as_mut()
            .filter(|pending| pending.id == id && pending.answer.is_none())
        else {
            return false;
        };
        // The GUI uses explicit numbered options, with zero reserved for decline.
        if text != "0" {
            let count = rows(&pending.request).len();
            let mut selected = Vec::new();
            for token in text.split(',') {
                let Ok(index) = token.parse::<usize>() else {
                    return false;
                };
                if index == 0 || index > count || selected.contains(&index) {
                    return false;
                }
                selected.push(index);
            }
            if !pending.request.multi && selected.len() != 1 {
                return false;
            }
        }
        pending.answer = Some(if text == "0" {
            AskOutcome::Declined
        } else {
            parse_repl_answer(&pending.request, text)
        });
        true
    }

    pub(crate) fn take_answer(&self, id: &str) -> Option<AskOutcome> {
        let mut slot = self.0.lock().expect("question slot poisoned");
        if slot
            .as_ref()
            .is_some_and(|pending| pending.id == id && pending.answer.is_some())
        {
            slot.take().and_then(|pending| pending.answer)
        } else {
            None
        }
    }

    /// Closes atomically, honoring a remote answer already accepted before local input.
    pub(crate) fn close(&self, id: &str) -> Option<AskOutcome> {
        let mut slot = self.0.lock().expect("question slot poisoned");
        if slot.as_ref().is_some_and(|pending| pending.id == id) {
            slot.take().and_then(|pending| pending.answer)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::ask::AskOption;

    fn request(multi: bool) -> AskRequest {
        AskRequest {
            question: "Choose?".into(),
            header: "Choice".into(),
            multi,
            allow_chat: true,
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

    #[test]
    fn remote_questions_require_matching_ids_and_explicit_valid_choices() {
        let questions = Questions::default();
        questions.open("q1", request(false));
        for (id, answer) in [
            ("old", "1"),
            ("q1", ""),
            ("q1", "4"),
            ("q1", "1,2"),
            ("q1", "yes"),
        ] {
            assert!(!questions.answer(id, answer));
        }
        assert!(questions.answer("q1", "2"));
        assert!(!questions.answer("q1", "1"));
        assert_eq!(
            questions.close("q1"),
            Some(AskOutcome::Answered(vec!["Two".into()]))
        );
        assert!(!questions.answer("q1", "1"));
        questions.open("q2", request(true));
        assert!(!questions.answer("q2", "1,1"));
        assert!(questions.answer("q2", "1,2"));
        assert_eq!(
            questions.take_answer("q2"),
            Some(AskOutcome::Answered(vec!["One".into(), "Two".into()]))
        );
        assert!(questions.take_answer("q2").is_none());
        questions.open("q3", request(false));
        assert!(questions.answer("q3", "3"));
        assert_eq!(questions.take_answer("q3"), Some(AskOutcome::Chat));
        questions.open("q4", request(false));
        assert!(questions.answer("q4", "0"));
        assert_eq!(questions.take_answer("q4"), Some(AskOutcome::Declined));
    }
}
