//! Bounded job-output pages shared by engine and graphical clients.

use serde::{Deserialize, Serialize};

/// Nominal bytes per page; UTF-8 alignment may add up to three bytes.
pub const PAGE_BYTES: u64 = 16 * 1024;

/// Validates an opaque output-observer identity without accepting paths or control text.
#[must_use]
pub fn valid_observer(observer: &str) -> bool {
    !observer.is_empty()
        && observer.len() <= 128
        && observer
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
}

/// One requested slice of a tracked job's output file.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Page {
    pub id: i64,
    pub page: u64,
    pub total: u64,
    pub start: u64,
    pub end: u64,
    pub text: String,
    pub lossy: bool,
    #[serde(default)]
    pub tail: bool,
    #[serde(default)]
    pub following: bool,
    #[serde(default)]
    pub observer: Option<String>,
}

impl std::fmt::Debug for Page {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JobOutputPage")
            .field("id", &self.id)
            .field("page", &self.page)
            .field("bytes", &(self.start, self.end))
            .finish_non_exhaustive()
    }
}

impl Page {
    /// Validates range, page and text bounds before rendering remote output.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        if self.id <= 0 || self.page == 0 || self.page > self.total.div_ceil(PAGE_BYTES).max(1) {
            return false;
        }
        if self
            .observer
            .as_deref()
            .is_some_and(|observer| !valid_observer(observer))
        {
            return false;
        }
        if (self.following && !self.tail)
            || (self.tail && self.page != self.total.div_ceil(PAGE_BYTES).max(1))
        {
            return false;
        }
        let nominal = if self.tail {
            self.total.saturating_sub(PAGE_BYTES)
        } else {
            (self.page - 1) * PAGE_BYTES
        };
        let end = nominal.saturating_add(PAGE_BYTES).min(self.total);
        self.start >= nominal
            && (self.page != 1 || self.start == 0)
            && self.start <= nominal.saturating_add(3).min(self.total)
            && self.end >= end
            && self.end <= end.saturating_add(3).min(self.total)
            && self.start <= self.end
            && self.text.len() as u64 >= self.end - self.start
            && (self.lossy || self.text.len() as u64 == self.end - self.start)
            && self.text.len() as u64 <= (self.end - self.start) * 3
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn job_output_follow_requires_a_valid_tail_and_preserves_legacy_snapshots() {
        let live = Page {
            id: 1,
            page: 2,
            total: PAGE_BYTES + 5,
            start: 5,
            end: PAGE_BYTES + 5,
            text: "x".repeat(usize::try_from(PAGE_BYTES).unwrap()),
            lossy: false,
            tail: true,
            following: true,
            observer: Some("test-observer".into()),
        };
        assert!(live.is_valid());
        let mut forged = live.clone();
        forged.tail = false;
        assert!(!forged.is_valid());
        forged = live.clone();
        forged.page = 1;
        assert!(!forged.is_valid());
        forged = live;
        forged.start = 0;
        assert!(!forged.is_valid());
        let legacy: Page = serde_json::from_value(serde_json::json!({"id":1,"page":1,"total":2,"start":0,"end":2,"text":"ok","lossy":false})).unwrap();
        assert!(legacy.is_valid() && !legacy.tail && !legacy.following);
    }

    #[test]
    fn job_output_pages_reject_forged_ranges_and_keep_text_out_of_debug() {
        let valid = Page {
            id: 1,
            page: 1,
            total: 7,
            start: 0,
            end: 7,
            text: "private".into(),
            lossy: false,
            tail: false,
            following: false,
            observer: None,
        };
        assert!(valid.is_valid());
        assert!(!format!("{valid:?}").contains("private"));
        assert_eq!(
            serde_json::from_str::<Page>(&serde_json::to_string(&valid).unwrap()).unwrap(),
            valid
        );
        let mut forged = valid.clone();
        forged.start = 1;
        assert!(!forged.is_valid());
        forged = valid.clone();
        forged.page = 0;
        assert!(!forged.is_valid());
        forged = valid.clone();
        forged.page = u64::MAX;
        assert!(!forged.is_valid());
        forged = valid.clone();
        forged.text.clear();
        assert!(!forged.is_valid());
        forged = valid.clone();
        forged.end = 8;
        assert!(!forged.is_valid());
        forged = valid;
        forged.id = 0;
        assert!(!forged.is_valid());
    }
}
