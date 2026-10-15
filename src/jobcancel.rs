//! Scoped cancellation of output observation without borrowing the agent.

use std::sync::{
    Arc, Mutex, Weak,
    atomic::{AtomicU8, Ordering},
};

pub(crate) const RUNNING: u8 = 0;
pub(crate) const RETIRED: u8 = 1;
pub(crate) const REQUESTED: u8 = 2;

#[derive(Debug, Default)]
pub(crate) struct Hub(Mutex<Option<(String, Weak<AtomicU8>)>>);

pub(crate) struct Lease {
    hub: Arc<Hub>,
    observer: String,
}

impl Drop for Lease {
    fn drop(&mut self) {
        if let Ok(mut current) = self.hub.0.lock()
            && current
                .as_ref()
                .is_some_and(|(observer, _)| observer == &self.observer)
        {
            *current = None;
        }
    }
}

impl Hub {
    pub(crate) fn register(
        self: &Arc<Self>,
        observer: &str,
        control: &Arc<AtomicU8>,
    ) -> Result<Lease, String> {
        *self.0.lock().map_err(|_| "Output control state poisoned")? =
            Some((observer.into(), Arc::downgrade(control)));
        Ok(Lease {
            hub: Arc::clone(self),
            observer: observer.into(),
        })
    }

    pub(crate) fn request(&self, observer: &str) -> bool {
        if !crate::joboutput::valid_observer(observer) {
            return false;
        }
        let Ok(current) = self.0.lock() else {
            return false;
        };
        current
            .as_ref()
            .filter(|(id, _)| id == observer)
            .and_then(|(_, control)| control.upgrade())
            .is_some_and(|control| {
                matches!(
                    control.compare_exchange(
                        RUNNING,
                        REQUESTED,
                        Ordering::AcqRel,
                        Ordering::Acquire
                    ),
                    Ok(_) | Err(REQUESTED)
                )
            })
    }
}

pub(crate) fn mailbox_requested(path: &std::path::Path, observer: &str) -> bool {
    use std::io::Read as _;
    let Ok(file) = std::fs::File::open(path) else {
        return false;
    };
    let mut body = Vec::new();
    // A control packet contains only an opaque identity, never prompt text.
    if file.take(513).read_to_end(&mut body).is_err() || body.len() > 512 {
        return false;
    }
    serde_json::from_slice::<serde_json::Value>(&body)
        .is_ok_and(|request| request["observer"].as_str() == Some(observer))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn job_output_cancel_fences_replacements_and_retired_observers() {
        let hub = Arc::new(Hub::default());
        let old = Arc::new(AtomicU8::new(RUNNING));
        let old_lease = hub.register("old", &old).unwrap();
        assert!(!hub.request("wrong"));
        assert!(!hub.request("invalid identity"));
        assert_eq!(old.load(Ordering::Acquire), RUNNING);
        let current = Arc::new(AtomicU8::new(RUNNING));
        let current_lease = hub.register("current", &current).unwrap();
        drop(old_lease);
        assert!(!hub.request("old"));
        assert!(hub.request("current"));
        assert!(hub.request("current"), "retries are idempotent");
        assert_eq!(current.load(Ordering::Acquire), REQUESTED);
        current.store(RETIRED, Ordering::Release);
        assert!(!hub.request("current"));
        drop(current_lease);
        assert!(!hub.request("current"));
        assert_eq!(old.load(Ordering::Acquire), RUNNING);
    }

    #[test]
    fn job_output_cancel_mailbox_rejects_other_observers_and_invalid_packets() {
        let mailbox = std::env::temp_dir().join(format!(
            "plank-job-control-test-{}",
            crate::remote::control::replay_instance_id().unwrap()
        ));
        assert!(!mailbox_requested(&mailbox, "live"));
        for body in [
            "{}".to_owned(),
            "not JSON".into(),
            r#"{"observer":"stale"}"#.into(),
            format!(r#"{{"observer":"live","padding":"{}"}}"#, "x".repeat(512)),
        ] {
            std::fs::write(&mailbox, body).unwrap();
            assert!(!mailbox_requested(&mailbox, "live"));
        }
        std::fs::write(&mailbox, r#"{"observer":"live"}"#).unwrap();
        assert!(mailbox_requested(&mailbox, "live"));
        std::fs::remove_file(mailbox).unwrap();
    }
}
