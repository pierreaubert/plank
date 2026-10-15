//! Owned subprocesses for cancellable foreground model downloads.
//!
//! A bounded private stdin packet carries the URL and partial-file path. The
//! parent retains that pipe as a lifeline; its closure terminates the worker,
//! including blocked network calls. Stdout carries only bounded size metadata.
//! Cancellation reaps the child before a partial file can be inspected or moved.

use std::{
    fmt,
    io::{self, BufRead, Read, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, Command, ExitStatus, Stdio},
    sync::mpsc::{self, Receiver, TryRecvError},
    thread::JoinHandle,
};

use serde::{Deserialize, Serialize};

/// Private worker startup mode; download URLs never appear in its arguments.
pub const WORKER_ARG: &str = "--foreground-url-download-worker";
// Includes JSON escaping; ample for catalog URLs/paths without unbounded input.
const REQUEST_BYTES: usize = 64 * 1024;
const META_BYTES: usize = 4096;
const META_PREFIX: &str = "+PLANK_URL_FETCH ";

#[derive(Serialize, Deserialize)]
struct Request {
    url: String,
    part: PathBuf,
}

impl fmt::Debug for Request {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Request").finish_non_exhaustive()
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct Metadata {
    total: Option<u64>,
}

fn packet_error() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        "Invalid foreground download packet",
    )
}

fn read_request(reader: impl BufRead) -> io::Result<Request> {
    let mut bytes = Vec::new();
    reader
        .take(REQUEST_BYTES as u64 + 1)
        .read_until(b'\n', &mut bytes)?;
    if bytes.len() > REQUEST_BYTES || bytes.last() != Some(&b'\n') {
        return Err(packet_error());
    }
    let request: Request = serde_json::from_slice(&bytes).map_err(|_| packet_error())?;
    if request.url.is_empty() || request.part.as_os_str().is_empty() {
        return Err(packet_error());
    }
    Ok(request)
}

fn send_metadata(total: Option<u64>) -> io::Result<()> {
    let mut stdout = io::stdout().lock();
    writeln!(
        stdout,
        "{META_PREFIX}{}",
        serde_json::to_string(&Metadata { total }).map_err(|_| packet_error())?
    )?;
    stdout.flush()
}

fn watch_parent() {
    std::thread::spawn(|| {
        let mut byte = [0];
        // Only the request belongs on this pipe. EOF, errors, or trailing input
        // mean the owner is gone or the private protocol has been violated.
        let _ = io::stdin().read(&mut byte);
        std::process::exit(1);
    });
}

/// Runs the private fetch worker without loading settings, plugins, or an engine.
///
/// Returns zero on success, one on fetch failure, or two on malformed input.
/// Diagnostic output never includes the private request.
///
/// # Abort
/// Exits this process when its parent's stdin lifeline closes. Call only from
/// the private worker startup mode, never inside an agent process.
#[must_use]
pub fn run_worker() -> u8 {
    let Ok(request) = read_request(io::stdin().lock()) else {
        return 2;
    };
    watch_parent();
    let total = super::content_length(&request.url);
    if send_metadata(total).is_err() {
        return 1;
    }
    if total.is_some_and(|total| {
        total > 0 && std::fs::metadata(&request.part).is_ok_and(|metadata| metadata.len() == total)
    }) {
        return 0;
    }
    let cancel = std::sync::atomic::AtomicBool::new(false);
    u8::from(super::fetch(&request.url, &request.part, &cancel).is_err())
}

/// Owns a fetch process, its lifeline, and its bounded metadata reader.
#[derive(Debug)]
pub(super) struct Transfer {
    child: Child,
    input: Option<ChildStdin>,
    metadata: Receiver<io::Result<Option<u64>>>,
    reader: Option<JoinHandle<()>>,
    header: Option<Metadata>,
    status: Option<ExitStatus>,
}

impl Transfer {
    pub(super) fn start(mut command: Command, url: &str, part: &Path) -> io::Result<Self> {
        if url.len() > 16 * 1024 || part.as_os_str().as_encoded_bytes().len() > 4096 {
            return Err(packet_error());
        }
        let request = Request {
            url: url.into(),
            part: part.into(),
        };
        let mut packet = serde_json::to_vec(&request).map_err(|_| packet_error())?;
        packet.push(b'\n');
        if packet.len() > REQUEST_BYTES {
            return Err(packet_error());
        }
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        let mut child = command.spawn()?;
        let input = child.stdin.take();
        let stdout = child.stdout.take();
        let (sender, metadata) = mpsc::channel();
        let mut transfer = Self {
            child,
            input,
            metadata,
            reader: None,
            header: None,
            status: None,
        };
        transfer
            .input
            .as_mut()
            .ok_or_else(packet_error)?
            .write_all(&packet)?;
        let stdout = stdout.ok_or_else(packet_error)?;
        transfer.reader = Some(std::thread::spawn(move || {
            let mut reader = io::BufReader::new(stdout);
            let result = read_metadata(&mut reader);
            let _ = sender.send(result);
            // Keep stdout open until exit, including test-harness trailers.
            let _ = io::copy(&mut reader, &mut io::sink());
        }));
        Ok(transfer)
    }

    pub(super) fn total(&self) -> Option<u64> {
        self.header.as_ref().and_then(|header| header.total)
    }

    pub(super) fn poll(&mut self) -> io::Result<bool> {
        if self.header.is_none() {
            match self.metadata.try_recv() {
                Ok(result) => self.header = Some(Metadata { total: result? }),
                Err(TryRecvError::Disconnected) => return Err(packet_error()),
                Err(TryRecvError::Empty) => {}
            }
        }
        if self.status.is_none() {
            self.status = self.child.try_wait()?;
        }
        if self.status.is_some_and(|status| !status.success()) {
            return Err(io::Error::other(
                "Model download failed; partial file retained",
            ));
        }
        Ok(self.status.is_some() && self.header.is_some())
    }

    pub(super) fn cancel(&mut self) -> io::Result<()> {
        self.input.take();
        if self.status.is_none() {
            // try_wait avoids treating an already finished worker as a kill error.
            self.status = self.child.try_wait()?;
            if self.status.is_none() {
                self.child.kill()?;
                self.status = Some(self.child.wait()?);
            }
        }
        if let Some(reader) = self.reader.take() {
            reader
                .join()
                .map_err(|_| io::Error::other("Download metadata reader panicked"))?;
        }
        Ok(())
    }
}

impl Drop for Transfer {
    fn drop(&mut self) {
        let _ = self.cancel();
    }
}

fn read_metadata(reader: &mut impl BufRead) -> io::Result<Option<u64>> {
    // The production worker writes one line; bounded test-harness noise may
    // precede it. Never accumulate or forward arbitrary worker stdout.
    for _ in 0..16 {
        let mut bytes = Vec::new();
        reader
            .take(META_BYTES as u64 + 1)
            .read_until(b'\n', &mut bytes)?;
        if bytes.len() > META_BYTES || bytes.last() != Some(&b'\n') {
            return Err(packet_error());
        }
        if let Some(packet) = bytes.strip_prefix(META_PREFIX.as_bytes()) {
            let metadata: Metadata = serde_json::from_slice(packet).map_err(|_| packet_error())?;
            return Ok(metadata.total);
        }
    }
    Err(packet_error())
}

pub(super) fn finalize(dest: &Path, total: Option<u64>) -> io::Result<()> {
    let expected = total.filter(|total| *total > 0).ok_or_else(|| {
        io::Error::other("Download has no verified content length; partial file retained")
    })?;
    let part = dest.with_extension("part");
    if std::fs::metadata(&part)?.len() != expected {
        return Err(io::Error::other(
            "Download is incomplete; partial file retained",
        ));
    }
    if dest.exists() {
        return Err(io::Error::other(
            "Download destination appeared during transfer",
        ));
    }
    std::fs::rename(part, dest)?;
    let _ = std::fs::remove_file(super::part_url_path(dest));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    const SECRET_URL: &str = "https://fixture.invalid/model?private=do-not-log";

    fn command(mode: &str) -> Command {
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args([
                "--exact",
                "download::foreground::tests::worker_probe",
                "--ignored",
                "--nocapture",
            ])
            .env("PLANK_FOREGROUND_TEST_MODE", mode);
        assert!(!format!("{command:?}").contains(SECRET_URL));
        command
    }

    fn wait_for(transfer: &mut Transfer, finished: bool) {
        let started = Instant::now();
        loop {
            let done = transfer.poll().unwrap();
            if (finished && done) || (!finished && transfer.header.is_some()) {
                return;
            }
            assert!(started.elapsed() < Duration::from_secs(3));
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    #[test]
    fn foreground_url_download_packets_are_bounded_and_redacted() {
        let request = Request {
            url: SECRET_URL.into(),
            part: PathBuf::from("/private/model.part"),
        };
        assert!(!format!("{request:?}").contains(SECRET_URL));
        let mut encoded = serde_json::to_vec(&request).unwrap();
        encoded.push(b'\n');
        assert_eq!(
            read_request(io::Cursor::new(encoded)).unwrap().url,
            SECRET_URL
        );
        for bytes in [
            vec![b'x'; REQUEST_BYTES + 1],
            b"invalid private=do-not-log\n".to_vec(),
            b"{}".to_vec(),
        ] {
            let error = read_request(io::Cursor::new(bytes)).unwrap_err();
            assert!(!error.to_string().contains("do-not-log"));
        }
        let oversized = format!("{META_PREFIX}{}\n", "x".repeat(META_BYTES));
        assert!(read_metadata(&mut io::Cursor::new(oversized)).is_err());
        assert_eq!(
            read_metadata(&mut io::Cursor::new(format!(
                "harness\n{META_PREFIX}{{\"total\":4}}\n"
            )))
            .unwrap(),
            Some(4)
        );
    }

    #[test]
    fn foreground_url_download_processes_complete_cancel_and_stop_on_lifeline_loss() {
        let root = crate::downloader::tests::tempdir();
        let part = root.join("model.part");
        let mut transfer = Transfer::start(command("complete"), SECRET_URL, &part).unwrap();
        wait_for(&mut transfer, true);
        assert_eq!(transfer.total(), Some(4));
        assert_eq!(std::fs::read(&part).unwrap(), b"data");
        transfer.cancel().unwrap();
        for lifeline in [false, true] {
            let mut transfer = Transfer::start(command("stall"), SECRET_URL, &part).unwrap();
            wait_for(&mut transfer, false);
            assert!(transfer.child.try_wait().unwrap().is_none());
            let started = Instant::now();
            if lifeline {
                transfer.input.take();
                while transfer.child.try_wait().unwrap().is_none() {
                    assert!(started.elapsed() < Duration::from_secs(3));
                    std::thread::sleep(Duration::from_millis(10));
                }
            }
            transfer.cancel().unwrap();
            assert!(started.elapsed() < Duration::from_secs(3));
            assert!(transfer.status.is_some());
            assert_eq!(std::fs::read(&part).unwrap(), b"data");
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn foreground_url_download_finalization_requires_complete_bytes_and_preserves_existing_files() {
        let root = crate::downloader::tests::tempdir();
        let dest = root.join("model.gguf");
        let part = dest.with_extension("part");
        std::fs::write(&part, b"data").unwrap();
        assert!(finalize(&dest, None).is_err());
        assert!(finalize(&dest, Some(5)).is_err());
        std::fs::write(&dest, b"existing").unwrap();
        assert!(finalize(&dest, Some(4)).is_err());
        assert_eq!(std::fs::read(&dest).unwrap(), b"existing");
        std::fs::remove_file(&dest).unwrap();
        std::fs::write(super::super::part_url_path(&dest), SECRET_URL).unwrap();
        finalize(&dest, Some(4)).unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), b"data");
        assert!(!part.exists());
        assert!(!super::super::part_url_path(&dest).exists());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    #[ignore = "Invoked by the owned subprocess regression"]
    fn worker_probe() {
        let Ok(mode) = std::env::var("PLANK_FOREGROUND_TEST_MODE") else {
            return;
        };
        let request = read_request(io::stdin().lock()).unwrap();
        assert!(std::env::args().all(|argument| !argument.contains("do-not-log")));
        watch_parent();
        send_metadata(Some(4)).unwrap();
        std::fs::write(request.part, b"data").unwrap();
        if mode == "stall" {
            loop {
                std::thread::sleep(Duration::from_millis(50));
            }
        }
    }
}
