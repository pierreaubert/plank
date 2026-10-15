//! Owned remote-listener recipes and private Unix process handover.
//!
//! This module transfers credentials through an inherited anonymous socket.
//! Callers keep the returned descriptor number in an internal startup argument;
//! credentials never need command arguments, environment variables, or files.
//! Restoring the listener and startup questions remains the launcher's responsibility.

// Rust guideline compliant 2026-02-21
use std::{
    fmt,
    io::{self, Read, Write},
    net::SocketAddr,
    os::{
        fd::AsRawFd,
        unix::{net::UnixStream, process::CommandExt},
    },
    process::Command,
    time::{Duration, Instant},
};

use serde::{Deserialize, Serialize};

use super::control::ServerConfig;

// Small enough to send before exec without a draining thread; includes escaped JSON.
const PACKET_BYTES: usize = 4096;
// An inherited packet must already be complete, rather than stall model startup.
const READ_TIMEOUT: Duration = Duration::from_secs(5);

/// Internal startup argument containing a descriptor number, never a credential.
pub const DESCRIPTOR_ARG: &str = "--remote-restart-fd";

/// Consumes the private descriptor argument before launch arguments are recorded.
///
/// # Errors
/// Rejects duplicates, missing values, malformed numbers, and standard descriptors.
/// Validation errors leave `args` unchanged and omit argument contents.
pub fn take_descriptor_argument(args: &mut Vec<String>) -> io::Result<Option<i32>> {
    let mut descriptor = None;
    let mut retained = Vec::with_capacity(args.len());
    let mut index = 0;
    while index < args.len() {
        if args[index] == DESCRIPTOR_ARG {
            if descriptor.is_some() {
                return Err(invalid("Duplicate remote restart descriptor"));
            }
            let fd: i32 = args
                .get(index + 1)
                .and_then(|value| value.parse().ok())
                .filter(|fd| *fd >= 3)
                .ok_or_else(|| invalid("Invalid remote restart descriptor argument"))?;
            descriptor = Some(fd);
            index += 2;
        } else {
            retained.push(args[index].clone());
            index += 1;
        }
    }
    *args = retained;
    Ok(descriptor)
}

/// The listener recipe, excluding live grants, queued inputs, and replay state.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemoteRestart {
    version: u32,
    address: SocketAddr,
    token: String,
    local_present: bool,
    allow_control: bool,
    allowed_origins: Vec<String>,
    queue_max: usize,
}

impl fmt::Debug for RemoteRestart {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RemoteRestart")
            .field("address", &self.address)
            .field("local_present", &self.local_present)
            .field("allow_control", &self.allow_control)
            .field("origins", &self.allowed_origins.len())
            .field("queue_max", &self.queue_max)
            .finish_non_exhaustive()
    }
}

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

impl RemoteRestart {
    /// Captures a loopback listener's settings without preserving controller ownership.
    ///
    /// # Errors
    /// Returns invalid-listener or oversized-packet errors.
    pub fn new(address: SocketAddr, config: ServerConfig) -> io::Result<Self> {
        let record = Self {
            version: 1,
            address,
            token: config.token,
            local_present: config.local_present,
            allow_control: config.allow_control,
            allowed_origins: config.allowed_origins,
            queue_max: config.queue_max,
        };
        record.validate()?;
        record.encode()?;
        Ok(record)
    }

    fn validate(&self) -> io::Result<()> {
        if self.version != 1
            || !self.address.ip().is_loopback()
            || self.address.port() == 0
            || self.token.is_empty()
            || self.queue_max == 0
        {
            return Err(invalid("Invalid remote listener restart settings"));
        }
        Ok(())
    }

    fn encode(&self) -> io::Result<Vec<u8>> {
        let bytes = serde_json::to_vec(self)
            .map_err(|_| invalid("Cannot encode remote listener handover"))?;
        if bytes.len() > PACKET_BYTES {
            return Err(invalid("Remote listener handover exceeds 4 KiB"));
        }
        Ok(bytes)
    }

    /// Builds a fresh listener configuration, retaining its address and base policy.
    ///
    /// # Errors
    /// Returns invalid or oversized-recipe errors, including deserialized values.
    pub fn into_listener(self) -> io::Result<(SocketAddr, ServerConfig)> {
        self.validate()?;
        self.encode()?;
        Ok((
            self.address,
            ServerConfig {
                token: self.token,
                local_present: self.local_present,
                allow_control: self.allow_control,
                allowed_origins: self.allowed_origins,
                queue_max: self.queue_max,
            },
        ))
    }

    /// Restores the endpoint with fresh replay, input, and controller state.
    ///
    /// The caller owns the returned server during startup, then attaches it to
    /// the replacement agent. Failure never falls back to a different port or token.
    ///
    /// # Errors
    /// Returns recipe validation, listener binding, or accept-thread creation errors.
    pub fn restore(self) -> io::Result<super::RemoteServer> {
        let (address, config) = self.into_listener()?;
        super::RemoteServer::start(
            &address.to_string(),
            config,
            std::sync::Arc::new(crate::worker::BroadcastBus::new()),
            std::sync::Arc::new(crate::worker::TurnShared::default()),
        )
    }

    /// Prepares an anonymous, bounded packet for one replacement process.
    ///
    /// # Errors
    /// Returns validation, socket creation, or packet-write errors.
    pub fn prepare(&self) -> io::Result<PreparedRestart> {
        self.validate()?;
        let bytes = self.encode()?;
        let (mut sender, receiver) = UnixStream::pair()?;
        sender.set_write_timeout(Some(READ_TIMEOUT))?;
        sender.write_all(&bytes)?;
        drop(sender);
        Ok(PreparedRestart { receiver })
    }
}

/// An owned handover socket; Debug never exposes the packet.
#[derive(Debug)]
pub struct PreparedRestart {
    receiver: UnixStream,
}

impl PreparedRestart {
    /// Keeps a private descriptor alive in `command` until a child is spawned.
    ///
    /// The returned number can be passed as an internal startup argument. Dropping
    /// this object before launching is safe: the command owns its own descriptor.
    /// Only that descriptor loses close-on-exec in the child. Use [`Self::exec`]
    /// for direct process replacement; this spawn hook refuses in-place exec.
    ///
    /// # Errors
    /// Returns descriptor-cloning errors. Launch errors leave parent descriptors scoped.
    pub fn configure(&self, command: &mut Command) -> io::Result<i32> {
        self.install(command, false)
    }

    /// Replaces this process with an owned command carrying the private recipe.
    ///
    /// Adds the internal descriptor argument. Returns only on failure, after
    /// dropping the command and its private descriptor. Call after stopping the
    /// old listener, workers, and engine resources.
    ///
    /// # Errors
    /// Returns descriptor-preparation or exec errors without exposing the recipe.
    #[must_use]
    pub fn exec(&self, mut command: Command) -> io::Error {
        let fd = match self.install(&mut command, true) {
            Ok(fd) => fd,
            Err(error) => return error,
        };
        command.arg(DESCRIPTOR_ARG).arg(fd.to_string());
        let error = command.exec();
        drop(command);
        error
    }

    fn install(&self, command: &mut Command, allow_exec: bool) -> io::Result<i32> {
        let channel = self.receiver.try_clone()?;
        let fd = channel.as_raw_fd();
        // SAFETY: getpid reads process metadata and has no memory preconditions.
        let parent = unsafe { libc::getpid() };
        // SAFETY: the closure owns the UnixStream, so the fd cannot be closed or
        // reused before launch. It calls only async-signal-safe fcntl operations;
        // it allocates no Rust objects and takes no locks after fork.
        unsafe {
            command.pre_exec(move || {
                if !allow_exec && libc::getpid() == parent {
                    return Err(io::Error::from_raw_os_error(libc::EINVAL));
                }
                let fd = channel.as_raw_fd();
                let flags = libc::fcntl(fd, libc::F_GETFD);
                if flags < 0 || libc::fcntl(fd, libc::F_SETFD, flags & !libc::FD_CLOEXEC) < 0 {
                    return Err(io::Error::last_os_error());
                }
                Ok(())
            });
        }
        Ok(fd)
    }
}

/// Reads an inherited packet without assuming ownership of the caller's descriptor.
///
/// This duplicates a verified Unix stream and marks the original close-on-exec.
/// A launcher consumes its numeric startup argument before recording restart args.
/// The stream's prior blocking mode is restored after reading. No environment
/// mutation or controller lease restoration happens here.
///
/// # Errors
/// Returns descriptor, timeout, malformed-packet, or validation errors. Errors omit credentials.
pub fn read_inherited(fd: i32) -> io::Result<RemoteRestart> {
    let mut receiver = duplicate_stream(fd).map_err(|error| {
        io::Error::new(error.kind(), format!("Remote restart descriptor: {error}"))
    })?;
    // Darwin Unix sockets can refuse SO_RCVTIMEO; nonblocking reads plus poll
    // avoid relying on that option while preserving a bounded startup wait.
    // SAFETY: receiver owns a live descriptor; F_GETFL does not access user memory.
    let flags = unsafe { libc::fcntl(receiver.as_raw_fd(), libc::F_GETFL) };
    if flags < 0 {
        return Err(io::Error::last_os_error());
    }
    let was_nonblocking = flags & libc::O_NONBLOCK != 0;
    receiver.set_nonblocking(true)?;
    let result = decode(&mut DeadlineReader {
        receiver: &mut receiver,
        deadline: Instant::now() + READ_TIMEOUT,
    });
    let restored = receiver.set_nonblocking(was_nonblocking);
    let record = result?;
    restored?;
    Ok(record)
}

struct DeadlineReader<'a> {
    receiver: &'a mut UnixStream,
    deadline: Instant,
}

impl Read for DeadlineReader<'_> {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        loop {
            let remaining = self
                .deadline
                .checked_duration_since(Instant::now())
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::TimedOut,
                        "Remote restart packet did not finish",
                    )
                })?;
            match self.receiver.read(bytes) {
                Ok(count) => return Ok(count),
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {}
                Err(error) => return Err(error),
            }
            let timeout = i32::try_from(remaining.as_millis())
                .unwrap_or(i32::MAX)
                .max(1);
            let mut wait = libc::pollfd {
                fd: self.receiver.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            };
            // SAFETY: wait is live, correctly sized storage; the stream remains
            // owned for the entire poll. No pointer survives this call.
            if unsafe { libc::poll(&raw mut wait, 1, timeout) } < 0 {
                let error = io::Error::last_os_error();
                if error.kind() != io::ErrorKind::Interrupted {
                    return Err(error);
                }
            }
        }
    }
}

fn decode(reader: &mut impl Read) -> io::Result<RemoteRestart> {
    let mut bytes = Vec::new();
    reader
        .take(u64::try_from(PACKET_BYTES + 1).expect("packet bound fits u64"))
        .read_to_end(&mut bytes)?;
    if bytes.len() > PACKET_BYTES {
        return Err(invalid("Remote listener handover exceeds 4 KiB"));
    }
    let record: RemoteRestart = serde_json::from_slice(&bytes)
        .map_err(|_| invalid("Malformed remote listener handover"))?;
    record.validate()?;
    Ok(record)
}

fn duplicate_stream(fd: i32) -> io::Result<UnixStream> {
    use std::os::fd::FromRawFd;
    if fd < 3 {
        return Err(invalid(
            "Remote listener handover requires a private descriptor",
        ));
    }
    // SAFETY: fcntl does not dereference user memory. On success it creates an
    // independent owned descriptor; failure creates none and returns an OS error.
    let copied = unsafe { libc::fcntl(fd, libc::F_DUPFD_CLOEXEC, 3) };
    if copied < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: copied is a new live descriptor owned only here. File accepts any
    // open descriptor, so malformed caller fds cannot violate typed I/O ownership.
    let owned = unsafe { std::fs::File::from_raw_fd(copied) };
    let mut socket_type = 0_i32;
    let mut type_length = libc::socklen_t::try_from(std::mem::size_of_val(&socket_type))
        .expect("socket option fits socklen_t");
    // SAFETY: sockaddr_storage contains only integer/byte fields; zero is valid.
    let mut address: libc::sockaddr_storage = unsafe { std::mem::zeroed() };
    let mut address_length = libc::socklen_t::try_from(std::mem::size_of_val(&address))
        .expect("socket address fits socklen_t");
    // SAFETY: both output buffers and lengths point to live, correctly sized
    // storage. The owned descriptor remains open for the entire validation.
    let valid = unsafe {
        libc::getsockopt(
            copied,
            libc::SOL_SOCKET,
            libc::SO_TYPE,
            (&raw mut socket_type).cast(),
            &raw mut type_length,
        ) == 0
            && libc::getsockname(copied, (&raw mut address).cast(), &raw mut address_length) == 0
    };
    if !valid || socket_type != libc::SOCK_STREAM || i32::from(address.ss_family) != libc::AF_UNIX {
        return Err(invalid("Remote listener handover requires a Unix stream"));
    }
    // SAFETY: fd is still owned by the caller; fcntl only changes descriptor flags.
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFD) };
    // SAFETY: the descriptor-flags update neither closes fd nor takes caller ownership.
    let marked =
        flags >= 0 && unsafe { libc::fcntl(fd, libc::F_SETFD, flags | libc::FD_CLOEXEC) } >= 0;
    if !marked {
        return Err(io::Error::last_os_error());
    }
    Ok(UnixStream::from(std::os::fd::OwnedFd::from(owned)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> RemoteRestart {
        RemoteRestart::new(
            "127.0.0.1:48731".parse().unwrap(),
            ServerConfig {
                token: "private-handover-token".into(),
                local_present: true,
                allow_control: false,
                allowed_origins: vec!["http://localhost:8080".into()],
                queue_max: 65536,
            },
        )
        .unwrap()
    }

    #[test]
    fn restart_handover_bounds_packets_and_redacts_credentials() {
        let record = fixture();
        assert!(!format!("{record:?}").contains("private-handover-token"));
        let (_, config) = record.clone().into_listener().unwrap();
        assert!(!format!("{config:?}").contains("private-handover-token"));
        assert_eq!(
            decode(&mut record.encode().unwrap().as_slice()).unwrap(),
            record
        );
        for (field, value) in [
            ("version", serde_json::json!(2)),
            ("address", serde_json::json!("0.0.0.0:48731")),
            ("token", serde_json::json!("")),
            ("queue_max", serde_json::json!(0)),
        ] {
            let mut malformed = serde_json::to_value(&record).unwrap();
            malformed[field] = value;
            let error =
                decode(&mut serde_json::to_vec(&malformed).unwrap().as_slice()).unwrap_err();
            assert!(!format!("{error:?}").contains("private-handover-token"));
        }
        assert!(decode(&mut vec![b'x'; PACKET_BYTES + 1].as_slice()).is_err());
        assert!(decode(&mut b"truncated secret".as_slice()).is_err());
        let mut record = fixture();
        record.token = "x".repeat(PACKET_BYTES);
        assert!(record.prepare().is_err());
    }

    #[test]
    fn restart_handover_refuses_standard_closed_and_non_socket_descriptors() {
        for fd in [-1, 0, 1, 2, i32::MAX] {
            assert!(read_inherited(fd).is_err());
        }
        let file = std::fs::File::open("Cargo.toml").unwrap();
        assert!(read_inherited(file.as_raw_fd()).is_err());
        assert!(
            file.metadata().is_ok(),
            "validation must not take caller ownership"
        );
    }

    #[test]
    fn restart_handover_survives_dropping_the_prepared_packet_before_child_launch() {
        let prepared = fixture().prepare().unwrap();
        let mut child = Command::new(std::env::current_exe().unwrap());
        child.args([
            "--exact",
            "remote::restart::tests::handover_child_probe",
            "--ignored",
            "--nocapture",
        ]);
        let fd = prepared.configure(&mut child).unwrap();
        child.env("PLANK_HANDOVER_TEST_FD", fd.to_string());
        drop(prepared);
        assert!(!format!("{child:?}").contains("private-handover-token"));
        let output = child.output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[test]
    fn restart_handover_rejects_stalled_packets_with_a_deadline() {
        let (_sender, mut receiver) = UnixStream::pair().unwrap();
        receiver.set_nonblocking(true).unwrap();
        let error = decode(&mut DeadlineReader {
            receiver: &mut receiver,
            deadline: Instant::now() + Duration::from_millis(20),
        })
        .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::TimedOut);
    }

    #[test]
    fn restart_handover_failed_spawn_and_exec_keep_the_parent_packet_owned() {
        let prepared = fixture().prepare().unwrap();
        let mut missing = Command::new("/nonexistent/plank-restart-fixture");
        let copied = prepared.configure(&mut missing).unwrap();
        assert!(missing.spawn().is_err());
        // SAFETY: copied is owned by the still-live command; F_GETFD is metadata-only.
        assert_ne!(
            unsafe { libc::fcntl(copied, libc::F_GETFD) } & libc::FD_CLOEXEC,
            0
        );
        drop(missing);
        // SAFETY: the parent receiver remains owned here; fcntl only reads flags.
        let flags = unsafe { libc::fcntl(prepared.receiver.as_raw_fd(), libc::F_GETFD) };
        let error = prepared.exec(Command::new("/nonexistent/plank-restart-fixture"));
        assert_eq!(error.kind(), io::ErrorKind::NotFound);
        // SAFETY: the original receiver is still owned by prepared after failed exec.
        assert_eq!(
            unsafe { libc::fcntl(prepared.receiver.as_raw_fd(), libc::F_GETFD) },
            flags
        );
        assert_eq!(
            read_inherited(prepared.receiver.as_raw_fd()).unwrap(),
            fixture()
        );
    }

    #[test]
    fn restart_handover_arguments_are_consumed_once_without_altering_other_launch_options() {
        let mut args = vec![
            "--model:current".into(),
            DESCRIPTOR_ARG.into(),
            "7".into(),
            "/resume".into(),
            "saved".into(),
        ];
        assert_eq!(take_descriptor_argument(&mut args).unwrap(), Some(7));
        assert_eq!(args, ["--model:current", "/resume", "saved"]);
        assert_eq!(take_descriptor_argument(&mut args).unwrap(), None);
        for mut invalid_args in [
            vec![DESCRIPTOR_ARG.into()],
            vec![DESCRIPTOR_ARG.into(), "2".into()],
            vec![DESCRIPTOR_ARG.into(), "secret-invalid-value".into()],
            vec![
                DESCRIPTOR_ARG.into(),
                "7".into(),
                DESCRIPTOR_ARG.into(),
                "8".into(),
            ],
        ] {
            let before = invalid_args.clone();
            let error = take_descriptor_argument(&mut invalid_args).unwrap_err();
            assert_eq!(invalid_args, before);
            assert!(!error.to_string().contains("secret-invalid-value"));
        }
    }

    #[test]
    fn restart_handover_spawn_hook_refuses_in_place_exec_without_exposing_the_descriptor() {
        let prepared = fixture().prepare().unwrap();
        let mut command = Command::new("/nonexistent/plank-restart-fixture");
        let fd = prepared.configure(&mut command).unwrap();
        let error = command.exec();
        assert_eq!(error.raw_os_error(), Some(libc::EINVAL));
        // SAFETY: command still owns this descriptor; F_GETFD is metadata-only.
        let flags = unsafe { libc::fcntl(fd, libc::F_GETFD) };
        assert!(flags >= 0 && flags & libc::FD_CLOEXEC != 0);
    }

    #[test]
    #[ignore = "invoked by the handover process test with an inherited socket"]
    fn handover_child_probe() {
        let Some(fd) = std::env::var("PLANK_HANDOVER_TEST_FD").ok() else {
            return;
        };
        let record = read_inherited(fd.parse().unwrap()).unwrap();
        assert_eq!(record, fixture());
    }
}
