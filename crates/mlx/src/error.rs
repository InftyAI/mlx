//! Error handling for MLX operations.
//!
//! MLX's C API reports failures two ways: an operation returns a non-zero
//! status code, and it routes a message to a globally-installed error handler.
//! The *default* handler prints to stderr and calls `exit(-1)` — fatal for a
//! library. On first use we install our own handler that captures the message
//! into thread-local storage instead, so failures surface as [`Error`] values.

use std::cell::RefCell;
use std::ffi::{CStr, c_char, c_void};
use std::fmt;
use std::ptr;
use std::sync::Once;

use mlxr_sys as sys;

/// An error returned by an MLX operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    message: String,
}

impl Error {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    /// The message MLX reported for this failure.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MLX error: {}", self.message)
    }
}

impl std::error::Error for Error {}

/// A `Result` whose error type is an MLX [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

thread_local! {
    /// The most recent message from the MLX error handler, on this thread.
    static LAST_ERROR: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// The handler MLX invokes on failure. Called synchronously on the same thread
/// as the failing op, so thread-local capture is race-free.
unsafe extern "C" fn error_handler(msg: *const c_char, _data: *mut c_void) {
    if msg.is_null() {
        return;
    }
    // SAFETY: mlx passes a valid, NUL-terminated C string.
    let message = unsafe { CStr::from_ptr(msg) }
        .to_string_lossy()
        .into_owned();
    LAST_ERROR.with(|e| *e.borrow_mut() = Some(message));
}

static INSTALL: Once = Once::new();

/// Installs our capturing error handler, replacing MLX's fatal default.
///
/// Idempotent and cheap to call before every operation.
pub(crate) fn install() {
    INSTALL.call_once(|| {
        // SAFETY: `error_handler` has the required C signature; no user data or
        // destructor is needed.
        unsafe {
            sys::mlx_set_error_handler(Some(error_handler), ptr::null_mut(), None);
        }
    });
}

/// Removes and returns the last captured error message on this thread.
pub(crate) fn take() -> Option<String> {
    LAST_ERROR.with(|e| e.borrow_mut().take())
}

/// Converts a status code into a `Result`, attaching the captured message.
///
/// A non-zero `status` means the op failed; the message (if any) comes from the
/// handler that fired during the call.
///
/// Defensively ensures our handler is installed. This cannot rescue the op
/// whose `status` we are checking — that call has already returned — but it
/// guarantees any op reaching this central conversion point leaves the
/// process-wide handler in the non-fatal state, so MLX's default `exit(-1)`
/// handler can never linger even if a call site forgets to call [`install`].
pub(crate) fn check(status: i32) -> Result<()> {
    install();
    if status == 0 {
        // Clear any message left by an earlier call so it can never be
        // misattributed to a later failure whose handler did not fire.
        take();
        Ok(())
    } else {
        Err(Error::new(
            take().unwrap_or_else(|| "unknown MLX error".to_string()),
        ))
    }
}
