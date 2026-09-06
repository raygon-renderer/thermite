//! Apple (macOS / iOS) thread scheduling hints.
//!
//! Apple silicon is hybrid, but unlike x86 it exposes no way to ask *which*
//! core the calling thread is on (see [`current_core_type`]). The platform's
//! model is the other way round: you declare the **intent** of the work with a
//! quality-of-service class, and the scheduler decides where to run it. On
//! Apple silicon that decision includes P-core versus E-core placement --
//! [`QosClass::Background`] work is confined to the efficiency cores.
//!
//! So this is the closest thing to core-type *control* the platform offers, and
//! it is the counterpart to the read-only [`Topology`] counts.
//!
//! `pthread` lives in libSystem, which every Apple target links
//! unconditionally, so none of this needs `std` or a `libc` dependency.
//!
//! [`current_core_type`]: super::current_core_type
//! [`Topology`]: super::Topology

use core::ffi::{c_int, c_void};

unsafe extern "C" {
    fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: c_int) -> c_int;
    fn pthread_get_qos_class_np(thread: *mut c_void, qos_class: *mut u32, relative_priority: *mut c_int) -> c_int;
    fn pthread_self() -> *mut c_void;
}

/// Quality-of-service class: what the work is *for*, from which the scheduler
/// derives CPU/IO priority, timer coalescing, and -- on Apple silicon -- which
/// cluster it lands on.
///
/// Discriminants are the ABI values from `<sys/qos.h>`, ordered from most to
/// least latency-sensitive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u32)]
pub enum QosClass {
    /// Work the user is actively waiting on: a frame, a UI event. Highest
    /// priority, and never the right answer for background compute.
    UserInteractive = 0x21,
    /// Work the user initiated and is waiting for, but not frame-by-frame.
    UserInitiated = 0x19,
    /// The default when a thread expresses no intent.
    Default = 0x15,
    /// Long-running work with a progress indicator. Energy-efficient
    /// scheduling; the usual choice for a compute pool.
    Utility = 0x11,
    /// Work the user is not aware of. **On Apple silicon this confines the
    /// thread to the efficiency cores**, which is the one QoS class that
    /// amounts to an explicit core-type request.
    Background = 0x09,
    /// Opted out of the QoS system. Can be *returned* by [`thread_qos`]; it is
    /// not a legal argument to [`set_thread_qos`].
    Unspecified = 0x00,
}

impl QosClass {
    /// Reconstruct from the raw ABI value, or `None` if libSystem returned a
    /// class this build does not know.
    const fn from_raw(raw: u32) -> Option<Self> {
        Some(match raw {
            0x21 => Self::UserInteractive,
            0x19 => Self::UserInitiated,
            0x15 => Self::Default,
            0x11 => Self::Utility,
            0x09 => Self::Background,
            0x00 => Self::Unspecified,
            _ => return None,
        })
    }
}

/// The most negative `relative_priority` accepted by [`set_thread_qos`].
///
/// Relative priority is an offset *below* the maximum for the class, so the
/// legal range is `MIN_RELATIVE_PRIORITY..=0`.
pub const MIN_RELATIVE_PRIORITY: c_int = -15;

/// Why a QoS request was refused. Wraps the `errno`-style code returned by
/// `pthread`, which does not set `errno` itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QosError(c_int);

impl QosError {
    /// The raw error code: `EINVAL` (22) for a bad class or out-of-range
    /// priority, `EPERM` (1) if the thread has permanently opted out of the QoS
    /// system -- which a thread taken from a dispatch work queue has.
    #[inline]
    pub const fn code(self) -> c_int {
        self.0
    }
}

impl core::fmt::Display for QosError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "failed to set thread QoS class (error {})", self.0)
    }
}

impl core::error::Error for QosError {}

/// Request a QoS class for the **calling thread**.
///
/// `relative_priority` is an offset below the maximum priority within the
/// class, so it must be in `MIN_RELATIVE_PRIORITY..=0`; pass `0` unless you have
/// a reason not to. Out-of-range values are rejected here rather than handed to
/// the OS.
///
/// This is a *request*: it takes effect for work the thread does from here on,
/// and the scheduler remains free to interpret it. It cannot raise a thread that
/// has opted out of QoS (`EPERM`).
///
/// ```no_run
/// use thermite::cpu::apple::{QosClass, set_thread_qos};
///
/// // Confine a background rebuild to the efficiency cores.
/// set_thread_qos(QosClass::Background, 0).unwrap();
/// ```
pub fn set_thread_qos(class: QosClass, relative_priority: c_int) -> Result<(), QosError> {
    const EINVAL: c_int = 22;

    // Rejected by the OS anyway; caught here so the error is unambiguous.
    if matches!(class, QosClass::Unspecified) || !(MIN_RELATIVE_PRIORITY..=0).contains(&relative_priority) {
        return Err(QosError(EINVAL));
    }

    // SAFETY: affects only the calling thread's scheduling parameters, touches
    // no memory, and both arguments are validated above.
    match unsafe { pthread_set_qos_class_self_np(class as u32, relative_priority) } {
        0 => Ok(()),
        code => Err(QosError(code)),
    }
}

/// The calling thread's current QoS class and relative priority.
///
/// `None` if the query failed or libSystem reported a class this build does not
/// recognise. [`QosClass::Unspecified`] means the thread is opted out.
pub fn thread_qos() -> Option<(QosClass, c_int)> {
    let mut class = 0u32;
    let mut relative_priority = 0 as c_int;

    // SAFETY: both out-pointers are valid for the writes libSystem performs, and
    // `pthread_self()` is always a valid thread handle for the caller.
    let rc = unsafe { pthread_get_qos_class_np(pthread_self(), &raw mut class, &raw mut relative_priority) };

    (rc == 0).then(|| Some((QosClass::from_raw(class)?, relative_priority)))?
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_values_round_trip() {
        for class in [
            QosClass::UserInteractive,
            QosClass::UserInitiated,
            QosClass::Default,
            QosClass::Utility,
            QosClass::Background,
            QosClass::Unspecified,
        ] {
            assert_eq!(QosClass::from_raw(class as u32), Some(class));
        }
        assert_eq!(QosClass::from_raw(0xdead), None);

        // Ordering is most- to least-latency-sensitive.
        assert!(QosClass::UserInteractive > QosClass::Background);
    }

    #[test]
    fn rejects_invalid_requests() {
        assert!(set_thread_qos(QosClass::Unspecified, 0).is_err());
        assert!(set_thread_qos(QosClass::Utility, 1).is_err());
        assert!(set_thread_qos(QosClass::Utility, MIN_RELATIVE_PRIORITY - 1).is_err());
    }

    #[test]
    fn set_and_read_back() {
        // A thread from a dispatch work queue is permanently opted out, so a
        // refusal is legitimate; only a *wrong* readback is a failure.
        if set_thread_qos(QosClass::Utility, 0).is_ok() {
            assert_eq!(thread_qos(), Some((QosClass::Utility, 0)));
        }
    }
}
