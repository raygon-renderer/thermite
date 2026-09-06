//! Write-once, lock-free publication of a runtime detection result.

use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicU8, Ordering};

const UNINIT: u8 = 0;
const BUSY: u8 = 1;
const READY: u8 = 2;

/// Write-once cell: a state machine in an atomic guarding a single publish,
/// rather than a lock. `no_std`, and never parks.
///
/// Generic over the payload so the ISA detector and the machine-fact caches in
/// `thermite-cpu` share one implementation. Each is "run a short `cpuid`
/// sequence once, publish the result forever", and a second hand-rolled copy of
/// this is exactly the kind of thing that acquires a subtle ordering bug in only
/// one of its versions.
///
/// ```
/// use thermite::isa::DetectOnce;
///
/// static LANES: DetectOnce<u32> = DetectOnce::new(0);
/// fn detect() -> u32 { 8 }
///
/// assert_eq!(*LANES.get(detect), 8);
/// ```
pub struct DetectOnce<T: 'static> {
    state: AtomicU8,
    value: UnsafeCell<T>,
}

// SAFETY: `value` is written exactly once, by whichever thread wins the CAS to
// `BUSY`, and is only ever read after an `Acquire` load observes `READY`,
// which synchronizes with that writer's `Release` store.
unsafe impl<T: Send> Sync for DetectOnce<T> {}

impl<T> DetectOnce<T> {
    /// An unpublished cell. `placeholder` is never observable: it only exists so
    /// the cell can be a `static`, and is overwritten by the first `get`.
    pub const fn new(placeholder: T) -> Self {
        Self {
            state: AtomicU8::new(UNINIT),
            value: UnsafeCell::new(placeholder),
        }
    }

    /// The cached value, running `detect` exactly once across all threads.
    #[inline]
    pub fn get(&'static self, detect: fn() -> T) -> &'static T {
        // Fast path: already published by whoever won the race.
        if self.state.load(Ordering::Acquire) != READY {
            self.init(detect);
        }

        // SAFETY: the state is `READY`, reached through an `Acquire` load that
        // synchronizes with the writer's `Release` store, so the write has
        // completed and no writer can still be running. Nothing mutates it again.
        unsafe { &*self.value.get() }
    }

    #[inline(never)]
    fn init(&self, detect: fn() -> T) {
        match self
            .state
            .compare_exchange(UNINIT, BUSY, Ordering::AcqRel, Ordering::Acquire)
        {
            Ok(_) => {
                let detected = detect();
                // SAFETY: the CAS made this thread the unique writer, and no
                // reader can observe the cell until the store below publishes it.
                unsafe { *self.value.get() = detected };
                self.state.store(READY, Ordering::Release);
            }
            // Another thread is detecting. It is a short, lock-free, non-blocking
            // job (a handful of `cpuid`s), so spin rather than park.
            Err(BUSY) => {
                while self.state.load(Ordering::Acquire) != READY {
                    core::hint::spin_loop();
                }
            }
            // Already `READY`.
            Err(_) => {}
        }
    }
}
