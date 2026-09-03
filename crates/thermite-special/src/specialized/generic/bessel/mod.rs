//! The Bessel family and what is built on it.
//!
//! One directory, seven kernels, all sharing the two recurrence directions (`J`/`I` are the
//! minimal solutions and walk down on ratios, while `Y`/`K` are dominant and walk up) and the
//! envelope-relative accuracy contract the oscillating members need. Read
//! [`jy`](self::jy) first for that contract. It is the decision everything else is graded by.
//!
//! | module | what | order |
//! |---|---|---|
//! | [`ik`] | `$I_n$`, `$K_n$`: fitted rationals at 0 and 1, recurrences and an asymptotic arm above | whole, const or per-lane |
//! | [`jy`] | `$J_n$`, `$Y_n$`: the same shape for the oscillating pair | whole |
//! | [`half`] | all four at half-integer order, where they are elementary | `k/2` |
//! | [`spherical`] | `$j_n$`, `$y_n$`, `$i_n$`, `$k_n$`: [`half`]'s walks seeded in the spherical normalization | whole |
//! | [`jy_real`] | `$J_\nu$`, `$Y_\nu$` at arbitrary real order: series, Steed, Temme, Hankel | real |
//! | [`ik_real`] | `$I_\nu$`, `$K_\nu$` at arbitrary real order, generic over a real or complex argument | real |
//! | [`airy`] | `$\mathrm{Ai}$`, `$\mathrm{Bi}$` and derivatives, as Bessel functions at thirds | - |
//!
//! The order dispatch (which of these a runtime [`BesselOrder`](crate::BesselOrder) reaches)
//! is in the per-element impls (`specialized/pd.rs`, `ps.rs`). The entry-point stamping and
//! the [`BesselDetails`](crate::specialized::BesselDetails) trait are in
//! `specialized/bessel.rs`.

pub mod airy;
pub mod half;
pub mod ik;
pub mod ik_real;
pub mod jy;
pub mod jy_real;
pub mod ratio;
pub mod spherical;
