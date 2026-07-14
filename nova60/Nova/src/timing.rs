//! Timing helpers for aggregating recursive proving costs.

use std::sync::{Mutex, OnceLock};
use std::time::Duration;

#[derive(Clone, Copy, Debug, Default)]
struct RecursiveTimingTotals {
  compute_w: f64,
  compute_t: f64,
  commit_w: f64,
  commit_t: f64,
  neutron_commit_e: f64,
  neutron_multiply_vec_z1: f64,
  neutron_multiply_vec_z2: f64,
  neutron_prove_helper: f64,
  neutron_poly_finalize: f64,
  neutron_fold_instance: f64,
  neutron_fold_witness: f64,
  neutron_nifs_total: f64,
  neutron_augmented_synthesize: f64,
  neutron_extract_instance_witness: f64,
  neutron_prove_step_total: f64,
  neutron_new_base_total: f64,
}

/// A snapshot of the accumulated recursive proving timings.
#[derive(Clone, Copy, Debug, Default)]
pub struct RecursiveTimingSnapshot {
  /// Total witness-generation time in seconds.
  pub compute_w: f64,
  /// Total cross-term computation time in seconds.
  pub compute_t: f64,
  /// Total witness commitment time in seconds.
  pub commit_w: f64,
  /// Total cross-term commitment time in seconds.
  pub commit_t: f64,
  /// Total Neutron eq-polynomial commitment time in seconds.
  pub neutron_commit_e: f64,
  /// Total Neutron `multiply_vec` time for the folded running relation in seconds.
  pub neutron_multiply_vec_z1: f64,
  /// Total Neutron `multiply_vec` time for the fresh local relation in seconds.
  pub neutron_multiply_vec_z2: f64,
  /// Total Neutron sumcheck helper time in seconds.
  pub neutron_prove_helper: f64,
  /// Total Neutron polynomial finalization time in seconds.
  pub neutron_poly_finalize: f64,
  /// Total Neutron folded-instance update time in seconds.
  pub neutron_fold_instance: f64,
  /// Total Neutron folded-witness update time in seconds.
  pub neutron_fold_witness: f64,
  /// Total Neutron `NIFS::prove` time in seconds.
  pub neutron_nifs_total: f64,
  /// Total Neutron augmented-circuit synthesis time in seconds.
  pub neutron_augmented_synthesize: f64,
  /// Total Neutron instance/witness extraction time in seconds.
  pub neutron_extract_instance_witness: f64,
  /// Total Neutron `prove_step` time in seconds.
  pub neutron_prove_step_total: f64,
  /// Total Neutron base-case construction time in seconds.
  pub neutron_new_base_total: f64,
}

fn recursive_timing() -> &'static Mutex<RecursiveTimingTotals> {
  static RECURSIVE_TIMING: OnceLock<Mutex<RecursiveTimingTotals>> = OnceLock::new();
  RECURSIVE_TIMING.get_or_init(|| Mutex::new(RecursiveTimingTotals::default()))
}

/// Clears the accumulated recursive timing totals.
pub fn reset_recursive_timing() {
  let mut totals = recursive_timing()
    .lock()
    .expect("recursive timing mutex poisoned");
  *totals = RecursiveTimingTotals::default();
}

/// Returns the current recursive timing totals.
pub fn snapshot_recursive_timing() -> RecursiveTimingSnapshot {
  let totals = *recursive_timing()
    .lock()
    .expect("recursive timing mutex poisoned");
  RecursiveTimingSnapshot {
    compute_w: totals.compute_w,
    compute_t: totals.compute_t,
    commit_w: totals.commit_w,
    commit_t: totals.commit_t,
    neutron_commit_e: totals.neutron_commit_e,
    neutron_multiply_vec_z1: totals.neutron_multiply_vec_z1,
    neutron_multiply_vec_z2: totals.neutron_multiply_vec_z2,
    neutron_prove_helper: totals.neutron_prove_helper,
    neutron_poly_finalize: totals.neutron_poly_finalize,
    neutron_fold_instance: totals.neutron_fold_instance,
    neutron_fold_witness: totals.neutron_fold_witness,
    neutron_nifs_total: totals.neutron_nifs_total,
    neutron_augmented_synthesize: totals.neutron_augmented_synthesize,
    neutron_extract_instance_witness: totals.neutron_extract_instance_witness,
    neutron_prove_step_total: totals.neutron_prove_step_total,
    neutron_new_base_total: totals.neutron_new_base_total,
  }
}

/// Adds witness-generation time to the recursive totals.
pub fn record_compute_w(elapsed: Duration) {
  let mut totals = recursive_timing()
    .lock()
    .expect("recursive timing mutex poisoned");
  totals.compute_w += elapsed.as_secs_f64();
}

/// Adds cross-term computation time to the recursive totals.
pub fn record_compute_t(elapsed: Duration) {
  let mut totals = recursive_timing()
    .lock()
    .expect("recursive timing mutex poisoned");
  totals.compute_t += elapsed.as_secs_f64();
}

/// Adds witness commitment time to the recursive totals.
pub fn record_commit_w(elapsed: Duration) {
  let mut totals = recursive_timing()
    .lock()
    .expect("recursive timing mutex poisoned");
  totals.commit_w += elapsed.as_secs_f64();
}

/// Adds cross-term commitment time to the recursive totals.
pub fn record_commit_t(elapsed: Duration) {
  let mut totals = recursive_timing()
    .lock()
    .expect("recursive timing mutex poisoned");
  totals.commit_t += elapsed.as_secs_f64();
}

/// Adds Neutron eq-polynomial commitment time to the recursive totals.
pub fn record_neutron_commit_e(elapsed: Duration) {
  let mut totals = recursive_timing()
    .lock()
    .expect("recursive timing mutex poisoned");
  totals.neutron_commit_e += elapsed.as_secs_f64();
}

/// Adds Neutron `multiply_vec` time for the folded running relation to the recursive totals.
pub fn record_neutron_multiply_vec_z1(elapsed: Duration) {
  let mut totals = recursive_timing()
    .lock()
    .expect("recursive timing mutex poisoned");
  totals.neutron_multiply_vec_z1 += elapsed.as_secs_f64();
}

/// Adds Neutron `multiply_vec` time for the fresh local relation to the recursive totals.
pub fn record_neutron_multiply_vec_z2(elapsed: Duration) {
  let mut totals = recursive_timing()
    .lock()
    .expect("recursive timing mutex poisoned");
  totals.neutron_multiply_vec_z2 += elapsed.as_secs_f64();
}

/// Adds Neutron sumcheck helper time to the recursive totals.
pub fn record_neutron_prove_helper(elapsed: Duration) {
  let mut totals = recursive_timing()
    .lock()
    .expect("recursive timing mutex poisoned");
  totals.neutron_prove_helper += elapsed.as_secs_f64();
}

/// Adds Neutron polynomial finalization time to the recursive totals.
pub fn record_neutron_poly_finalize(elapsed: Duration) {
  let mut totals = recursive_timing()
    .lock()
    .expect("recursive timing mutex poisoned");
  totals.neutron_poly_finalize += elapsed.as_secs_f64();
}

/// Adds Neutron folded-instance update time to the recursive totals.
pub fn record_neutron_fold_instance(elapsed: Duration) {
  let mut totals = recursive_timing()
    .lock()
    .expect("recursive timing mutex poisoned");
  totals.neutron_fold_instance += elapsed.as_secs_f64();
}

/// Adds Neutron folded-witness update time to the recursive totals.
pub fn record_neutron_fold_witness(elapsed: Duration) {
  let mut totals = recursive_timing()
    .lock()
    .expect("recursive timing mutex poisoned");
  totals.neutron_fold_witness += elapsed.as_secs_f64();
}

/// Adds total Neutron `NIFS::prove` time to the recursive totals.
pub fn record_neutron_nifs_total(elapsed: Duration) {
  let mut totals = recursive_timing()
    .lock()
    .expect("recursive timing mutex poisoned");
  totals.neutron_nifs_total += elapsed.as_secs_f64();
}

/// Adds Neutron augmented-circuit synthesis time to the recursive totals.
pub fn record_neutron_augmented_synthesize(elapsed: Duration) {
  let mut totals = recursive_timing()
    .lock()
    .expect("recursive timing mutex poisoned");
  totals.neutron_augmented_synthesize += elapsed.as_secs_f64();
}

/// Adds Neutron instance/witness extraction time to the recursive totals.
pub fn record_neutron_extract_instance_witness(elapsed: Duration) {
  let mut totals = recursive_timing()
    .lock()
    .expect("recursive timing mutex poisoned");
  totals.neutron_extract_instance_witness += elapsed.as_secs_f64();
}

/// Adds total Neutron `prove_step` time to the recursive totals.
pub fn record_neutron_prove_step_total(elapsed: Duration) {
  let mut totals = recursive_timing()
    .lock()
    .expect("recursive timing mutex poisoned");
  totals.neutron_prove_step_total += elapsed.as_secs_f64();
}

/// Adds total Neutron base-case construction time to the recursive totals.
pub fn record_neutron_new_base_total(elapsed: Duration) {
  let mut totals = recursive_timing()
    .lock()
    .expect("recursive timing mutex poisoned");
  totals.neutron_new_base_total += elapsed.as_secs_f64();
}
