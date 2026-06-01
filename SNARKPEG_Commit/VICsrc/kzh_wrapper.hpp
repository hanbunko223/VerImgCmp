#pragma once

#include "global_var.hpp"

struct KzhMetrics {
    double prover_time_sec = 0.0;
    double verifier_time_sec = 0.0;
    double proof_size_kb = 0.0;
};

bool run_kzh_pcs(const vector<F> &poly,
                 const vector<F> &point,
                 const F &value,
                 u8 num_vars,
                 KzhMetrics &metrics);
