#include "timer.hpp"

void timer::start() {
    assert(status == false);
    t0 = std::chrono::high_resolution_clock::now();
    status = true;
}

void timer::stop() {
    assert(status == true);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto time_span_sec = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0);
    total_time_sec += time_span_sec.count();
    status = false;
}
