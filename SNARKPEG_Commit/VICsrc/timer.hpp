#pragma once

#include <assert.h>
#include <chrono>

class timer {
public:
    timer() : total_time_sec(0.0), status(false) {}

    void start();
    void stop();
    void clear() { total_time_sec = 0.0; status = false; }
    double elapse_sec() const {
        assert(status == false);
        return total_time_sec;
    }

private:
    std::chrono::high_resolution_clock::time_point t0;
    double total_time_sec;
    bool status;
};
