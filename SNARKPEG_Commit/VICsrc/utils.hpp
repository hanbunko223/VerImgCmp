//
// Created by 69029 on 3/9/2021.
//

#ifndef ZKCNN_UTILS_HPP
#define ZKCNN_UTILS_HPP

#include "circuit.h"
#include <queue>
#include <mutex>
#include <condition_variable>

int ceilPow2BitLengthSigned(double n);
int floorPow2BitLengthSigned(double n);

char ceilPow2BitLength(u32 n);
char floorPow2BitLength(u32 n);


void fft(vector<F> &arr, int logn, bool flag);

void
initBetaTable(vector<F> &beta_g, u8 gLength, const vector<F>::const_iterator &r_0, const vector<F>::const_iterator &r_1,
              const F &alpha, const F &beta, int thread = 1);

void initPhiTable(F *phi_g, const layer &cur_layer, const F *r_0, const F *r_1, F alpha, F beta);

void phiGInit(vector<F> &phi_g, const vector<F>::const_iterator &rx, const F &scale, int n, bool isIFFT);

void initBetaTable(vector<F> &beta_g, u8 gLength, const vector<F>::const_iterator &r, const F &init);
void initBetaTable(vector<F> &beta_g, u8 gLength, const vector<F>::const_iterator &r, const F &init, int thread);

template <typename T>
class ThreadSafeQueue {
public:
    ThreadSafeQueue() = default;
    void Push(T value) {
        std::unique_lock<std::mutex> lock(mutex_);
        queue_.push(value);
        condition_variable_.notify_one();
    }
    bool TryPop(T &value) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (queue_.empty()) return false;
        value = queue_.front();
        queue_.pop();
        return true;
    }
    void WaitAndPop(T &value) {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_variable_.wait(lock, [this] { return !queue_.empty(); });
        value = queue_.front();
        queue_.pop();
    }
    bool Empty() {
        std::unique_lock<std::mutex> lock(mutex_);
        return queue_.empty();
    }
    size_t Size() {
        std::unique_lock<std::mutex> lock(mutex_);
        return queue_.size();
    }
    void Clear() {
        std::unique_lock<std::mutex> lock(mutex_);
        queue_ = std::queue<T>();
    }

private:
    std::mutex mutex_;
    std::queue<T> queue_;
    std::condition_variable condition_variable_;
};

bool check(long x, long y, long nx, long ny);

long matIdx(long x, long y, long n);

long cubIdx(long x, long y, long z, long n, long m);

long tesIdx(long w, long x, long y, long z, long n, long m, long l);

void initLayer(layer &circuit, long size, layerType ty);

long sqr(long x);

double byte2KB(size_t x);

double byte2MB(size_t x);

double byte2GB(size_t x);

F getRootOfUnit(int n);

#endif //ZKCNN_UTILS_HPP
