//
// Created by 69029 on 3/9/2021.
//

#ifndef ZKCNN_UTILS_HPP
#define ZKCNN_UTILS_HPP

#include "circuit.h"
#include <functional>

// Number of hardware threads to target for parallel work in this process.
unsigned hwThreads();

// Returns hwThreads() when `n` is large enough to amortize thread spin-up
// overhead, otherwise 1 (i.e. run serially).
unsigned parallelThreadsFor(u64 n);

// Splits [begin, end) into `threads` contiguous chunks and runs `body(l, r)`
// for each chunk on its own thread, blocking until all have joined. Falls
// back to calling body(begin, end) directly when threads <= 1.
void parallelFor(u64 begin, u64 end, unsigned threads, const std::function<void(u64, u64)> &body);

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
