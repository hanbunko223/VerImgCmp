#include "dctq.hpp"
#include "utils.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <thread>

using std::cerr;
using std::endl;

static i64 toFieldInt(double v) {
    return static_cast<i64>(std::llround(v));
}

static i64 toSignedInt(const F &v) {
    if (!v.isNegative()) return v.getInt64();
    F abs_v = -v;
    return -abs_v.getInt64();
}

static constexpr int kThreads = 8;
static constexpr int kIntervalParts = kThreads * 4;

static void buildIntervals(layer &circuit) {
    circuit.uni_interval.clear();
    circuit.bin_interval.clear();

    auto make_intervals = [](size_t total, std::vector<std::pair<int,int>> &intervals) {
        if (total == 0) return;
        size_t parts = std::min<size_t>(kIntervalParts, total);
        size_t chunk = (total + parts - 1) / parts;
        for (size_t i = 0; i < parts; ++i) {
            size_t l = i * chunk;
            if (l >= total) break;
            size_t r = std::min(total, l + chunk);
            intervals.emplace_back(static_cast<int>(l), static_cast<int>(r));
        }
    };

    make_intervals(circuit.uni_gates.size(), circuit.uni_interval);
    make_intervals(circuit.bin_gates.size(), circuit.bin_interval);
}

static void parallelRange(i64 begin, i64 end, int threads, const std::function<void(i64, i64)> &work) {
    if (threads <= 1 || end <= begin) {
        work(begin, end);
        return;
    }
    i64 total = end - begin;
    i64 chunk = (total + threads - 1) / threads;
    std::vector<std::thread> workers;
    workers.reserve(threads);
    for (int t = 0; t < threads; ++t) {
        i64 start = begin + t * chunk;
        if (start >= end) break;
        i64 finish = std::min(end, start + chunk);
        workers.emplace_back([=, &work]() { work(start, finish); });
    }
    for (auto &th : workers) th.join();
}

dctq::dctq(i64 width_, i64 height_, const std::string &input_file, const std::string &dct_file,
           const std::string &q_file, const std::string &output_file)
        : width(width_), height(height_) {
    if (!input_file.empty()) {
        in.open(input_file);
        if (!in.is_open())
            fprintf(stderr, "Can't find the input file!!!\n");
    }
    if (!dct_file.empty()) {
        dct_in.open(dct_file);
        if (!dct_in.is_open())
            fprintf(stderr, "Can't find the q_D file!!!\n");
    }
    if (!q_file.empty()) {
        q_in.open(q_file);
        if (!q_in.is_open())
            fprintf(stderr, "Can't find the q_R file!!!\n");
    }
    if (!output_file.empty()) out.open(output_file);
    initLayout();
}

void dctq::initLayout() {
    image_size = width * height;
    dct_size = kBlock * kBlock;
    q_size = kBlock * kBlock;

    image_start_id = 0;
    dct_left_start_id = image_start_id + image_size;
    total_in_size = dct_left_start_id + dct_size + q_size;

    SIZE = 4;
}

void dctq::create(prover &pr, bool only_compute) {
    pr.C.init(Q_BIT_SIZE, SIZE);
    pr.val.resize(SIZE);
    val = pr.val.begin();
    two_mul = pr.C.two_mul.begin();

    i64 layer_id = 0;
    inputLayer(pr.C.circuit[layer_id++]);
    dctLeftLayer(pr.C.circuit[layer_id], layer_id);
    dctRightLayer(pr.C.circuit[layer_id], layer_id);
    hadamardLayer(pr.C.circuit[layer_id], layer_id);

    if (out.is_open()) writeOutput(pr);

    if (only_compute) return;
    pr.C.initSubset();
    cerr << "finish creating circuit." << endl;
}

void dctq::inputLayer(layer &circuit) {
    initLayer(circuit, total_in_size, layerType::INPUT);

    circuit.uni_gates.reserve(total_in_size);
    for (i64 i = 0; i < total_in_size; ++i)
        circuit.uni_gates.emplace_back(i, i, 0, 0);

    val[0].resize(total_in_size);

    // Input layout: [image][q_D][q_R]
    loadImage();
    initDctLeft();
    initDctRight();
    initHadamard();
}

void dctq::dctLeftLayer(layer &circuit, i64 &layer_id) {
    i64 out_size = height * width;
    initLayer(circuit, out_size, layerType::FCONN);
    circuit.need_phase2 = true;

    for (i64 br = 0; br < height; br += kBlock) {
        for (i64 r = 0; r < kBlock; ++r) {
            i64 out_r = br + r;
            for (i64 c = 0; c < width; ++c) {
                i64 g = matIdx(out_r, c, width);
                for (i64 k = 0; k < kBlock; ++k) {
                    i64 u = image_start_id + matIdx(br + k, c, width);
                    i64 v = dct_left_start_id + matIdx(r, k, kBlock);
                    circuit.bin_gates.emplace_back(g, u, v, 0, 0);
                }
            }
        }
    }
    buildIntervals(circuit);

    val[layer_id].resize(out_size);
    auto &out = val[layer_id];
    const auto &in_img = val[0];
    const auto &dct = val[0];
    i64 blocks = height / kBlock;
    parallelRange(0, blocks, kThreads, [&](i64 b0, i64 b1) {
        for (i64 b = b0; b < b1; ++b) {
            i64 br = b * kBlock;
            for (i64 r = 0; r < kBlock; ++r) {
                i64 out_r = br + r;
                for (i64 c = 0; c < width; ++c) {
                    F sum = F_ZERO;
                    for (i64 k = 0; k < kBlock; ++k) {
                        i64 u = image_start_id + matIdx(br + k, c, width);
                        i64 v = dct_left_start_id + matIdx(r, k, kBlock);
                        sum = sum + in_img.at(u) * dct.at(v);
                    }
                    i64 g = matIdx(out_r, c, width);
                    out.at(g) = sum * circuit.scale;
                }
            }
        }
    });
    ++layer_id;
}

void dctq::dctRightLayer(layer &circuit, i64 &layer_id) {
    i64 out_size = height * width;
    initLayer(circuit, out_size, layerType::FCONN);
    circuit.need_phase2 = true;

    for (i64 r = 0; r < height; ++r) {
        for (i64 bc = 0; bc < width; bc += kBlock) {
            for (i64 c = 0; c < kBlock; ++c) {
                i64 out_c = bc + c;
                i64 g = matIdx(r, out_c, width);
                for (i64 k = 0; k < kBlock; ++k) {
                    i64 u = matIdx(r, bc + k, width);
                    i64 v = dct_left_start_id + matIdx(c, k, kBlock);
                    circuit.bin_gates.emplace_back(g, u, v, 0, 2);
                }
            }
        }
    }
    buildIntervals(circuit);

    val[layer_id].resize(out_size);
    auto &out = val[layer_id];
    const auto &in_prev = val[layer_id - 1];
    const auto &dct = val[0];
    parallelRange(0, height, kThreads, [&](i64 r0, i64 r1) {
        for (i64 r = r0; r < r1; ++r) {
            for (i64 bc = 0; bc < width; bc += kBlock) {
                for (i64 c = 0; c < kBlock; ++c) {
                    i64 out_c = bc + c;
                    F sum = F_ZERO;
                    for (i64 k = 0; k < kBlock; ++k) {
                        i64 u = matIdx(r, bc + k, width);
                        i64 v = dct_left_start_id + matIdx(c, k, kBlock);
                        sum = sum + in_prev.at(u) * dct.at(v);
                    }
                    i64 g = matIdx(r, out_c, width);
                    out.at(g) = sum * circuit.scale;
                }
            }
        }
    });
    ++layer_id;
}

void dctq::hadamardLayer(layer &circuit, i64 &layer_id) {
    i64 out_size = height * width;
    initLayer(circuit, out_size, layerType::FCONN);
    circuit.need_phase2 = true;

    i64 q_start = dct_left_start_id + dct_size;
    for (i64 r = 0; r < height; ++r)
        for (i64 c = 0; c < width; ++c) {
            i64 g = matIdx(r, c, width);
            i64 u = g;
            i64 qr = r & (kBlock - 1);
            i64 qc = c & (kBlock - 1);
            i64 v = q_start + matIdx(qr, qc, kBlock);
            circuit.bin_gates.emplace_back(g, u, v, 0, 2);
        }
    buildIntervals(circuit);

    val[layer_id].resize(out_size);
    auto &out = val[layer_id];
    const auto &in_prev = val[layer_id - 1];
    const auto &qmat = val[0];
    parallelRange(0, height, kThreads, [&](i64 r0, i64 r1) {
        for (i64 r = r0; r < r1; ++r) {
            i64 qr = r & (kBlock - 1);
            for (i64 c = 0; c < width; ++c) {
                i64 qc = c & (kBlock - 1);
                i64 g = matIdx(r, c, width);
                i64 v = q_start + matIdx(qr, qc, kBlock);
                out.at(g) = in_prev.at(g) * qmat.at(v) * circuit.scale;
            }
        }
    });
    ++layer_id;
}

void dctq::calcNormalLayer(const layer &circuit, i64 layer_id) {
    val[layer_id].resize(circuit.size);
    for (auto &x : val[layer_id]) x.clear();

    for (auto &gate : circuit.uni_gates) {
        val[layer_id].at(gate.g) = val[layer_id].at(gate.g) +
                                   val[gate.lu].at(gate.u) * two_mul[gate.sc];
    }

    for (auto &gate : circuit.bin_gates) {
        u8 bin_lu = gate.getLayerIdU(layer_id);
        u8 bin_lv = gate.getLayerIdV(layer_id);
        val[layer_id].at(gate.g) = val[layer_id].at(gate.g) +
                                   val[bin_lu].at(gate.u) * val[bin_lv][gate.v] * two_mul[gate.sc];
    }

    for (i64 g = 0; g < circuit.size; ++g)
        val[layer_id].at(g) = val[layer_id].at(g) * circuit.scale;
}

void dctq::loadImage() {
    double num = 0.0;
    for (i64 r = 0; r < height; ++r)
        for (i64 c = 0; c < width; ++c) {
            i64 idx = image_start_id + matIdx(r, c, width);
            if (in >> num)
                val[0].at(idx) = F(toFieldInt(num));
            else
                val[0].at(idx) = F_ZERO;
        }
}

void dctq::initDctLeft() {
    double num = 0.0;
    for (i64 r = 0; r < kBlock; ++r)
        for (i64 c = 0; c < kBlock; ++c) {
            i64 idx = dct_left_start_id + matIdx(r, c, kBlock);
            if (dct_in >> num)
                val[0].at(idx) = F(toFieldInt(num));
            else
                val[0].at(idx) = (r == c) ? F_ONE : F_ZERO;
        }
}

void dctq::initDctRight() {
    // right DCT reuses the same 8x8 matrix stored in initDctLeft(),
    // and the right layer reads it as q_D^T by swapping indices.
}

void dctq::initHadamard() {
    double num = 0.0;
    i64 q_start = dct_left_start_id + dct_size;
    for (i64 r = 0; r < kBlock; ++r)
        for (i64 c = 0; c < kBlock; ++c) {
            i64 idx = q_start + matIdx(r, c, kBlock);
            if (q_in >> num)
                val[0].at(idx) = F(toFieldInt(num));
            else
                val[0].at(idx) = F_ONE;
        }
}

void dctq::writeOutput(const prover &pr) {
    const auto &out_layer = pr.val[SIZE - 1];
    for (i64 r = 0; r < height; ++r) {
        for (i64 c = 0; c < width; ++c) {
            i64 idx = matIdx(r, c, width);
            out << toSignedInt(out_layer.at(idx));
            if (c + 1 < width) out << ' ';
        }
        out << '\n';
    }
}
