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

static const int kThreads = (int) hwThreads();

static void markDctqLayer(layer &circuit, layerSpecialization specialization,
                          i64 width, i64 height, i64 block,
                          i64 image_start, i64 qd_start, i64 qr_start) {
    circuit.specialization = specialization;
    circuit.dctq_width = static_cast<u32>(width);
    circuit.dctq_height = static_cast<u32>(height);
    circuit.dctq_block = static_cast<u32>(block);
    circuit.dctq_image_start = static_cast<u32>(image_start);
    circuit.dctq_qd_start = static_cast<u32>(qd_start);
    circuit.dctq_qr_start = static_cast<u32>(qr_start);
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

    // circuit[0]'s uni_gates would normally be an identity mapping, but the
    // GKR sumcheck loop only ever proves layers 1..size-1 (the DCTQ layers
    // below read straight from `val[0]`), so populating it here is dead
    // work -- nothing ever reads circuit[0].uni_gates.

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
    markDctqLayer(circuit, layerSpecialization::DctqLeft, width, height, kBlock,
                  image_start_id, dct_left_start_id, dct_left_start_id + dct_size);

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
    markDctqLayer(circuit, layerSpecialization::DctqRight, width, height, kBlock,
                  image_start_id, dct_left_start_id, dct_left_start_id + dct_size);

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
    markDctqLayer(circuit, layerSpecialization::DctqHadamard, width, height, kBlock,
                  image_start_id, dct_left_start_id, q_start);

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

// Parses one whitespace-separated decimal number starting at `p` (stopping
// at `end`), writing the position just past it to `next`. Returns 0 if `p`
// is already at `end`, matching the old `ifstream >> double` "else F_ZERO"
// fallback for a short/malformed row.
static double parseDoubleAt(const char *p, const char *end, const char *&next) {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n')) ++p;
    bool neg = false;
    if (p < end && (*p == '-' || *p == '+')) { neg = (*p == '-'); ++p; }
    double v = 0.0;
    while (p < end && *p >= '0' && *p <= '9') { v = v * 10.0 + (*p - '0'); ++p; }
    if (p < end && *p == '.') {
        ++p;
        double frac = 0.1;
        while (p < end && *p >= '0' && *p <= '9') { v += (*p - '0') * frac; frac *= 0.1; ++p; }
    }
    if (p < end && (*p == 'e' || *p == 'E')) {
        ++p;
        bool eneg = false;
        if (p < end && (*p == '-' || *p == '+')) { eneg = (*p == '-'); ++p; }
        int exp = 0;
        while (p < end && *p >= '0' && *p <= '9') { exp = exp * 10 + (*p - '0'); ++p; }
        v *= std::pow(10.0, eneg ? -exp : exp);
    }
    next = p;
    return neg ? -v : v;
}

void dctq::loadImage() {
    // ifstream::operator>> is a poor fit here: it's a single serial cursor
    // over the whole file, so it can't be parallelized in place. Instead,
    // slurp the file once (one fast bulk read), then parse rows in
    // parallel -- each row is self-contained (exactly `width` values, per
    // infer_matrix_shape's up-front validation), so a thread's starting
    // value-index only depends on its row range, not on how much any other
    // thread has parsed.
    in.clear();
    in.seekg(0, std::ios::end);
    auto len = (size_t) in.tellg();
    in.seekg(0, std::ios::beg);
    std::vector<char> buf(len);
    if (len) in.read(buf.data(), (std::streamsize) len);

    std::vector<size_t> row_start((size_t) height + 1);
    size_t pos = 0;
    for (i64 r = 0; r < height; ++r) {
        while (pos < len && (buf[pos] == '\n' || buf[pos] == '\r')) ++pos;
        row_start[r] = pos;
        while (pos < len && buf[pos] != '\n') ++pos;
    }
    row_start[(size_t) height] = len;

    parallelFor(0, height, parallelThreadsFor((u64) height * width), [&](u64 r0, u64 r1) {
        for (u64 r = r0; r < r1; ++r) {
            const char *p = buf.data() + row_start[r];
            const char *end = buf.data() + row_start[r + 1];
            for (i64 c = 0; c < width; ++c) {
                const char *next;
                double num = parseDoubleAt(p, end, next);
                p = next;
                i64 idx = image_start_id + matIdx((i64) r, c, width);
                val[0].at(idx) = F(toFieldInt(num));
            }
        }
    });
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
