#include "circuit.h"
#include "utils.hpp"

static void resetLayerSubset(layer &cur) {
    for (int i = 0; i < 2; ++i) {
        cur.size_u[i] = 0;
        cur.size_v[i] = 0;
        cur.bit_length_u[i] = -1;
        cur.bit_length_v[i] = -1;
    }
    cur.ori_id_u.clear();
    cur.ori_id_v.clear();
}

static void appendRange(vector<u32> &out, u32 start, u32 count) {
    out.reserve(out.size() + count);
    for (u32 i = 0; i < count; ++i) out.push_back(start + i);
}

static void initDctqSubset(layer &cur, const layer &lst) {
    const u32 image_size = cur.dctq_width * cur.dctq_height;
    const u32 block_entries = cur.dctq_block * cur.dctq_block;

    switch (cur.specialization) {
        case layerSpecialization::DctqLeft:
            cur.size_u[0] = image_size;
            cur.bit_length_u[0] = ceilPow2BitLength(cur.size_u[0]);
            appendRange(cur.ori_id_u, cur.dctq_image_start, image_size);

            cur.size_v[0] = block_entries;
            cur.bit_length_v[0] = ceilPow2BitLength(cur.size_v[0]);
            appendRange(cur.ori_id_v, cur.dctq_qd_start, block_entries);
            break;
        case layerSpecialization::DctqRight:
            cur.size_u[1] = lst.size;
            cur.bit_length_u[1] = lst.bit_length;

            cur.size_v[0] = block_entries;
            cur.bit_length_v[0] = ceilPow2BitLength(cur.size_v[0]);
            appendRange(cur.ori_id_v, cur.dctq_qd_start, block_entries);
            break;
        case layerSpecialization::DctqHadamard:
            cur.size_u[1] = lst.size;
            cur.bit_length_u[1] = lst.bit_length;

            cur.size_v[0] = block_entries;
            cur.bit_length_v[0] = ceilPow2BitLength(cur.size_v[0]);
            appendRange(cur.ori_id_v, cur.dctq_qr_start, block_entries);
            break;
        case layerSpecialization::None:
            break;
    }

    cur.updateSize();
}

void layeredCircuit::initSubset() {
    cerr << "begin subset init." << endl;
    vector<int> visited_uidx(circuit[0].size);  // whether the i-th layer, j-th gate has been visited in the current layer
    vector<u64> subset_uidx(circuit[0].size);   // the subset index of the i-th layer, j-th gate
    vector<int> visited_vidx(circuit[0].size);  // whether the i-th layer, j-th gate has been visited in the current layer
    vector<u64> subset_vidx(circuit[0].size);   // the subset index of the i-th layer, j-th gate

    for (u8 i = 1; i < size; ++i) {
        auto &cur = circuit[i], &lst = circuit[i - 1];
        resetLayerSubset(cur);
        if (cur.isDctqStructured()) {
            initDctqSubset(cur, lst);
            continue;
        }

        bool has_pre_layer_u = circuit[i].ty == layerType::FFT || circuit[i].ty == layerType::IFFT;
        bool has_pre_layer_v = false;

        for (auto &gate: cur.uni_gates) {
            if (!gate.lu) {
                if (visited_uidx[gate.u] != i) {
                    visited_uidx[gate.u] = i;
                    subset_uidx[gate.u] = cur.size_u[0];
                    cur.ori_id_u.push_back(gate.u);
                    ++cur.size_u[0];
                }
                gate.u = subset_uidx[gate.u];
            }
            has_pre_layer_u |= (gate.lu != 0);
        }

        for (auto &gate: cur.bin_gates) {
            if (!gate.getLayerIdU(i)) {
                if (visited_uidx[gate.u] != i) {
                    visited_uidx[gate.u] = i;
                    subset_uidx[gate.u] = cur.size_u[0];
                    cur.ori_id_u.push_back(gate.u);
                    ++cur.size_u[0];
                }
                gate.u = subset_uidx[gate.u];
            }
            if (!gate.getLayerIdV(i)) {
                if (visited_vidx[gate.v] != i) {
                    visited_vidx[gate.v] = i;
                    subset_vidx[gate.v] = cur.size_v[0];
                    cur.ori_id_v.push_back(gate.v);
                    ++cur.size_v[0];
                }
                gate.v = subset_vidx[gate.v];
            }
            has_pre_layer_u |= (gate.getLayerIdU(i) != 0);
            has_pre_layer_v |= (gate.getLayerIdV(i) != 0);
        }

        cur.bit_length_u[0] = ceilPow2BitLength(cur.size_u[0]);
        cur.bit_length_v[0] = ceilPow2BitLength(cur.size_v[0]);

        if (has_pre_layer_u) switch (cur.ty) {
                case layerType::FFT:
                    cur.size_u[1] = 1ULL << cur.fft_bit_length - 1;
                    cur.bit_length_u[1] = cur.fft_bit_length - 1;
                    break;
                case layerType::IFFT:
                    cur.size_u[1] = 1ULL << cur.fft_bit_length;
                    cur.bit_length_u[1] = cur.fft_bit_length;
                    break;
                default:
                    cur.size_u[1] = lst.size ;
                    cur.bit_length_u[1] = lst.bit_length;
                    break;
            } else {
            cur.size_u[1] = 0;
            cur.bit_length_u[1] = -1;
        }

        if (has_pre_layer_v) {
            if (cur.ty == layerType::DOT_PROD) {
                cur.size_v[1] = lst.size >> cur.fft_bit_length;
                cur.bit_length_v[1] = lst.bit_length - cur.fft_bit_length;
            } else {
                cur.size_v[1] = lst.size;
                cur.bit_length_v[1] = lst.bit_length;
            }
        } else {
            cur.size_v[1] = 0;
            cur.bit_length_v[1] = -1;
        }
        cur.updateSize();
    }
    cerr << "begin subset finish." << endl;
}

void layeredCircuit::init(u8 q_bit_size, u8 _layer_sz) {
    two_mul.resize((q_bit_size + 1) << 1);
    two_mul[0] = F_ONE;
    two_mul[q_bit_size + 1] = -F_ONE;
    for (int i = 1; i <= q_bit_size; ++i) {
        two_mul[i] = two_mul[i - 1] + two_mul[i - 1];
        two_mul[i + q_bit_size + 1] = -two_mul[i];
    }
    size = _layer_sz;
    circuit.resize(size);
}
