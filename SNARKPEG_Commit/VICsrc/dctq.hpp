//
// DCTQ: q_D * A * q_D^T, followed by Hadamard product with q_R.
//

#ifndef VIC_DCTQ_HPP
#define VIC_DCTQ_HPP

#include <fstream>
#include <string>

#include "circuit.h"
#include "prover.hpp"

class dctq {
public:
    dctq(i64 width, i64 height, const std::string &input_file, const std::string &dct_file,
         const std::string &q_file, const std::string &output_file);
    void create(prover &pr, bool only_compute);

private:
    void initLayout();

    void inputLayer(layer &circuit);
    void dctLeftLayer(layer &circuit, i64 &layer_id);
    void dctRightLayer(layer &circuit, i64 &layer_id);
    void hadamardLayer(layer &circuit, i64 &layer_id);

    void calcNormalLayer(const layer &circuit, i64 layer_id);

    void loadImage();
    void initDctLeft();
    void initDctRight();
    void initHadamard();
    void writeOutput(const prover &pr);

    static constexpr i64 kBlock = 8;

    i64 width;
    i64 height;

    i64 image_size;
    i64 dct_size;
    i64 q_size;

    i64 image_start_id;
    i64 dct_left_start_id;
    i64 total_in_size;

    i64 SIZE;
    const i64 Q_BIT_SIZE = 220;

    std::ifstream in;
    std::ifstream dct_in;
    std::ifstream q_in;
    std::ofstream out;

    std::vector<std::vector<F>>::iterator val;
    std::vector<F>::iterator two_mul;
};

#endif // VIC_DCTQ_HPP
