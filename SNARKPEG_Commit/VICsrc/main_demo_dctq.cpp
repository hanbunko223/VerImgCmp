//
// Demo entry for DCTQ
//

#include "dctq.hpp"
#include "verifier.hpp"
#include "global_var.hpp"

#include <fstream>
#include <sstream>

#define INPUT_FILE_ID 1
#define DCT_FILE_ID 2
#define Q_FILE_ID 3
#define OUTPUT_FILE_ID 4

vector<std::string> output_tb(16, "");

static bool infer_matrix_shape(const char *path, i64 &width, i64 &height) {
    std::ifstream in(path);
    if (!in.is_open()) {
        fprintf(stderr, "Can't open input file: %s\n", path);
        return false;
    }

    width = -1;
    height = 0;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        i64 cnt = 0;
        long long value = 0;
        while (ss >> value) ++cnt;
        if (cnt == 0) continue;
        if (width == -1) width = cnt;
        else if (width != cnt) {
            fprintf(stderr, "Malformed input matrix: inconsistent row width in %s\n", path);
            return false;
        }
        ++height;
    }

    if (width <= 0 || height <= 0) {
        fprintf(stderr, "Malformed input matrix: empty matrix in %s\n", path);
        return false;
    }
    if ((width % 8) != 0 || (height % 8) != 0) {
        fprintf(stderr, "Input dimensions must be divisible by 8, got %lldx%lld\n",
                (long long)width, (long long)height);
        return false;
    }
    return true;
}

int main(int argc, char **argv) {
    initPairing(mcl::BLS12_381);

    if (argc < 5) {
        fprintf(stderr, "Usage: %s <input_file> <qD_file> <qR_file> <output_file>\n", argv[0]);
        return 1;
    }

    const char *i_filename = argv[INPUT_FILE_ID];
    const char *dct_filename = argv[DCT_FILE_ID];
    const char *q_filename = argv[Q_FILE_ID];
    const char *o_filename = argv[OUTPUT_FILE_ID];

    i64 width = 0, height = 0;
    if (!infer_matrix_shape(i_filename, width, height))
        return 1;

    prover p;
    dctq comp(width, height, i_filename, dct_filename, q_filename, o_filename);
    comp.create(p, false);

    verifier v(&p, p.C);
    v.verify();

    return 0;
}
