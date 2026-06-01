#include "kzh_wrapper.hpp"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

bool path_exists(const std::string &path) {
    struct stat st {};
    return ::stat(path.c_str(), &st) == 0;
}

std::string parent_dir(const std::string &path) {
    const size_t pos = path.find_last_of('/');
    if (pos == std::string::npos) return ".";
    if (pos == 0) return "/";
    return path.substr(0, pos);
}

std::string shell_quote(const std::string &input) {
    std::string out = "'";
    for (char ch : input) {
        if (ch == '\'') out += "'\\''";
        else out += ch;
    }
    out += "'";
    return out;
}

bool ensure_dir_recursive(const std::string &path) {
    if (path.empty()) return false;
    if (path_exists(path)) return true;

    std::string cur;
    for (size_t i = 0; i < path.size(); ++i) {
        char ch = path[i];
        cur.push_back(ch);
        if (ch != '/' && i + 1 != path.size()) continue;
        if (cur.empty() || cur == "/") continue;
        if (::mkdir(cur.c_str(), 0755) != 0 && errno != EEXIST) return false;
    }
    if (::mkdir(path.c_str(), 0755) != 0 && errno != EEXIST) return false;
    return true;
}

bool write_u64(std::ofstream &out, u64 value) {
    unsigned char buf[8];
    for (int i = 0; i < 8; ++i) buf[i] = static_cast<unsigned char>((value >> (8 * i)) & 0xFF);
    out.write(reinterpret_cast<const char *>(buf), 8);
    return static_cast<bool>(out);
}

bool write_field_scalar(std::ofstream &out, const F &value) {
    unsigned char buf[32];
    std::memset(buf, 0, sizeof(buf));
    value.getLittleEndian(buf, sizeof(buf));
    out.write(reinterpret_cast<const char *>(buf), sizeof(buf));
    return static_cast<bool>(out);
}

bool write_field_vector(const std::string &path, const vector<F> &values) {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) return false;
    if (!write_u64(out, values.size())) return false;
    for (const auto &value : values) {
        if (!write_field_scalar(out, value)) return false;
    }
    return true;
}

bool write_field_value(const std::string &path, const F &value) {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) return false;
    return write_field_scalar(out, value);
}

bool parse_metrics_file(const std::string &path, std::unordered_map<std::string, double> &metrics) {
    std::ifstream in(path);
    if (!in.is_open()) return false;
    std::string key;
    double value = 0.0;
    while (in >> key >> value) metrics[key] = value;
    return true;
}

bool run_command(const std::string &cmd) {
    int rc = std::system(cmd.c_str());
    if (rc == -1) return false;
    if (WIFEXITED(rc)) return WEXITSTATUS(rc) == 0;
    return false;
}

std::string get_env_or_default(const char *name, const char *fallback) {
    const char *value = std::getenv(name);
    return (value && *value) ? std::string(value) : std::string(fallback);
}

void cleanup_temp_dir(const std::string &dir) {
    const char *keep = std::getenv("ZKCNN_KZH_KEEP_TMP");
    if (keep && std::string(keep) == "1") return;
    const std::string files[] = {
            "/poly.bin",
            "/point.bin",
            "/value.bin",
            "/artifact.bin",
            "/prove_metrics.txt",
            "/verify_metrics.txt"
    };
    for (const auto &suffix : files) {
        std::string path = dir + suffix;
        ::unlink(path.c_str());
    }
    ::rmdir(dir.c_str());
}

} // namespace

bool run_kzh_pcs(const vector<F> &poly,
                 const vector<F> &point,
                 const F &value,
                 u8 num_vars,
                 KzhMetrics &metrics) {
    const std::string kzh_bin = get_env_or_default("ZKCNN_KZH_BIN", "kzh_gnark/zkcnn_kzh_cli");
    const std::string default_lib_dir = parent_dir(kzh_bin) + "/deps/icicle/lib";
    const std::string kzh_lib_dir = get_env_or_default("ZKCNN_KZH_LIB_DIR", default_lib_dir.c_str());
    const std::string srs_dir = get_env_or_default("ZKCNN_KZH_SRS_DIR", "output/kzh_srs");
    if (!path_exists(kzh_bin)) {
        fprintf(stderr, "Missing KZH sidecar binary: %s\n", kzh_bin.c_str());
        return false;
    }
    if (!ensure_dir_recursive(srs_dir)) {
        fprintf(stderr, "Failed to create KZH SRS directory: %s\n", srs_dir.c_str());
        return false;
    }

    const std::string command_env_prefix =
            path_exists(kzh_lib_dir)
            ? "DYLD_LIBRARY_PATH=" + shell_quote(kzh_lib_dir) + " "
            : "";

    const std::string srs_path = srs_dir + "/bls12_381_KZH4_gnark_n" + std::to_string(static_cast<int>(num_vars)) + ".srs";
    if (!path_exists(srs_path)) {
        std::ostringstream setup_cmd;
        setup_cmd << command_env_prefix
                  << shell_quote(kzh_bin)
                  << " setup --num-vars " << static_cast<int>(num_vars)
                  << " --srs " << shell_quote(srs_path);
        if (!run_command(setup_cmd.str())) {
            fprintf(stderr, "KZH setup failed for %s\n", srs_path.c_str());
            return false;
        }
    }

    char tmp_template[] = "/tmp/zkcnn-kzh-XXXXXX";
    char *tmp_dir_buf = ::mkdtemp(tmp_template);
    if (tmp_dir_buf == nullptr) {
        fprintf(stderr, "Failed to create temp dir for KZH artifacts.\n");
        return false;
    }
    std::string tmp_dir = tmp_dir_buf;
    const std::string poly_path = tmp_dir + "/poly.bin";
    const std::string point_path = tmp_dir + "/point.bin";
    const std::string value_path = tmp_dir + "/value.bin";
    const std::string artifact_path = tmp_dir + "/artifact.bin";
    const std::string prove_metrics_path = tmp_dir + "/prove_metrics.txt";
    const std::string verify_metrics_path = tmp_dir + "/verify_metrics.txt";

    if (!write_field_vector(poly_path, poly) ||
        !write_field_vector(point_path, point) ||
        !write_field_value(value_path, value)) {
        fprintf(stderr, "Failed to serialize KZH inputs.\n");
        cleanup_temp_dir(tmp_dir);
        return false;
    }

    std::ostringstream prove_cmd;
    prove_cmd << command_env_prefix
              << shell_quote(kzh_bin)
              << " prove"
              << " --srs " << shell_quote(srs_path)
              << " --poly " << shell_quote(poly_path)
              << " --point " << shell_quote(point_path)
              << " --artifact " << shell_quote(artifact_path)
              << " --metrics " << shell_quote(prove_metrics_path);
    if (!run_command(prove_cmd.str())) {
        fprintf(stderr, "KZH prove command failed.\n");
        cleanup_temp_dir(tmp_dir);
        return false;
    }

    std::unordered_map<std::string, double> prove_metrics;
    if (!parse_metrics_file(prove_metrics_path, prove_metrics)) {
        fprintf(stderr, "Failed to parse KZH prover metrics.\n");
        cleanup_temp_dir(tmp_dir);
        return false;
    }

    std::ostringstream verify_cmd;
    verify_cmd << command_env_prefix
               << shell_quote(kzh_bin)
               << " verify"
               << " --srs " << shell_quote(srs_path)
               << " --point " << shell_quote(point_path)
               << " --value " << shell_quote(value_path)
               << " --artifact " << shell_quote(artifact_path)
               << " --metrics " << shell_quote(verify_metrics_path);
    if (!run_command(verify_cmd.str())) {
        fprintf(stderr, "KZH verify command failed.\n");
        cleanup_temp_dir(tmp_dir);
        return false;
    }

    std::unordered_map<std::string, double> verify_metrics;
    if (!parse_metrics_file(verify_metrics_path, verify_metrics)) {
        fprintf(stderr, "Failed to parse KZH verifier metrics.\n");
        cleanup_temp_dir(tmp_dir);
        return false;
    }

    metrics.prover_time_sec = prove_metrics["prove_time_sec"];
    metrics.verifier_time_sec = verify_metrics["verify_time_sec"];
    metrics.proof_size_kb = prove_metrics["proof_size_bytes"] / 1024.0;

    cleanup_temp_dir(tmp_dir);
    return true;
}
