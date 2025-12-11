#include <algorithm>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>
#include <cstdint>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <iomanip>
#include <fstream>
#include <deque>
#include <random>
#include <algorithm>

#include "sobel.hpp"
#include "utils.hpp"
#include "ref.hpp"

static __inline__ unsigned long long rdtsc(void) {
    unsigned hi, lo;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return (static_cast<unsigned long long>(lo)) |
           (static_cast<unsigned long long>(hi) << 32);
}

static int test_failures = 0;

static void check(bool cond, const char* msg) {
    if (!cond) {
        ++test_failures;
        std::cerr << "FAIL: " << msg << std::endl;
    }
}

static inline int16_t sample_with_border(const std::vector<int16_t>& src, int M, int N, int r, int c) {
    if (r < 0 || r >= M || c < 0 || c >= N) {
        return 0;
    }
    return src[r * N + c];
}

static void sobel_reference(const std::vector<int16_t>& src, int M, int N,
                            std::vector<int16_t>& gx,
                            std::vector<int16_t>& gy,
                            std::vector<int16_t>& mag) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int p00 = sample_with_border(src, M, N, i - 1, j - 1);
            int p01 = sample_with_border(src, M, N, i - 1, j);
            int p02 = sample_with_border(src, M, N, i - 1, j + 1);
            int p10 = sample_with_border(src, M, N, i, j - 1);
            int p12 = sample_with_border(src, M, N, i, j + 1);
            int p20 = sample_with_border(src, M, N, i + 1, j - 1);
            int p21 = sample_with_border(src, M, N, i + 1, j);
            int p22 = sample_with_border(src, M, N, i + 1, j + 1);

            int gx_val = -p00 - 2 * p10 - p20 + p02 + 2 * p12 + p22;
            int gy_val = -p00 - 2 * p01 - p02 + p20 + 2 * p21 + p22;
            int mag_val = std::abs(gx_val) + std::abs(gy_val);

            int idx = i * N + j;
            gx[idx] = static_cast<int16_t>(gx_val);
            gy[idx] = static_cast<int16_t>(gy_val);
            mag[idx] = static_cast<int16_t>(mag_val);
        }
    }
}

static void sobel_random_image_test() {
    constexpr int M = 32;
    constexpr int N = 70;
    std::vector<int16_t> src(M * N);
    std::vector<int16_t> gx(M * N), gy(M * N), mag(M * N);
    std::vector<int16_t> gx_ref(M * N), gy_ref(M * N), mag_ref(M * N);

    std::mt19937 rng(1337);
    std::uniform_int_distribution<int> dist(-128, 127);
    for (auto& px : src) {
        px = static_cast<int16_t>(dist(rng));
    }

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << std::setw(5) << src[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
    
    sobel(src.data(), gx.data(), gy.data(), mag.data(), M, N);
    sobel_reference(src, M, N, gx_ref, gy_ref, mag_ref);

    int mismatches = 0;
    for (size_t idx = 0; idx < src.size(); ++idx) {
        if (gx[idx] != gx_ref[idx] || gy[idx] != gy_ref[idx] || mag[idx] != mag_ref[idx]) {
            if (mismatches < 20) {
                int r = static_cast<int>(idx / N);
                int c = static_cast<int>(idx % N);
                std::cerr << "Mismatch at (" << r << "," << c << ")"
                          << " gx=" << gx[idx] << " ref=" << gx_ref[idx]
                          << " gy=" << gy[idx] << " ref=" << gy_ref[idx]
                          << " mag=" << mag[idx] << " ref=" << mag_ref[idx]
                          << std::endl;
            }
            ++mismatches;
        }
    }
    if (mismatches != 0) {
        std::cerr << "Total mismatches: " << mismatches << std::endl;
    }
    check(mismatches == 0, "sobel matches reference on random image");
}


int main() {
    sobel_random_image_test();
    benchmark_performance();
    return test_failures;
}
