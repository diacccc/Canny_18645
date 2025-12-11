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
    constexpr int N = 64;
    std::vector<int16_t> src(M * N);
    std::vector<int16_t> gx(M * N), gy(M * N), mag(M * N);
    std::vector<int16_t> gx_ref(M * N), gy_ref(M * N), mag_ref(M * N);

    std::mt19937 rng(1337);
    std::uniform_int_distribution<int> dist(-128, 127);
    for (auto& px : src) {
        px = static_cast<int16_t>(dist(rng));
    }

    sobel(src.data(), gx.data(), gy.data(), mag.data(), M, N);
    sobel_reference(src, M, N, gx_ref, gy_ref, mag_ref);

    int mismatches = 0;
    for (size_t idx = 0; idx < src.size(); ++idx) {
        if (gx[idx] != gx_ref[idx] || gy[idx] != gy_ref[idx] || mag[idx] != mag_ref[idx]) {
            if (mismatches < 5) {
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

static void sobel_checkerboard_test() {
    constexpr int M = 16;
    constexpr int N = 16;
    std::vector<int16_t> src(M * N);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            src[i * N + j] = ((i + j) & 1) ? 255 : 0;
        }
    }
    std::vector<int16_t> gx(M * N), gy(M * N), mag(M * N);
    std::vector<int16_t> gx_ref(M * N), gy_ref(M * N), mag_ref(M * N);

    sobel(src.data(), gx.data(), gy.data(), mag.data(), M, N);
    sobel_reference(src, M, N, gx_ref, gy_ref, mag_ref);

    bool ok = std::equal(gx.begin(), gx.end(), gx_ref.begin()) &&
              std::equal(gy.begin(), gy.end(), gy_ref.begin()) &&
              std::equal(mag.begin(), mag.end(), mag_ref.begin());
    check(ok, "sobel matches reference on checkerboard");
}

static void sobel_single_column_test() {
    constexpr int M = 32;
    constexpr int N = 1;
    std::vector<int16_t> src(M * N, 42);
    std::vector<int16_t> gx(M * N, -1), gy(M * N, -1), mag(M * N, -1);

    sobel(src.data(), gx.data(), gy.data(), mag.data(), M, N);

    bool all_zero = true;
    for (int idx = 0; idx < M * N; ++idx) {
        if (gx[idx] != 0 || gy[idx] != 0 || mag[idx] != 0) {
            all_zero = false;
            break;
        }
    }
    check(all_zero, "sobel handles single-column images");
}

#define MAX_FREQ 3.2
#define BASE_FREQ 2.1

#define SOBEL_STEP_ONCE() \
    SOBEL_TILE(gx_ptr, gy_ptr, mag_ptr, prev_img_ptr, curr_img_ptr, next_img_ptr); \
    gx_ptr += 64; \
    gy_ptr += 64; \
    prev_img_ptr += 64; \
    curr_img_ptr += 64; \
    next_img_ptr += 64; \
    mag_ptr += 64;

#define REPEAT10_SOBEL() \
    SOBEL_STEP_ONCE(); SOBEL_STEP_ONCE(); SOBEL_STEP_ONCE(); SOBEL_STEP_ONCE(); SOBEL_STEP_ONCE(); \
    SOBEL_STEP_ONCE(); SOBEL_STEP_ONCE(); SOBEL_STEP_ONCE(); SOBEL_STEP_ONCE(); SOBEL_STEP_ONCE();

#define REPEAT100_SOBEL() \
    REPEAT10_SOBEL(); REPEAT10_SOBEL(); REPEAT10_SOBEL(); REPEAT10_SOBEL(); REPEAT10_SOBEL(); \
    REPEAT10_SOBEL(); REPEAT10_SOBEL(); REPEAT10_SOBEL(); REPEAT10_SOBEL(); REPEAT10_SOBEL();

#define REPEAT1000_SOBEL() \
    REPEAT100_SOBEL(); REPEAT100_SOBEL(); REPEAT100_SOBEL(); REPEAT100_SOBEL(); REPEAT100_SOBEL(); \
    REPEAT100_SOBEL(); REPEAT100_SOBEL(); REPEAT100_SOBEL(); REPEAT100_SOBEL(); REPEAT100_SOBEL(); 

void benchmark_performance()
{
	const int M = 3;      // rows
	const int N = 64;     // cols (must satisfy j+15 < N)
	const int i = 1;      // middle row (to allow i-1 and i+1 valid)
	const int j = 8;      // start column (ensure j-1 and j+15 within bounds)
    const int num_runs = 10000000;
    const int NUM_INST = 60;

	std::vector<int16_t> gx(M * N * NUM_INST, 0);  // strong x gradient
	std::vector<int16_t> gy(M * N * NUM_INST, 0);     // zero y gradient
	std::vector<int16_t> mag(M * N * NUM_INST, 0);  // baseline magnitude
    std::vector<int16_t> img(M * N * NUM_INST, 128);  // input image
    for (int i = 0; i < M * N * NUM_INST; ++i) {
        img[i] = (rand() % 512) - 256;
    }
	// Macro helpers: repeat the kernel 100 times without an explicit loop.
	// Each iteration advances all pointers by M*N elements.
    unsigned long long st;
    unsigned long long et;
    unsigned long long dt; 
    unsigned long long dt_min = (unsigned long long)-1;
    unsigned long long sum = 0;
    for (int run = 0; run < num_runs; ++run) {
        int16_t *gx_ptr = gx.data() + (N * NUM_INST);
        int16_t *gy_ptr = gy.data() + (N * NUM_INST);
        int16_t *prev_img_ptr = img.data();
        int16_t *curr_img_ptr = img.data() + (N * NUM_INST);
        int16_t *next_img_ptr = img.data() + (2 * N * NUM_INST);
        int16_t *mag_ptr = mag.data() + (N * NUM_INST);
        st = rdtsc();
        asm volatile("look_here:" ::: "memory");
        REPEAT10_SOBEL();
        REPEAT10_SOBEL();
        REPEAT10_SOBEL();
        REPEAT10_SOBEL();
        REPEAT10_SOBEL();
        REPEAT10_SOBEL();
        et = rdtsc();
        dt = et - st;
        if (dt < dt_min) dt_min = dt;
        sum += dt;
    }

    printf("Throughput : %lf \n\r", 64 * ((double)NUM_INST * 16) / (dt_min * MAX_FREQ/BASE_FREQ));

}

int main() {
    sobel_random_image_test();
    sobel_checkerboard_test();
    sobel_single_column_test();
    // benchmark_performance();
    return test_failures;
}
