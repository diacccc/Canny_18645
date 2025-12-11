#include <algorithm>
#include <chrono>
#include <deque>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
#include <opencv2/core/ocl.hpp>

#include "sobel.hpp"
#include "ref.hpp"
#include "utils.hpp"

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

void benchmark_performance()
{
    const int M = 3;
    const int N = 64;
    const int num_runs = 10000000;
    const int NUM_INST = 30;

    std::vector<int16_t> gx(M * N * NUM_INST, 0);
    std::vector<int16_t> gy(M * N * NUM_INST, 0);
    std::vector<int16_t> mag(M * N * NUM_INST, 0);
    std::vector<int16_t> img(M * N * NUM_INST, 128);
    for (int i = 0; i < M * N * NUM_INST; ++i) {
        img[i] = (rand() % 512) - 256;
    }

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
        // REPEAT10_SOBEL();
        // REPEAT10_SOBEL();
        // REPEAT10_SOBEL();
        et = rdtsc();
        dt = et - st;
        if (dt < dt_min) dt_min = dt;
        sum += dt;
    }

    printf("SOBEL KERNEL Throughput : %lf \n\r", 64 * ((double)NUM_INST * 16) / (dt_min * MAX_FREQ/BASE_FREQ));
}

using Clock = std::chrono::high_resolution_clock;

struct BenchmarkResult {
    std::string label;
    double avg_ms;
};

static BenchmarkResult time_custom_sobel(const std::vector<int16_t>& src,
                                         int M,
                                         int N,
                                         int iterations) {
    std::vector<int16_t> gx(M * N);
    std::vector<int16_t> gy(M * N);
    std::vector<int16_t> mag(M * N);

    auto start = Clock::now();
    for (int i = 0; i < iterations; ++i) {
        sobel(src.data(), gx.data(), gy.data(), mag.data(), M, N);
    }
    auto end = Clock::now();
    double avg_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
    return {"custom_sobel", avg_ms};
}

static BenchmarkResult time_stage1(const std::vector<int16_t>& src,
                                   int M,
                                   int N,
                                   int iterations) {
    cv::Mat gray(M, N, CV_16S, const_cast<int16_t*>(src.data()));
    std::deque<uchar*> borderPeaks;
    cv::Mat map;
    cv::customizedCanny canny(gray, map, borderPeaks, 0, 0, 3, false);
    cv::Range all_rows(0, M);

    cv::Mat dx, dy;
    auto start = Clock::now();
    for (int i = 0; i < iterations; ++i) {
        canny.stage1(all_rows, dx, dy);
    }
    auto end = Clock::now();
    double avg_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
    return {"opencv_stage1", avg_ms};
}

static void print_results_header(std::ostream& out) {
    out << "M,N,custom_ms,opencv_ms,speedup(stage1/custom)" << std::endl;
}

static void print_results_row(std::ostream& out, int M, int N, const BenchmarkResult& custom, const BenchmarkResult& stage1) {
    double speedup = stage1.avg_ms / custom.avg_ms;
    out << M << "," << N << ',' << custom.avg_ms << ',' << stage1.avg_ms << ',' << speedup << std::endl;
}

int main(int argc, char** argv) {
    cv::setNumThreads(1);
    cv::ocl::setUseOpenCL(false);
    
    benchmark_performance();

    int iterations = 10;
    if (argc >= 2) {
        iterations = std::stoi(argv[1]);
    }

    std::vector<std::pair<int, int>> sizes = {
        {128, 128}, {256, 256}, {512, 512}, {1024, 1024}, {2048, 2048}, {4096, 4096}
    };

    const std::string csv_path = "sobel_benchmark.csv";
    std::ofstream csv(csv_path);
    if (!csv.is_open()) {
        std::cerr << "Failed to open " << csv_path << " for writing" << std::endl;
        return 1;
    }
    csv << "iterations," << iterations << std::endl;
    print_results_header(csv);

    for (const auto& size : sizes) {
        int M = size.first;
        int N = size.second;
        std::vector<int16_t> src(M * N);

        std::mt19937 rng(M * 31 + N);
        std::uniform_int_distribution<int> dist(-512, 511);
        for (auto& px : src) {
            px = static_cast<int16_t>(dist(rng));
        }

        auto custom = time_custom_sobel(src, M, N, iterations);
        auto stage1 = time_stage1(src, M, N, iterations);
        print_results_row(csv, M, N, custom, stage1);
    }

    std::cout << "Sobel vs OpenCV stage1 results written to " << csv_path << std::endl;
    return 0;
}
