#include <algorithm>
#include <chrono>
#include <deque>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include "nms.hpp"
#include "ref.hpp"
#include "utils.hpp"

#define MAX_FREQ 3.2
#define BASE_FREQ 2.1

#define NMS_STEP_ONCE() \
    NMS_TILE(gx_ptr, gy_ptr, prev_mag_ptr, curr_mag_ptr, next_mag_ptr, res_ptr, high_threshold, map_ptr); \
    gx_ptr += 16; \
    gy_ptr += 16; \
    prev_mag_ptr += 16; \
    curr_mag_ptr += 16; \
    next_mag_ptr += 16; \
    res_ptr += 16; \
    map_ptr += 16;

#define REPEAT10_NMS() \
    NMS_STEP_ONCE(); NMS_STEP_ONCE(); NMS_STEP_ONCE(); NMS_STEP_ONCE(); NMS_STEP_ONCE(); \
    NMS_STEP_ONCE(); NMS_STEP_ONCE(); NMS_STEP_ONCE(); NMS_STEP_ONCE(); NMS_STEP_ONCE();

#define REPEAT100_NMS() \
    REPEAT10_NMS(); REPEAT10_NMS(); REPEAT10_NMS(); REPEAT10_NMS(); REPEAT10_NMS(); \
    REPEAT10_NMS(); REPEAT10_NMS(); REPEAT10_NMS(); REPEAT10_NMS(); REPEAT10_NMS();

#define REPEAT1000_NMS() \
    REPEAT100_NMS(); REPEAT100_NMS(); REPEAT100_NMS(); REPEAT100_NMS(); REPEAT100_NMS(); \
    REPEAT100_NMS(); REPEAT100_NMS(); REPEAT100_NMS(); REPEAT100_NMS(); REPEAT100_NMS(); 

void benchmark_performance()
{
	const int M = 3;      // rows
	const int N = 16;     // cols (must satisfy j+15 < N)
	const int i = 1;      // middle row (to allow i-1 and i+1 valid)
	const int j = 8;      // start column (ensure j-1 and j+15 within bounds)
    const int num_runs = 100000;
    const int NUM_INST = 120; // number of simd instructions in the kernel

	std::vector<int16_t> gx(M * N * NUM_INST, 1);  // strong x gradient
	std::vector<int16_t> gy(M * N * NUM_INST, 1);     // zero y gradient
	std::vector<int16_t> mag(M * N * NUM_INST, 2);  // baseline magnitude
	std::vector<int16_t> res(M * N * NUM_INST, 0);    // output
    std::vector<int16_t> map(M * N * NUM_INST, 0);    // map outoput
    const int16_t high_threshold = 240;
    for (int i = 0; i < M * N * NUM_INST; ++i) {
        gx[i] = (rand() % 512) - 256;
        gy[i] = (rand() % 512) - 256;
        mag[i] = abs(gx[i]) + abs(gy[i]);
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
        int16_t *prev_mag_ptr = mag.data();
        int16_t *curr_mag_ptr = mag.data() + (N * NUM_INST);
        int16_t *next_mag_ptr = mag.data() + (2 * N * NUM_INST);
        int16_t *res_ptr = res.data() + (N * NUM_INST);
        int16_t *map_ptr = map.data() + (N * NUM_INST);
        st = rdtsc();
        asm volatile("look_here:" ::: "memory");
        REPEAT100_NMS();
        REPEAT10_NMS();
        REPEAT10_NMS();
        et = rdtsc();
        dt = et - st;
        if (dt < dt_min) dt_min = dt;
        sum += dt;
    }

    printf("NMS KERNEL Throughput : %lf \n\r", 16 * ((double)NUM_INST * 24) / (dt_min * MAX_FREQ/BASE_FREQ));

}



using Clock = std::chrono::high_resolution_clock;

struct BenchmarkResult {
	std::string label;
	double avg_ms;
};

static BenchmarkResult time_custom_nms(const std::vector<int16_t>& gx,
									   const std::vector<int16_t>& gy,
									   const std::vector<int16_t>& mag,
									   int16_t high_threshold,
									   int M,
									   int N,
									   int iterations) {
	std::vector<int16_t> res(M * N);
	std::vector<int16_t> map(M * N);

	std::vector<int16_t> gx_copy = gx;
	std::vector<int16_t> gy_copy = gy;
	std::vector<int16_t> mag_copy = mag;

	auto start = Clock::now();
	for (int i = 0; i < iterations; ++i) {
		non_max_suppression(gx_copy.data(), gy_copy.data(), mag_copy.data(), high_threshold,
							res.data(), map.data(), M, N);
	}
	auto end = Clock::now();
	double avg_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
	return {"custom_nms", avg_ms};
}

static BenchmarkResult time_stage2(const std::vector<int16_t>& gx,
								   const std::vector<int16_t>& gy,
								   int low_threshold,
								   int high_threshold,
								   int M,
								   int N,
								   int iterations) {
	cv::Mat dx(M, N, CV_16S, const_cast<int16_t*>(gx.data()));
	cv::Mat dy(M, N, CV_16S, const_cast<int16_t*>(gy.data()));
	cv::Mat map;
	std::deque<uchar*> borderPeaksParallel;
	std::deque<uchar*> stack;
	cv::customizedCanny canny(dx, dy, map, borderPeaksParallel, low_threshold, high_threshold, false);
	cv::Range all_rows(0, M);

	auto start = Clock::now();
	for (int i = 0; i < iterations; ++i) {
		std::deque<uchar*> localStack, localBorderPeaks;
		canny.stage2(all_rows, dx, dy, localStack, localBorderPeaks);
	}
	auto end = Clock::now();
	double avg_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
	return {"opencv_stage2", avg_ms};
}

static void print_results_header(std::ostream& out) {
	out << "M,N,custom_ms,opencv_ms,speedup(stage2/custom)" << std::endl;
}

static void print_results_row(std::ostream& out, int M, int N, const BenchmarkResult& custom, const BenchmarkResult& stage2) {
	double speedup = stage2.avg_ms / custom.avg_ms;
	out << M << "," << N << ',' << custom.avg_ms << ',' << stage2.avg_ms << ',' << speedup << std::endl;
}

int main(int argc, char** argv) {
	cv::setNumThreads(1);
    cv::ocl::setUseOpenCL(false);
	benchmark_performance();

	int iterations = 20;
	int low_threshold = 50;
	int high_threshold = 150;

	if (argc >= 2) {
		iterations = std::stoi(argv[1]);
	}

	std::vector<std::pair<int, int>> sizes = {
		{128, 128}, {512, 512}, {1024, 1024}, {2048, 2048}, {4096, 4096}, {8192, 8192}
	};

	const std::string csv_path = "nms_benchmark.csv";
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
		std::vector<int16_t> gx(M * N);
		std::vector<int16_t> gy(M * N);
		std::vector<int16_t> mag(M * N);

		std::mt19937 rng(M * 17 + N);
		std::uniform_int_distribution<int> dist(-512, 512);
		for (int i = 0; i < M * N; ++i) {
			gx[i] = static_cast<int16_t>(dist(rng));
			gy[i] = static_cast<int16_t>(dist(rng));
			mag[i] = static_cast<int16_t>(std::abs(gx[i]) + std::abs(gy[i]));
		}

		auto custom = time_custom_nms(gx, gy, mag, static_cast<int16_t>(high_threshold), M, N, iterations);
		auto stage2 = time_stage2(gx, gy, low_threshold, high_threshold, M, N, iterations);
		print_results_row(csv, M, N, custom, stage2);
	}

	std::cout << "Comparison results over OpenCV written to " << csv_path << std::endl;

	return 0;
}
