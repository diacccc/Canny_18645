#include <algorithm>
#include <chrono>
#include <deque>
#include <iostream>
#include <random>
#include <vector>

#include "nms.hpp"
#include "ref.hpp"

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

static void print_results_header(int iterations) {
	std::cout << "NMS Performance Benchmark" << std::endl;
	std::cout << "Iterations per size: " << iterations << std::endl;
	std::cout << "size\tcustom_ms\topencv_ms\tspeedup(stage2/custom)" << std::endl;
}

static void print_results_row(int M, int N, const BenchmarkResult& custom, const BenchmarkResult& stage2) {
	double speedup = stage2.avg_ms / custom.avg_ms;
	std::cout << std::setw(4) << M << "x" << std::setw(4) << N << "\t" << custom.avg_ms << "\t" << stage2.avg_ms << "\t" << speedup << std::endl;
}

int main(int argc, char** argv) {
	int iterations = 20;
	int low_threshold = 50;
	int high_threshold = 150;

	if (argc >= 2) {
		iterations = std::stoi(argv[1]);
	}

	std::vector<std::pair<int, int>> sizes = {
		{128, 128}, {512, 512}, {1024, 1024}, {2048, 2048}, {4096, 4096}, {8192, 8192}
	};

	print_results_header(iterations);

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
		print_results_row(M, N, custom, stage2);
	}

	return 0;
}
