// Test for NMS_KERNEL_1x16 macro
// Validates retention of a single local maximum in a controlled gradient scenario.

#include <cstdint>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include "nms.hpp"
#include "utils.hpp"
static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}


static int failures = 0;
static void check(bool cond, const char* msg) {
	if(!cond) { ++failures; std::cerr << "FAIL: " << msg << std::endl; }
}

#define MAX_FREQ 3.2
#define BASE_FREQ 2.1

#define NMS_STEP_ONCE() \
    NMS_KERNEL_1x16(gx_ptr, gy_ptr, mag_ptr, res_ptr, M, N); \
    res_ptr += (M)*(N); \

#define REPEAT10_NMS() \
    NMS_STEP_ONCE(); NMS_STEP_ONCE(); NMS_STEP_ONCE(); NMS_STEP_ONCE(); NMS_STEP_ONCE(); \
    NMS_STEP_ONCE(); NMS_STEP_ONCE(); NMS_STEP_ONCE(); NMS_STEP_ONCE(); NMS_STEP_ONCE();

#define REPEAT100_NMS() \
    REPEAT10_NMS(); REPEAT10_NMS(); REPEAT10_NMS(); REPEAT10_NMS(); REPEAT10_NMS(); \
    REPEAT10_NMS(); REPEAT10_NMS(); REPEAT10_NMS(); REPEAT10_NMS(); REPEAT10_NMS();

#define REPEAT1000_NMS() \
    REPEAT100_NMS(); REPEAT100_NMS(); REPEAT100_NMS(); REPEAT100_NMS(); REPEAT100_NMS(); \
    REPEAT100_NMS(); REPEAT100_NMS(); REPEAT100_NMS(); REPEAT100_NMS(); REPEAT100_NMS(); 

#define REPEAT10000_NMS() \
    REPEAT1000_NMS(); REPEAT1000_NMS(); REPEAT1000_NMS(); REPEAT1000_NMS(); REPEAT1000_NMS(); \
    REPEAT1000_NMS(); REPEAT1000_NMS(); REPEAT1000_NMS(); REPEAT1000_NMS(); REPEAT1000_NMS();

int main() {
	const int M = 3;      // rows
	const int N = 16;     // cols (must satisfy j+15 < N)
	const int i = 1;      // middle row (to allow i-1 and i+1 valid)
	const int j = 8;      // start column (ensure j-1 and j+15 within bounds)
    const int num_runs = 100000;
    const int NUM_INST = 10000; // number of simd instructions in the kernel

	std::vector<int16_t> gx(M * N * NUM_INST, 1);  // strong x gradient
	std::vector<int16_t> gy(M * N * NUM_INST, 1);     // zero y gradient
	std::vector<int16_t> mag(M * N * NUM_INST, 2);  // baseline magnitude
	std::vector<int16_t> res(M * N * NUM_INST, 0);    // output
    std::vector<int16_t> exp(M * N * NUM_INST, 0);    // expected output

    for (int i = 0; i < M * N * NUM_INST; ++i) {
        gx[i] = (rand() % 512) - 256;
        gy[i] = (rand() % 512) - 256;
        mag[i] = abs(gx[i]) + abs(gy[i]);
    }
    std::ofstream log_file("log.txt");

    log_file << "Input gx:" << std::endl;
	for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            log_file << "(" << setw(6) << gx[i*N + j] << ", " << setw(6) << gy[i*N + j] << ") ";
        }
        log_file << std::endl;
    }
    int16_t* mag_ptr = mag.data() + N;
    int16_t* res_ptr = res.data() + N;
    int16_t* gx_ptr = gx.data() + N;
    int16_t* gy_ptr = gy.data() + N;
    int16_t* exp_ptr = exp.data() + N;
    NMS_KERNEL_REF(gx_ptr, gy_ptr, mag_ptr, exp_ptr, M, N);
    NMS_KERNEL_1x16(gx_ptr, gy_ptr, mag_ptr, res_ptr, M, N);
    for (int idx = 0; idx < M * N; ++idx) {
        check(res[idx] == exp[idx], "NMS_KERNEL_1x16 output mismatch");
    }
    log_file << "Actual output:" << std::endl;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            log_file << setw(6) << res[i*N + j] << " ";
        }
        log_file << std::endl;
    }
    log_file << "Expected output:" << std::endl;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            log_file << setw(6) << exp[i*N + j] << " ";
        }
        log_file << std::endl;
    }


	// Macro helpers: repeat the kernel 100 times without an explicit loop.
	// Each iteration advances all pointers by M*N elements.
    unsigned long long st;
    unsigned long long et;
    unsigned long long dt; 
    unsigned long long dt_min = (unsigned long long)-1;
    unsigned long long sum = 0;
    for (int run = 0; run < num_runs; ++run) {
        mag_ptr = mag.data();
        res_ptr = res.data();
        gx_ptr = gx.data();
        gy_ptr = gy.data();
        st = rdtsc();
        REPEAT10000_NMS();
        et = rdtsc();
        dt = et - st;
        if (dt < dt_min) dt_min = dt;
        sum += dt;
    }

    printf("Throughput : %lf \n\r", 11 * ((double)NUM_INST * 16) / (dt_min * MAX_FREQ/BASE_FREQ));


	if(failures == 0) {
		std::cout << "NMS_KERNEL_1x16 test passed" << std::endl;
		return 0;
	}
	std::cerr << failures << " failure(s)" << std::endl;
	return 1;
}

