#include "../include/Sobel.hpp"
#include <cstdio>

#define MAX_FREQ 3.4
#define BASE_FREQ 2.4

#define MM 5
#define NN 18

#define NUM_ITER 500000000

//timing routine for reading the time stamp counter
static __inline__ unsigned long long rdtsc(void) {
	unsigned hi, lo;
	__asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
	return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}


int main() {
    uint16_t* Img;
    int16_t* Gx;
    int16_t* Gy;

    posix_memalign((void**) &Img, 64, MM * NN * sizeof(uint16_t));
    posix_memalign((void**) &Gx, 64, MM * NN * sizeof(int16_t));
    posix_memalign((void**) &Gy, 64, MM * NN * sizeof(int16_t));

    for (int i = 0; i < MM * NN; i++) {
    	Img[i] = (uint16_t)((i * 37 + 13) & 0xFF); // pseudo-random pattern
    }

    uint64_t r0, r1;
    uint64_t sum = 0;
    for (int i = 0; i < NUM_ITER; i++) {
        r0 = rdtsc();
        sobel(Img, MM, NN, 5, 18, Gx, Gy);
        r1 = rdtsc();
        sum += (r1 - r0);
    }
    printf("Throughput: %lf\n", ((double)(16.0 * NUM_ITER * (MM - 2) * NN) / ((double)(sum)*MAX_FREQ/BASE_FREQ)));

}