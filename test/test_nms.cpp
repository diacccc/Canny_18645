// Test for NMS_KERNEL_1x16 macro
// Validates retention of a single local maximum in a controlled gradient scenario.

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
#include "nms.hpp"
#include "utils.hpp"
#include "ref.hpp"

#if (defined(CV_SIMD) && CV_SIMD) || (defined(CV_SIMD_SCALABLE) && CV_SIMD_SCALABLE)
static constexpr int kStage2MapOffset = CV_SIMD_WIDTH;
#else
static constexpr int kStage2MapOffset = 1;
#endif

static int failures = 0;
static void check(bool cond, const char* msg) {
	if(!cond) { ++failures; std::cerr << "FAIL: " << msg << std::endl; }
}


// Debug helper: Print a region of data
void print_region(const char* label, const int16_t* data, int M, int N, int row, int col, int radius = 2) {
    std::cout << "\n" << label << " around (" << row << "," << col << "):" << std::endl;
    for (int i = std::max(0, row - radius); i <= std::min(M - 1, row + radius); ++i) {
        for (int j = std::max(0, col - radius); j <= std::min(N - 1, col + radius); ++j) {
            if (i == row && j == col)
                std::cout << "[" << std::setw(5) << data[i * N + j] << "] ";
            else
                std::cout << " " << std::setw(5) << data[i * N + j] << "  ";
        }
        std::cout << std::endl;
    }
}

// Debug helper: Visualize gradient direction at a pixel
void print_gradient_info(int16_t gx, int16_t gy, int16_t mag, int row, int col) {
    std::cout << "\nGradient at (" << row << "," << col << "):" << std::endl;
    std::cout << "  gx=" << gx << ", gy=" << gy << ", mag=" << mag << std::endl;
    
    int x = std::abs(gx);
    int y = std::abs(gy) << 15;
    int tg22x = x * 13573;
    int tg67x = tg22x + (x << 16);
    
    std::cout << "  |gx|=" << x << ", |gy|<<15=" << y << std::endl;
    std::cout << "  tg22x=" << tg22x << ", tg67x=" << tg67x << std::endl;
    
    if (y < tg22x) {
        std::cout << "  Direction: 0° (horizontal)" << std::endl;
    } else if (y > tg67x) {
        std::cout << "  Direction: 90° (vertical)" << std::endl;
    } else {
        int s = (gx ^ gy) < 0 ? 1 : -1;
        if (s == 1)
            std::cout << "  Direction: 135° (diagonal \\)" << std::endl;
        else
            std::cout << "  Direction: 45° (diagonal /)" << std::endl;
    }
}

// Debug helper: Manually compute NMS for a single pixel
int16_t debug_nms_pixel(const int16_t* gx, const int16_t* gy, const int16_t* mag, 
                        int M, int N, int i, int j, bool verbose = false) {
    const int TG22 = 13573;
    int16_t gx_val = gx[i * N + j];
    int16_t gy_val = gy[i * N + j];
    int16_t mag_val = mag[i * N + j];
    
    if (verbose) {
        print_gradient_info(gx_val, gy_val, mag_val, i, j);
        print_region("Magnitude", mag, M, N, i, j, 2);
    }
    
    int x = std::abs(gx_val);
    int y = std::abs(gy_val) << 15;
    int tg22x = x * TG22;
    
    if (y < tg22x) {
        // 0° direction - check left and right
        if (verbose) std::cout << "  Comparing with left=" << mag[i*N + j-1] 
                               << ", right=" << mag[i*N + j+1] << std::endl;
        if (mag_val > mag[i*N + j-1] && mag_val > mag[i*N + j+1]) {
            if (verbose) std::cout << "  -> LOCAL MAX" << std::endl;
            return mag_val;
        }
    } else {
        int tg67x = tg22x + (x << 16);
        if (y > tg67x) {
            // 90° direction - check up and down
            if (verbose) std::cout << "  Comparing with up=" << mag[(i-1)*N + j] 
                                   << ", down=" << mag[(i+1)*N + j] << std::endl;
            if (mag_val > mag[(i-1)*N + j] && mag_val > mag[(i+1)*N + j]) {
                if (verbose) std::cout << "  -> LOCAL MAX" << std::endl;
                return mag_val;
            }
        } else {
            // 45° or 135° direction
            int s = (gx_val ^ gy_val) < 0 ? 1 : -1;
            if (verbose) {
                if (s == 1)
                    std::cout << "  Comparing with diag1=" << mag[(i-1)*N + j-1] 
                              << ", diag2=" << mag[(i+1)*N + j+1] << std::endl;
                else
                    std::cout << "  Comparing with diag1=" << mag[(i-1)*N + j+1] 
                              << ", diag2=" << mag[(i+1)*N + j-1] << std::endl;
            }
            if (mag_val > mag[(i-1)*N + j-s] && mag_val > mag[(i+1)*N + j+s]) {
                if (verbose) std::cout << "  -> LOCAL MAX" << std::endl;
                return mag_val;
            }
        }
    }
    if (verbose) std::cout << "  -> SUPPRESSED" << std::endl;
    return 0;
}

static void nms_edge_reference_block(const int16_t* gx, const int16_t* gy,
                                        const int16_t* prev_mag, const int16_t* curr_mag,
                                        const int16_t* next_mag, int len, int pos,
                                        int16_t* dst) {
    const int TG22 = 13573;
    for (int ii = 0; ii < len; ++ii) {
        int16_t gx_val = gx[ii];
        int16_t gy_val = gy[ii];
        int16_t mag_val = curr_mag[ii];

        dst[ii] = 0;

        int x = std::abs(gx_val);
        int y = std::abs(gy_val) << 15;
        int tg22x = x * TG22;

        if (y < tg22x) {
            bool left_pass = (pos == 1 && ii == 0) ? true : (mag_val > curr_mag[ii - 1]);
            bool right_pass = (pos == -1 && ii == len - 1) ? true : (mag_val > curr_mag[ii + 1]);
            if (left_pass && right_pass) {
                dst[ii] = mag_val;
            }
        } else {
            int tg67x = tg22x + (x << 16);
            if (y > tg67x) {
                bool up_pass = mag_val > prev_mag[ii];
                bool down_pass = mag_val >= next_mag[ii];
                if (up_pass && down_pass) {
                    dst[ii] = mag_val;
                }
            } else {
                int s = (gx_val ^ gy_val) < 0 ? 1 : -1;
                if (s == 1) {
                    bool diag0 = (pos == 1 && ii == 0) ? true : (mag_val > prev_mag[ii - 1]);
                    bool diag1 = (pos == -1 && ii == len - 1) ? true : (mag_val > next_mag[ii + 1]);
                    if (diag0 && diag1) {
                        dst[ii] = mag_val;
                    }
                } else {
                    bool diag0 = (pos == -1 && ii == len - 1) ? true : (mag_val > prev_mag[ii + 1]);
                    bool diag1 = (pos == 1 && ii == 0) ? true : (mag_val > next_mag[ii - 1]);
                    if (diag0 && diag1) {
                        dst[ii] = mag_val;
                    }
                }
            }
        }
    }
}

void nms_tile_unit_test() {
    std::cout << "\n=== NMS_TILE Unit Test ===" << std::endl;

    const int M = 6;
    const int N = 48;  // ensure room for 16-wide tiles with halo

    std::vector<int16_t> gx(M * N);
    std::vector<int16_t> gy(M * N);
    std::vector<int16_t> mag(M * N);
    std::vector<int16_t> res_tile(M * N, 0);
    std::vector<int16_t> map_tile(M * N, 0);
    const int16_t high_threshold = 100;
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dist(-255, 255);

    for (int idx = 0; idx < M * N; ++idx) {
        gx[idx] = static_cast<int16_t>(dist(rng));
        gy[idx] = static_cast<int16_t>(dist(rng));
        mag[idx] = static_cast<int16_t>(std::abs(gx[idx]) + std::abs(gy[idx]));
    }

    int tiles_tested = 0;
    int mismatches = 0;

    for (int i = 1; i < M - 1; ++i) {
        for (int j = 1; j <= N - 17; j += 16) {
            const int16_t* gx_ptr = gx.data() + i * N + j;
            const int16_t* gy_ptr = gy.data() + i * N + j;
            const int16_t* prev_mag_ptr = mag.data() + (i - 1) * N + j;
            const int16_t* curr_mag_ptr = mag.data() + i * N + j;
            const int16_t* next_mag_ptr = mag.data() + (i + 1) * N + j;
            int16_t* res_ptr = res_tile.data() + i * N + j;
            int16_t* map_ptr = map_tile.data() + i * N + j;

            NMS_TILE(gx_ptr, gy_ptr, prev_mag_ptr, curr_mag_ptr, next_mag_ptr, res_ptr, high_threshold, map_ptr);
            ++tiles_tested;

            for (int k = 0; k < 16; ++k) {
                int col = j + k;
                int16_t expected = debug_nms_pixel(gx.data(), gy.data(), mag.data(), M, N, i, col, false);
                int16_t actual = res_tile[i * N + col];
                if (actual != expected) {
                    if (mismatches < 5) {
                        debug_nms_pixel(gx.data(), gy.data(), mag.data(), M, N, i, col, true);
                        std::cout << "Mismatch at (" << i << "," << col << ") "
                                  << "actual=" << actual << ", expected=" << expected << std::endl;
                    }
                    ++mismatches;
                }
            }
        }
    }

    std::cout << "  Tiles tested: " << tiles_tested << std::endl;
    printf(
        "  Mismatches: %d (%.2f%%)\n",
        mismatches,
        (tiles_tested * 16 > 0) ? (100.0 * mismatches / (tiles_tested * 16)) : 0.0
    );
    check(mismatches == 0, "NMS_TILE matches scalar reference");
}

void non_max_suppression_unit_test() {
    std::cout << "\n=== non_max_suppression Unit Test ===" << std::endl;

    const int M = 32;
    const int N = 64;
    const int high_threshold = 200;

    std::vector<int16_t> gx(M * N);
    std::vector<int16_t> gy(M * N);
    std::vector<int16_t> mag(M * N);
    std::vector<int16_t> mag_backup(M * N);
    std::vector<int16_t> res(M * N, 0);
    std::vector<int16_t> map(M * N, 0);

    std::mt19937 rng(24680);
    std::uniform_int_distribution<int> dist(-400, 400);

    for (int idx = 0; idx < M * N; ++idx) {
        gx[idx] = static_cast<int16_t>(dist(rng));
        gy[idx] = static_cast<int16_t>(dist(rng));
        int mag_val = std::abs(gx[idx]) + std::abs(gy[idx]);
        mag[idx] = static_cast<int16_t>(mag_val);
    }

    mag_backup = mag;

    non_max_suppression(gx.data(), gy.data(), mag.data(), high_threshold, res.data(), map.data(), M, N);

    int mismatches = 0;
    int mag_changes = 0;
    const int interior_pixels = (M - 2) * (N - 2);

    for (int i = 1; i < M - 1; ++i) {
        for (int j = 1; j < N - 1; ++j) {
            int idx = i * N + j;
            int16_t expected = debug_nms_pixel(gx.data(), gy.data(), mag_backup.data(), M, N, i, j, false);
            if (res[idx] != expected) {
                if (mismatches < 5) {
                    debug_nms_pixel(gx.data(), gy.data(), mag_backup.data(), M, N, i, j, true);
                    std::cout << "Mismatch at (" << i << "," << j << ") "
                              << "actual=" << res[idx] << ", expected=" << expected << std::endl;
                }
                ++mismatches;
            }
        }
    }
    // std::cout << "\nMagnitude after NMS:" << std::endl;
    // for (int i = 1; i < M - 1; ++i) {
    //     for (int j = 1; j < N - 1; ++j) {
    //         std::cout << std::setw(5) << mag[i * N + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // std::cout << "\nNMS Result:" << std::endl;
    // for (int i = 1; i < M - 1; ++i) {
    //     for (int j = 1; j < N - 1; ++j) {
    //         std::cout << std::setw(5) << res[i * N + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // std::cout << "\n Ref Result:" << std::endl;
    // for (int i = 1; i < M - 1; ++i)
    // {
    //     for (int j = 1; j < N - 1; ++j)
    //     {
    //         int16_t expected = debug_nms_pixel(gx.data(), gy.data(), mag_backup.data(), M, N, i, j, false);
    //         std::cout << std::setw(5) << expected << " ";
    //     }
    //     std::cout << std::endl;
    // }

    for (size_t idx = 0; idx < mag.size(); ++idx) {
        if (mag[idx] != mag_backup[idx]) {
            if (mag_changes < 5) {
                std::cout << "Magnitude modified at index " << idx
                          << ": original=" << mag_backup[idx]
                          << ", now=" << mag[idx] << std::endl;
            }
            ++mag_changes;
        }
    }

    std::cout << "  Interior pixels checked: " << interior_pixels << std::endl;
    std::cout << "  Mismatches: " << mismatches << std::endl;
    std::cout << "  Magnitude changes: " << mag_changes << std::endl;

    check(mismatches == 0, "non_max_suppression matches scalar reference");
    check(mag_changes == 0, "non_max_suppression does not alter magnitude");
}


void nms_edge_unit_test() {
    std::cout << "\n=== NMS_EDGE Unit Test ===" << std::endl;

    const int M = 7;
    const int N = 40;
    const int len_first = 3;
    const int len_last = 4;

    std::vector<int16_t> gx(M * N);
    std::vector<int16_t> gy(M * N);
    std::vector<int16_t> mag(M * N);
    std::vector<int16_t> map_tile(M * N, 0);
    const int16_t high_threshold = 100;
    std::mt19937 rng(67890);
    std::uniform_int_distribution<int> dist(-300, 300);

    for (int idx = 0; idx < M * N; ++idx) {
        gx[idx] = static_cast<int16_t>(dist(rng));
        gy[idx] = static_cast<int16_t>(dist(rng));
        mag[idx] = static_cast<int16_t>(std::abs(gx[idx]) + std::abs(gy[idx]));
    }

    std::vector<int16_t> res_first(len_first);
    std::vector<int16_t> ref_first(len_first);
    std::vector<int16_t> res_last(len_last);
    std::vector<int16_t> ref_last(len_last);

    int mismatches = 0;
    int rows_tested = 0;

    for (int i = 1; i < M - 1; ++i) {
        const int16_t* gx_row = gx.data() + i * N;
        const int16_t* gy_row = gy.data() + i * N;
        const int16_t* prev_row = mag.data() + (i - 1) * N;
        const int16_t* curr_row = mag.data() + i * N;
        const int16_t* next_row = mag.data() + (i + 1) * N;
        int16_t* map_row = map_tile.data() + i * N;

        std::fill(res_first.begin(), res_first.end(), 0);
        std::fill(res_last.begin(), res_last.end(), 0);

        NMS_EDGE(gx_row, gy_row, prev_row, curr_row, next_row, res_first.data(), high_threshold, map_row, len_first, 1);
        nms_edge_reference_block(gx_row, gy_row, prev_row, curr_row, next_row, len_first, 1, ref_first.data());

        int start_last = N - len_last;
        NMS_EDGE(gx_row + start_last, gy_row + start_last,
                 prev_row + start_last, curr_row + start_last, next_row + start_last,
                 res_last.data(), high_threshold, map_row + start_last, len_last, -1);
        nms_edge_reference_block(gx_row + start_last, gy_row + start_last,
                                 prev_row + start_last, curr_row + start_last, next_row + start_last,
                                 len_last, -1, ref_last.data());

        for (int ii = 0; ii < len_first; ++ii) {
            if (res_first[ii] != ref_first[ii]) {
                if (mismatches < 5) {
                    std::cout << "First-edge mismatch at row " << i
                              << ", col " << ii
                              << ": actual=" << res_first[ii]
                              << ", expected=" << ref_first[ii] << std::endl;
                }
                ++mismatches;
            }
        }

        for (int ii = 0; ii < len_last; ++ii) {
            int col = start_last + ii;
            if (res_last[ii] != ref_last[ii]) {
                if (mismatches < 5) {
                    std::cout << "Last-edge mismatch at row " << i
                              << ", col " << col
                              << ": actual=" << res_last[ii]
                              << ", expected=" << ref_last[ii] << std::endl;
                }
                ++mismatches;
            }
        }

        ++rows_tested;
    }

    const int total_pixels = rows_tested * (len_first + len_last);
    std::cout << "  Rows tested: " << rows_tested << std::endl;
    std::cout << "  Edge pixels checked: " << total_pixels << std::endl;
    std::cout << "  Mismatches: " << mismatches << std::endl;
    check(mismatches == 0, "NMS_EDGE matches scalar reference");
}



int main()
{
    nms_edge_unit_test();
    nms_tile_unit_test();
    non_max_suppression_unit_test();
    return failures;
}

