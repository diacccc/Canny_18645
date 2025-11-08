#include <cstdint>
#include <vector>
#include <iostream>
#include <string>

#include "kernel.hpp"

static int failures = 0;

static void check(bool cond, const std::string& msg)
{
    if (!cond)
    {
        ++failures;
        std::cerr << "FAIL: " << msg << std::endl;
    }
}

// Helper to run SOBEL_3x3 at a specific (x,y) on a flat buffer
static void run_sobel_at(const std::vector<uint8_t>& src, int width, int height, int x, int y,
                         std::vector<int16_t>& gx, std::vector<int16_t>& gy)
{
    const int step = width; // stride in elements
    const int idx  = y * step + x;
    SOBEL_3x3(gx, gy, src, step, idx);
}

int main()
{
    // Test 1: Constant image -> zero gradients
    {
        const int W = 5, H = 5;
        std::vector<uint8_t> src(W * H, 100);
        std::vector<int16_t> gx(W * H, 0), gy(W * H, 0);

        // Choose a center pixel away from borders
        run_sobel_at(src, W, H, 2, 2, gx, gy);

        check(gx[2 + 2 * W] == 0, "Constant image: gx should be 0");
        check(gy[2 + 2 * W] == 0, "Constant image: gy should be 0");
    }

    // Test 2: Horizontal edge (top=0, bottom=255)
    // According to the macro, gx uses (top row sum) - (bottom row sum), so it should be negative.
    // Expected magnitude: (1 + 2 + 1) * 255 = 1020
    {
        const int W = 5, H = 5;
        std::vector<uint8_t> src(W * H, 0);
        // bottom half to 255
        for (int y = H / 2; y < H; ++y)
            for (int x = 0; x < W; ++x)
                src[y * W + x] = 255;

        std::vector<int16_t> gx(W * H, 0), gy(W * H, 0);

        // Evaluate at the center row right above/below the edge; choose (2,2)
        run_sobel_at(src, W, H, 2, 2, gx, gy);

        const int idx = 2 + 2 * W;
        const int16_t expected_gx = -1020; // top(0) - bottom(1020)
        const int16_t expected_gy = 0;     // symmetric left/right columns

        check(gx[idx] == expected_gx, "Horizontal edge: gx should be -1020");
        check(gy[idx] == expected_gy, "Horizontal edge: gy should be 0");
    }

    // Test 3: Vertical edge (left=0, right=255)
    // According to the macro, gy uses (right column sum) - (left column sum), so it should be positive.
    // Expected magnitude: 1020
    {
        const int W = 5, H = 5;
        std::vector<uint8_t> src(W * H, 0);
        // right half to 255
        for (int y = 0; y < H; ++y)
            for (int x = W / 2; x < W; ++x)
                src[y * W + x] = 255;

        std::vector<int16_t> gx(W * H, 0), gy(W * H, 0);

        run_sobel_at(src, W, H, 2, 2, gx, gy);

        const int idx = 2 + 2 * W;
        const int16_t expected_gx = 0;      // symmetric top/bottom rows
        const int16_t expected_gy = 1020;    // right(1020) - left(0)

        check(gx[idx] == expected_gx, "Vertical edge: gx should be 0");
        check(gy[idx] == expected_gy, "Vertical edge: gy should be 1020");
    }

    // Test 4: Diagonal edge (top-left 0, bottom-right 255), expect both gx and gy non-zero
    {
        const int W = 5, H = 5;
        std::vector<uint8_t> src(W * H, 0);
        // Set a diagonal split: pixels with x+y >= W (roughly main anti-diagonal) to 255
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                if (x + y >= W)
                    src[y * W + x] = 255;

        std::vector<int16_t> gx(W * H, 0), gy(W * H, 0);

        run_sobel_at(src, W, H, 2, 2, gx, gy);

        const int idx = 2 + 2 * W;
        // We don't assert exact values here (depends on the diagonal pattern), only non-zero gradients.
        check(gx[idx] != 0, "Diagonal edge: gx should be non-zero");
        check(gy[idx] != 0, "Diagonal edge: gy should be non-zero");
    }

    if (failures == 0)
    {
        std::cout << "All SOBEL_3x3 tests passed" << std::endl;
        return 0;
    }
    std::cerr << failures << " test(s) failed" << std::endl;
    return 1;
}
