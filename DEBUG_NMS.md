# NMS Debugging Guide

## Quick Start

Run the debug tests:
```bash
cd build
make test_nms
./test_nms
```

## Debugging Strategy

### 1. **Start with Simple Tests (No OpenCV required)**

The test file now includes `debug_simple_test()` which:
- Uses a tiny 5×32 image
- Creates a controlled edge pattern
- Tests your NMS function in isolation
- Prints input and output for visual inspection

**What to check:**
- Does the output make sense visually?
- Are edge pixels preserved where expected?
- Are non-edge pixels suppressed?

### 2. **Test Individual Kernels**

Use `debug_kernel_test()` to:
- Test `NMS_TILE` macro on 16 pixels at a time
- Compare against reference implementation
- Identify if the SIMD kernel has issues

**What to check:**
- Does `NMS_TILE` produce same results as scalar reference?
- Print intermediate SIMD register values if needed

### 3. **Debug Individual Pixels**

Use the helper function `debug_nms_pixel()` to:
```cpp
// Compute NMS for pixel (i,j) with verbose output
int16_t result = debug_nms_pixel(gx.data(), gy.data(), mag.data(), i, j, N, true);
```

This prints:
- Gradient values (gx, gy, magnitude)
- Computed direction (0°, 45°, 90°, 135°)
- Neighbor comparison values
- Whether pixel is local maximum

### 4. **Visualize Data**

Use `print_region()` to see neighborhoods:
```cpp
print_region("Magnitude", mag.data(), M, N, row, col, radius);
```

### 5. **Compare with OpenCV Reference**

Once basic tests pass, use `function_correctness_test()`:
- Compares your NMS against OpenCV's `stage2`
- Shows first 3 mismatches in detail
- Runs manual computation for mismatched pixels

## Common Issues to Check

### Issue 1: Edge Handling
- Are you handling borders correctly?
- Check first/last rows and columns
- Use `NMS_EDGE` macro for borders

### Issue 2: Block Processing
- Is `BLOCK_WIDTH` alignment correct (multiple of 16)?
- Are pointers initialized correctly for each block?
- Check block boundary conditions

### Issue 3: Direction Computation
```cpp
const int TG22 = 13573;  // tan(22.5°) * 2^15
int x = std::abs(gx);
int y = std::abs(gy) << 15;  // Must shift left!
int tg22x = x * TG22;
int tg67x = tg22x + (x << 16);
```
- Verify the bit shifts are correct
- Check angle threshold comparisons

### Issue 4: Neighbor Selection
For diagonal directions:
```cpp
int s = (gx ^ gy) < 0 ? 1 : -1;
// s=1: check (i-1,j-1) and (i+1,j+1) for 135°
// s=-1: check (i-1,j+1) and (i+1,j-1) for 45°
```

### Issue 5: Pointer Arithmetic
In `non_max_suppression`:
```cpp
prev_mag_ptr = mag + (i-1) * N + j;
curr_mag_ptr = mag + i * N + j;
next_mag_ptr = mag + (i+1) * N + j;
```
- Verify all pointer offsets
- Check that you're not accessing out of bounds

### Issue 6: SIMD Alignment
- Data must be properly aligned for AVX2
- Use `_mm256_loadu_si256` for unaligned loads
- Check that indices are correct when using SIMD

## Debugging Workflow

### Step 1: Visual Inspection
```bash
./test_nms > debug_output.txt
```
Look at the printed matrices - do they make sense?

### Step 2: Isolate the Problem
- If simple test fails → issue in core algorithm
- If kernel test fails → issue in SIMD implementation
- If OpenCV test fails → issue with integration

### Step 3: Add Instrumentation
Add prints in your `non_max_suppression` function:
```cpp
if (i == 10 && j == 10) {  // Debug specific pixel
    printf("Processing (%d,%d): gx=%d, gy=%d, mag=%d\n", 
           i, j, gx_ptr[0], gy_ptr[0], curr_mag_ptr[0]);
}
```

### Step 4: Compare Intermediate Values
Save intermediate results and compare with reference:
```cpp
// In your function
std::ofstream debug_file("nms_debug.txt");
for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
        debug_file << res[i*N + j] << " ";
    }
    debug_file << "\n";
}
```

### Step 5: Unit Test Each Component
Test separately:
1. Direction computation
2. Neighbor comparison
3. Block iteration logic
4. SIMD kernel
5. Edge handling

## Using GDB

```bash
gdb ./test_nms
(gdb) break non_max_suppression
(gdb) run
(gdb) print i
(gdb) print j
(gdb) print mag[i*N + j]
(gdb) print mag[i*N + j - 1]
(gdb) print mag[i*N + j + 1]
```

## Verification Checklist

- [ ] Simple test passes (controlled pattern)
- [ ] Kernel test passes (SIMD vs scalar match)
- [ ] Borders handled correctly
- [ ] All four directions tested (0°, 45°, 90°, 135°)
- [ ] No out-of-bounds memory access (run with valgrind)
- [ ] Results close to OpenCV reference (< 1% difference)

## Performance Testing

Once correctness is verified:
```cpp
auto start = std::chrono::high_resolution_clock::now();
non_max_suppression(gx, gy, mag, threshold, res, map, M, N);
auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
std::cout << "Time: " << duration.count() << " µs" << std::endl;
```

## Additional Tools

1. **Valgrind** - check memory errors:
```bash
valgrind --leak-check=full ./test_nms
```

2. **Address Sanitizer** - add to CMakeLists.txt:
```cmake
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
```

3. **Print assembly** - verify SIMD code generation:
```bash
g++ -S -masm=intel -O3 -mavx2 test_nms.cpp
```
