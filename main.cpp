#include "ref.hpp"
#include "utils.hpp"

using namespace cv;
using namespace std;
using namespace chrono;



/* End of file. */

PerformanceResult benchmarkCanny(int width, int height, int iterations = 10000) {
    PerformanceResult result;
    result.width = width;
    result.height = height;
    result.pixels = width * height;
    
    // Generate synthetic grayscale image with random content
    Mat gray(height, width, CV_8UC1);
    randu(gray, Scalar(0), Scalar(255));
    
    // Add some shapes for more realistic edge detection
    for (int i = 0; i < min(10, width/50); i++) {
        circle(gray, Point(rand() % width, rand() % height), 
               rand() % (min(width, height)/10) + 10, Scalar(255), 2);
        rectangle(gray, 
                 Point(rand() % width, rand() % height),
                 Point(rand() % width, rand() % height),
                 Scalar(128), 2);
    }
    
    vector<double> times;

    Mat map;
    // Mat dx, dy;
    std::deque<uchar*> borderPeaksParallel, stack;
    cv::setNumThreads(1);    
    
    Mat dx, dy;
    // customizedCanny cc(gray, map, stack, 6, 10, 3, false);
    // cc.stage1(Range(0, gray.rows), dx, dy);
    // cc.stage2(Range(0, gray.rows), dx, dy, stack, borderPeaksParallel);
    // Warm-up runs
    for (int i = 0; i < 5; i++) {
        // Canny(gray, map, 50, 150, 3, false);
        // std::deque<uchar*> stack_copy = stack;
        // cc.stage3(Range(0, gray.rows), stack_copy, borderPeaksParallel);
        customizedCanny cc(gray, map, stack, 6, 10, 3, false);
        cc(Range(0, gray.rows));
    }
    
    // Actual benchmark
    for (int i = 0; i < iterations; i++) {
        auto start = high_resolution_clock::now();
        customizedCanny cc(gray, map, stack, 6, 10, 3, false);
        cc(Range(0, gray.rows));
        // Canny(gray, map, 50, 150, 3, false);
        
        // cc.stage3(Range(0, gray.rows), stack_copy, borderPeaksParallel);
        auto end = high_resolution_clock::now();
        duration<double, milli> elapsed = end - start;
        times.push_back(elapsed.count());
    }
    
    // Calculate statistics
    result.min_time_ms = *min_element(times.begin(), times.end());
    result.max_time_ms = *max_element(times.begin(), times.end());
    result.avg_time_ms = calculateAverage(times);
    result.throughput_fps = 1000.0 / result.avg_time_ms;
    
    return result;
}



int main(int argc, char** argv) {
    int iterations = 100;

    if (argc > 1) {
        iterations = atoi(argv[1]);
    }
    
    // Define test sizes
    vector<pair<int, int>> testSizes = {
        {128, 128},
        {128, 256},
        {256, 256},
        {512, 512},
        {1024, 1024},
        {2048, 2048}
    };
    
    vector<PerformanceResult> results;
    
    cout << "OpenCV Canny Edge Detection Performance Test" << endl;
    cout << "Iterations per size: " << iterations << endl;
    cout << "Running benchmarks...\n" << endl;
    
    // Run benchmarks for each size
    for (const auto& size : testSizes) {
        cout << "Testing size " << size.first << "x" << size.second << "... " << flush;
        
        PerformanceResult result = benchmarkCanny(size.first, size.second, iterations);
        results.push_back(result);
        
        cout << "Done! (" << fixed << setprecision(3) << result.avg_time_ms << " ms)" << endl;
    }
    
    // Print results
    printResults(results);
    
    // Save to CSV
    saveResultsToCSV(results, "canny_performance.csv");
    
    cout << "Run 'python3 plot_performance.py' to generate performance plots." << endl;
    
    return 0;
}
