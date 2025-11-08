#include <chrono>
#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>

using namespace std;
using namespace chrono;

struct PerformanceResult {
    int width;
    int height;
    int pixels;
    double avg_time_ms;
    double min_time_ms;
    double max_time_ms;
    double throughput_fps;
};

double calculateAverage(const vector<double>& times) {
    double sum = 0;
    for (double t : times) sum += t;
    return sum / times.size();
}


void saveResultsToCSV(const vector<PerformanceResult>& results, const string& filename) {
    ofstream file(filename);
    
    // Write header
    file << "width,height,pixels,avg_time_ms,min_time_ms,max_time_ms,std_dev_ms,throughput_fps\n";
    
    // Write data
    for (const auto& result : results) {
        file << result.width << ","
             << result.height << ","
             << result.pixels << ","
             << fixed << setprecision(4)
             << result.avg_time_ms << ","
             << result.min_time_ms << ","
             << result.max_time_ms << ","
             << setprecision(2)
             << result.throughput_fps << "\n";
    }
    
    file.close();
    cout << "Results saved to: " << filename << endl;
}

void printResults(const vector<PerformanceResult>& results) {
    cout << "\n=== Canny Edge Detection Performance Results ===\n" << endl;
    cout << setw(12) << "Size" 
         << setw(15) << "Pixels"
         << setw(15) << "Avg (ms)"
         << setw(15) << "Min (ms)"
         << setw(15) << "Max (ms)"
         << setw(15) << "FPS" << endl;
    cout << string(102, '-') << endl;
    
    for (const auto& result : results) {
        cout << setw(5) << result.width << "x" << setw(5) << result.height
             << setw(15) << result.pixels
             << setw(15) << fixed << setprecision(3) << result.avg_time_ms
             << setw(15) << result.min_time_ms
             << setw(15) << result.max_time_ms
             << setw(15) << setprecision(2) << result.throughput_fps
             << endl;
    }
    cout << endl;
}

