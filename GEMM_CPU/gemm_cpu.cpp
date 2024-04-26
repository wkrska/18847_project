#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <thread>

using namespace std;
using namespace cv;

int n_sizes = 9;
int n_types = 1;
int runs = 5;
const int sizes[] = {2, 4, 8, 16, 32, 64, 128, 256, 512};

int main() {

  for (int i = 0; i < n_types; i++) {
    for (int j = n_sizes-1; j >= 0; j++) {
      // Matrix size (adjust as needed)
      const int N = sizes[j];

      Mat A(N, N, CV_32F);
      Mat B(N, N, CV_32F);
      Mat C(N, N, CV_32F);

      // Initialize matrices with random values (optional)
      randu(A, 0.0f, 1.0f);
      randu(B, 0.0f, 1.0f);

      // Get starting time using high-resolution clock
      auto start_time = chrono::high_resolution_clock::now();

      // Perform GEMM using OpenCV's BLAS integration
      gemm(A, B, 1.0, C, 1.0, C);

      // Get ending time
      auto end_time = chrono::high_resolution_clock::now();

      // Calculate elapsed time in nanoseconds
      double elapsed_ns = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count() ;

      // Print execution time
      cout << elapsed_ns << "ns for " << sizes[j] << "x" << sizes[j] << " " << ((i==0)?"CV_32FC1":((i==1)?"CV_16FC1":"CV_8UC1")) << " matrix" << endl;
    }
  }
  return 0;
}
