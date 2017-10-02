#include <iostream>
#include "src/convolution.h"

using namespace std;

void Print(float * x, int x_len) {
  for (int i = 0; i < x_len; i++)
    cout << x[i] << endl;
}

void TestCorrelate1D() {
  const size_t N = 20;
  float a[N], b[N];
  for (int i = 0; i < N; i++)
    a[i] = i;
  const size_t N_w = 5;
  float w[N_w] = { -2, -1, 0, 1, 2 };
  float * d_a, * d_b, * d_w;   
  const size_t size = N * sizeof(float);
  cudaMalloc(&d_a, size);
  cudaMalloc(&d_b, size);
  cudaMalloc(&d_w, N_w * sizeof(float));
  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice); 
  cudaMemcpy(d_w, w, N_w * sizeof(float), cudaMemcpyHostToDevice);
  Correlate1D<<<N, 1>>>(d_a, N, d_w, N_w, d_b);
  cudaMemcpy(b, d_b, size, cudaMemcpyDeviceToHost);
  Print(b, N);  
  cudaFree(d_a);
  cudaFree(d_b);
}

int main() {
  TestCorrelate1D();
  return 0;
}
