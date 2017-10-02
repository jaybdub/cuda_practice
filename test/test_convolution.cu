#include <iostream>
#include "src/convolution.h"

using namespace std;

void Print(float * x, int x_len) {
  for (int i = 0; i < x_len; i++)
    cout << x[i] << endl;
}

void PrintImage(Image & image) {
  for (int i = 0; i < image.height; i++) {
    for (int j = 0; j < image.width; j++) {
      cout << *image.At(i, j) << "\t";
    }
    cout << endl;
  }
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

void TestCorrelate2D() {
  Image A(5, 5), W(3, 3), B(5, 5);
  A.data = (float*) malloc(A.Size());
  B.data = (float*) malloc(A.Size());
  W.data = (float*) malloc(A.Size());
  for (int i = 0; i < A.height; i++) {
    for (int j = 0; j < A.width; j++) {
      *A.At(i, j) = i * A.width + j; 
    }
  }
  for (int i = 0; i < W.height; i++) {
    for (int j = 0; j < W.width; j++) {
      *W.At(i, j) = 1;
    }
  }
  PrintImage(A);
  PrintImage(W);
  Image * d_A, * d_W, * d_B; // device pointers

  //free memory
  free(A.data);
  free(B.data);
  free(W.data);
  cudaFree(d_A.data);
  cudaFree(d_B.data);
  cudaFree(d_W.data);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_W);
}

int main() {
  TestCorrelate1D();
  TestCorrelate2D();
  return 0;
}
