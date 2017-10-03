#ifndef CUDA_SORT_H
#define CUDA_SORT_H

#include <vector>

using namespace std;

void Print(int * nums, size_t n) {
  for (int i = 0; i < n; i++)
    cout << nums[i] << " ";
  cout << endl;
}

/**
 * n: total length of array
 * m: maximum length of sub-arrays to merge 
 */
__global__ void merge(int * src, int * dst, const int n, const int m) {
  // compute division points
  const int p = 2 * threadIdx.x * m;
  const int q = 2 * threadIdx.x * m + m;
  int r = 2 * threadIdx.x * m + 2 * m;

  // handle end of array
  if (r > n)
    r = n;
   
  // merge greatest from each sub-array
  int a_idx = p;
  int b_idx = q;
  int dst_idx = p;
  while ((a_idx < q) && (b_idx < r)) {
    if (src[a_idx] < src[b_idx])
      dst[dst_idx++] = src[a_idx++];
    else
      dst[dst_idx++] = src[b_idx++];
  }

  // read out remaining
  while (a_idx < q)
    dst[dst_idx++] = src[a_idx++];
  while (b_idx < r)
    dst[dst_idx++] = src[b_idx++];
}

void cuda_merge_sort(int * a, int * b, size_t n) {

  // allocate and copy device memory
  int *d_a, *d_b;
  cudaMalloc(&d_a, n * sizeof(int));
  cudaMalloc(&d_b, n * sizeof(int));
  cudaMemcpy(d_a, a, n * sizeof(int), cudaMemcpyHostToDevice);

  // parallel merge sort on device
  size_t m = 1; 
  bool dir = 1;
  while (m < n) {
    const size_t nthreads = (n + 2 * m - 1) / (2 * m);
    if (dir) {
      merge<<<1, nthreads>>>(d_a, d_b, n, m);
    }
    else {
      merge<<<1, nthreads>>>(d_b, d_a, n, m);
    }
    dir = !dir;
    m *= 2;
  }

  // copy data back
  if (dir)
    cudaMemcpy(b, d_a, n * sizeof(int), cudaMemcpyDeviceToHost);
  else
    cudaMemcpy(b, d_b, n * sizeof(int), cudaMemcpyDeviceToHost);
}

#endif
