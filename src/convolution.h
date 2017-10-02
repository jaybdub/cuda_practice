#ifndef CONVOLUTION_H
#define CONVOLUTION_H

__global__ void Correlate1D(float * x, int x_len,
    float * w, int w_len, float * y) {
  int y_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (y_idx >= x_len)
    return;

  y[y_idx] = 0;
  for (int w_idx = 0; w_idx < w_len; w_idx++) {
    int x_idx = y_idx - w_len / 2 + w_idx;
    if (0 <= x_idx && x_idx < x_len)
      y[y_idx] += w[w_idx] * x[x_idx];
  }
}



#endif  // CONVOLUTION_H
