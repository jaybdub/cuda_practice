#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#define CUDA_CALLABLE __host__ __device__

struct Image {
  Image(int width, int height) : width(width), height(height) {}
  float * data;
  const size_t width;
  const size_t height;
  CUDA_CALLABLE int Size() { return sizeof(float) * width * height; }
  CUDA_CALLABLE int Area() { return width * height; }
  CUDA_CALLABLE float * At(int x, int y) { 
    const int idx = y * width + x;
    if (idx < Area())
      return &data[idx]; 
    else
      return NULL;
  };
};

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

__global__ void Correlate2D(Image X, Image W,
    Image Y) {
  const int Y_x = threadIdx.x;
  const int Y_y = threadIdx.y;
  float * y = Y.At(Y_x, Y_y); 
  *y = 0;
  for (int W_x = 0; W_x < W.width; W_x++) {
    const int X_x = Y_x - W.width / 2 + W_x;
    if (X_x < 0 || X_x >= X.width)
      continue;
    for (int W_y = 0; W_y < W.height; W_y++) {
      const int X_y = Y_y - W.height / 2 + W_y;
      if (X_y < 0 || X_y >= X.height)
        continue;
      *y += (*W.At(W_x, W_y)) * (*X.At(X_x, X_y));
    }
  }
}

#endif  // CONVOLUTION_H
