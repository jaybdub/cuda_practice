#ifndef CUDA_HOG_H
#define CUDA_HOG_H

#define CUDA_CALLABLE __host__ __device__
typedef struct {
  size_t pixels_per_cell = 8;
  size_t cells_per_block = 3;  
  size_t num_bins = 9;
  bool symmetric = 1;
} hog_config_t;

typedef struct {
  size_t height;
  size_t width;
  size_t depth;
  float *data;
  float *data_device;
} image_float32_t;

CUDA_CALLABLE inline size_t image_float32_count(const image_float32_t & image)
{
  return image.height * image.width * image.depth;
}

CUDA_CALLABLE inline size_t image_float32_size(const image_float32_t & image)
{
  return sizeof(float) * image_float32_count(image);
}

void image_float32_malloc(image_float32_t & image) 
{
  image.data = (float*) malloc(image_float32_size(image));
}

void image_float32_malloc_device(image_float32_t & image) 
{
  cudaMalloc(&image.data_device, image_float32_size(image));
}

void image_float32_update_host(image_float32_t & image)
{
  cudaMemcpy(image.data, image.data_device, image_float32_size(image),
      cudaMemcpyDeviceToHost);
}

void image_float32_update_device(image_float32_t & image)
{
  cudaMemcpy(image.data_device, image.data, image_float32_size(image),
      cudaMemcpyHostToDevice);
}

CUDA_CALLABLE inline size_t image_float32_index(const image_float32_t & image, 
    const size_t y, const size_t x, const size_t z)
{
  return y * image.width * image.depth + x * image.depth + z;
}

/** 
 * Normalize oriented gradient histogram using neighboring cells.
 */
void hog_normalize() 
{
}

/**
 * Computes oriented gradient histogram in cells (unnormalized).  Returns result
 * in array bins of dimension (width / dcell)x(height / dcell)x(nbins).
 */
void hog_cells() 
{
}

/**
 * Compute gradient along each dimension by convolving with [-1, 0, 1] and 
 * [-1, 0, 1]^T.  Returns result in grad array of dim (width)x(height)x2.
 */
void hog_gradient(const image_float32_t & image, image_float32_t & grad) 
{
  for (int i = 1; i < image.height - 1; i++) 
  {
    for (int j = 1; j < image.width - 1; j++)
    {
      float *g1 = &grad.data[image_float32_index(grad, i, j, 0)];
      float *g2 = &grad.data[image_float32_index(grad, i, j, 1)];
      *g1 = image.data[image_float32_index(image, i + 1, j, 0)]
        - image.data[image_float32_index(image, i - 1, j, 0)];
      *g2 = image.data[image_float32_index(image, i, j + 1, 0)]
        - image.data[image_float32_index(image, i, j - 1, 0)];
    }
  }
}

__global__ void hog_gradient_cuda_kernel(image_float32_t image, 
    image_float32_t grad) 
{
  const size_t i = threadIdx.x;
  const size_t j = threadIdx.y;

  // check boundary condition
  if (i < 1 || i > image.height - 2 || j < 1 || j > image.width - 2)
  {
    grad.data_device[image_float32_index(grad, i, j, 0)] = 0;
    grad.data_device[image_float32_index(grad, i, j, 1)] = 0;
    return;
  }

  // compute gradient
  float *g1 = &grad.data_device[image_float32_index(grad, i, j, 0)];
  float *g2 = &grad.data_device[image_float32_index(grad, i, j, 1)];
  *g1 = image.data_device[image_float32_index(image, i + 1, j, 0)]
    - image.data_device[image_float32_index(image, i - 1, j, 0)];
  *g2 = image.data_device[image_float32_index(image, i, j + 1, 0)]
    - image.data_device[image_float32_index(image, i, j - 1, 0)];
}

void hog_gradient_cuda(image_float32_t & image, image_float32_t & grad)
{
  // allocate and copy image
  image_float32_malloc_device(image);
  image_float32_update_device(image);

  // allocate gradient
  image_float32_malloc_device(grad);

  // run kernel
  dim3 threadsPerBlock(image.height, image.width);
  hog_gradient_cuda_kernel<<<1, threadsPerBlock>>>(image, grad);

  // copy gradient
  image_float32_update_host(grad);

  // release gpu memory
  cudaFree(grad.data_device);
  cudaFree(image.data_device);
}

/**
 * Computes histogram of oriented gradients.
 */
void hog() 
{
}

#endif
