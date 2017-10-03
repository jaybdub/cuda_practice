#include "src/tensor.h"

size_t tensor_dtype_size(const tensor_dtype_t dtype) {
  return tensor_dtype_sizes[dtype];
}

tensor_t tensor_create(const size_t * dims, const size_t ndim, 
    tensor_dtype_t dtype) {
  
  tensor_t tensor;
  tensor.ndim = ndim;
  tensor.dtype = dtype;

  // set dimensions
  tensor.dims = (size_t*) malloc(ndim * sizeof(size_t));
  for (int i = 0; i < tensor.ndim; i++)
    tensor.dims[i] = dims[i];

  // allocate data
  tensor.size = tensor_size(tensor);
  tensor.data = (void*) malloc(tensor.size);

  // compute strides
  tensor.strides = (size_t*) malloc(ndim * sizeof(size_t));
  tensor.strides[0] = 1;
  for (int i = 1; i < tensor.ndim; i++) {
    tensor.strides[i] = tensor.dims[i - 1] * tensor.strides[i - 1];
  }
  return tensor;
}

void tensor_free(tensor_t * tensor) {
  free(tensor->data);
  free(tensor->dims);
  free(tensor->strides);
}

size_t tensor_size(const tensor_t & tensor) {
  size_t count = 1;
  for (int i = 0; i < tensor.ndim; i++)
    count *= tensor.dims[i];
  return tensor_dtype_size(tensor.dtype) * count;
}

void tensor_print(const tensor_t & tensor) {
  printf("Data Type: %s\n", tensor_dtype_str(tensor.dtype));
  printf("Num Dimensions: %d\n", tensor.ndim);
  printf("Dimensions: ");
  for (int i = 0; i < tensor.ndim; i++) 
    printf("%d ", tensor.dims[i]);
  printf("\n");
}

char * tensor_dtype_str(tensor_dtype_t dtype) {
  switch (dtype) {
  case UINT8:
    return "UINT8";
  case FLOAT32:
    return "FLOAT32";
  }
  return "UNKNOWN";
}
