#ifndef CUDA_TENSOR_H
#define CUDA_TENSOR_H

#include <stdio.h>

typedef enum {
  UINT8,
  FLOAT32
} tensor_dtype_t;

const size_t tensor_dtype_sizes[] = { 1, 4 };

typedef struct {
  size_t ndim;
  tensor_dtype_t dtype;
  size_t size;
  void * data;
  size_t * dims;
  size_t * strides;
} tensor_t;

/**
 * Get the size (in bytes) of a tensor data type.
 */
size_t tensor_dtype_size(const tensor_dtype_t dtype);

/**
 * Create a tensor of the specified dimensions and type, allocating memory
 * for data.
 */
tensor_t tensor_create(const size_t * dims, const size_t ndim,
    tensor_dtype_t dtype);

/**
 * Deallocate memory used for tensor.
 */
void tensor_free(tensor_t * tensor);

/**
 * Compute size in bytes of tensor.
 */
size_t tensor_size(const tensor_t & tensor);

/**
 * Print tensor details.
 */
void tensor_print(const tensor_t & tensor);

char * tensor_dtype_str(tensor_dtype_t dtype);

#endif  // CUDA_TENSOR_H
