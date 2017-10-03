#include <stdio.h>
#include "src/tensor.h"

void Print1D(const tensor_t & tensor) {
  for (int i = 0; i < tensor.dims[0]; i++) {
    float * ptr = (float*) tensor.data;
    //printf("%f ", *ptr);
  }
}

int main() {
  const size_t A_dims[] = { 1 };
  tensor_t A = tensor_create(A_dims, 1, FLOAT32);
  tensor_print(A);
  return 0;
}
