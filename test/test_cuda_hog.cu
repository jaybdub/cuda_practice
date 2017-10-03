#include <stdio.h>
#include "src/cuda_hog.h"

void print_image(const image_float32_t & image, size_t z) 
{
  for (int i = 0; i < image.height; i++)
  {
    for (int j = 0; j < image.width; j++)
    {
      printf("%02f ", image.data[image_float32_index(image, i, j, z)]); 
    }
    printf("\n");
  }
}

inline void assert_true(bool condition, const char * message)
{
  if (!condition)
  {
    fprintf(stderr, "%s\n", message);
  }
}

inline void assert_close(float a, float b, const char * message)
{
  const float epsilon = 1e-5;
  bool close = (a - b) < epsilon && (a - b) > (-epsilon);
  assert_true(close, message);
}

void assert_image_close(const image_float32_t & a,
    const image_float32_t & b, const char * message)
{
for (int i = 0; i < a.height; i++) {
    for (int j = 0; j < a.width; j++) {
      for (int k = 0; k < a.depth; k++) {
        float av = a.data[image_float32_index(a, i, j, k)];
        float bv = b.data[image_float32_index(b, i, j, k)];
        assert_close(av, bv, message);
      }
    }
  }
}

void test__image_float32_count()
{
  image_float32_t image = {3, 2, 2};
  assert_true(image.height == 3, "image_float32_count: Invalid height");
  assert_true(image.width == 2, "image_float32_count: Invalid width");
  assert_true(image.depth == 2, "image_float32_count: Invalid depth");
  assert_true(image_float32_count(image) == 12,
      "image_float32_count: Invalid count");
}

void test__image_float32_size()
{
  image_float32_t image = {3, 2, 2};
  assert_true(image_float32_size(image) == 4 * 12, 
      "image_float32_size: Invalid size.");
}

void test__image_float32_index()
{
  image_float32_t image = {3, 2, 2};
  assert_true(image_float32_index(image, 0, 0, 0) == 0, 
      "image_float32_size: Invalid index.");
  assert_true(image_float32_index(image, 0, 0, 1) == 1, 
      "image_float32_size: Invalid index.");
  assert_true(image_float32_index(image, 0, 1, 0) == 2, 
      "image_float32_size: Invalid index.");
  assert_true(image_float32_index(image, 0, 1, 1) == 3, 
      "image_float32_size: Invalid index.");
  assert_true(image_float32_index(image, 1, 0, 0) == 4, 
      "image_float32_size: Invalid index.");
  assert_true(image_float32_index(image, 2, 0, 0) == 8, 
      "image_float32_size: Invalid index.");
  assert_true(image_float32_index(image, 2, 1, 1) == 11, 
      "image_float32_size: Invalid index.");
}

void test__hog_gradient()
{
  // Image
  image_float32_t image = {5, 5, 1};
  float image_data[] = { 
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 3, 1, 1, 1,
    1, 1, 2, 2, 1,
    1, 1, 1, 1, 1
  };
  image.data = image_data;

  // True gradient
  image_float32_t true_grad = {5, 5, 2};
  float true_grad_data[] = {
    0, 0,  0, 0,  0, 0,  0, 0,  0, 0,
    0, 0,  2, 0,  0, 0,  0, 0,  0, 0,
    0, 0,  0, 0,  1,-2,  1, 0,  0, 0,
    0, 0, -2, 1,  0, 1,  0,-1,  0, 0,
    0, 0,  0, 0,  0, 0,  0, 0,  0, 0,
  };
  true_grad.data = true_grad_data;

  // Compute gradient
  image_float32_t grad = {5, 5, 2};
  image_float32_malloc(grad);
  hog_gradient(image, grad);

  // Verify equal
  assert_image_close(grad, true_grad, "hog_gradient: Gradients to not match");

  // Compute gradient (cuda version)
  image_float32_t cuda_grad = {5, 5, 2};
  image_float32_malloc(cuda_grad);
  hog_gradient_cuda(image, cuda_grad);

  // Verify equal
  assert_image_close(cuda_grad, true_grad, 
      "hog_gradient_cuda: Gradients do not match");
}

int main() {
  test__image_float32_count();
  test__image_float32_size();
  test__image_float32_index();
  test__hog_gradient();
  return 0;
}
