#include <iostream>
#include <cuda.h>
#include <opencv2/opencv.hpp>
#include "src/argument_parser.h"
#include <string>
#include <sstream>

#define MMIN(a, b) a < b ? a : b
#define MMAX(a, b) a > b ? a : b

using namespace std;


__global__ void BlurKernel(uchar * dst, uchar * src, size_t width, size_t height, size_t channels, int ksize)
{
  const int sy = width * channels;
  const int sx = channels;
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = threadIdx.z;

  if (x > width)
    return;
  if (y > height)
    return;
  if (z > channels)
    return;

  // output pixel
  const int xk_min = MMAX(0, x - ksize / 2);
  const int xk_max = MMIN(width, xk_min + ksize);
  const int yk_min = MMAX(0, y - ksize / 2);
  const int yk_max = MMIN(height, yk_min + ksize);

  int sum = 0;
  for (int xk = xk_min; xk < xk_max; xk++)
  {
    for (int yk = yk_min; yk < yk_max; yk++) 
    {
      sum += src[yk * sy + xk * sx + z];
    }
  }

  dst[y * sy + x * sx + z] = sum / (ksize * ksize);
}

void BlurGPU(cv::Mat & dst, cv::Mat & src, int ksize) 
{
  uchar * d_src, * d_dst;
  const size_t width = dst.cols;
  const size_t height = dst.rows;
  const size_t channels = 3;

  const size_t dsize = width * height * channels;

  cudaMalloc(&d_src, dsize);
  cudaMalloc(&d_dst, dsize);

  cudaMemcpy(d_src, src.data, dsize, cudaMemcpyHostToDevice);

  const size_t bDim = 8;
  dim3 grid(1 + width / bDim, 1 + height / bDim);
  dim3 block(bDim, bDim, channels);

  BlurKernel<<<grid, block>>>(d_dst, d_src, width, height, channels, ksize);

  cudaMemcpy(dst.data, d_dst, dsize, cudaMemcpyDeviceToHost);
  cudaFree(d_src);
  cudaFree(d_dst);
}

void Blur(uchar * dst, uchar * src, size_t width, size_t height, size_t channels, int ksize) 
{
  const int sy = width * channels;
  const int sx = channels;

  // output pixel
  for (int y = 0; y < height; y++)
  {
    for (int x = 0; x < width; x++)
    {
      for (int z = 0; z < channels; z++) {
        const int xk_min = MMAX(0, x - ksize / 2);
        const int xk_max = MMIN(width, xk_min + ksize);
        const int yk_min = MMAX(0, y - ksize / 2);
        const int yk_max = MMIN(height, yk_min + ksize);

        int sum = 0;
        for (int xk = xk_min; xk < xk_max; xk++)
        {
          for (int yk = yk_min; yk < yk_max; yk++) 
          {
            sum += src[yk * sy + xk * sx + z];
          }
        }

        dst[y * sy + x * sx + z] = MIN(255, MAX(0, sum / (ksize * ksize)));
       }
    }
  }
}


int main(int argc, char * argv[]) 
{
  ArgumentSet argset;
  argset.AddNamedArgument("ksize").Shorthand('k').Default("3");
  argset.Parse(argc, argv);

  cv::VideoCapture cap(0);
  cv::Mat img;
  cv::Mat blurred;

  stringstream ss;
  ss << argset.GetArgument("ksize")->GetValue();
  int ksize;
  ss >> ksize;
  cout << "Kernel size: " << ksize << endl;

  while (true) 
  {
    if (!cap.read(img))
      break;
   
    blurred.create(img.rows, img.cols, img.type());
    BlurGPU(blurred, img, 30);
    // Blur(blurred.data, img.data, img.cols, img.rows, 3, 10);

    cv::imshow("Image", img);
    cv::imshow("Blurred", blurred);

    if (cv::waitKey(1) == 'q')
      break;
  }

  cap.release();
  cv::destroyAllWindows();

  return 0;
}
