#include <iostream>
#include <cuda.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "src/argument_parser.h"
#include <string>
#include <sstream>

#define MMIN(a, b) a < b ? a : b
#define MMAX(a, b) a > b ? a : b

using namespace std;


__global__ void GrayKernel(uchar * dst, uchar * src, size_t width, size_t
    height, size_t channels)

{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x > width)
    return;
  if (y > height)
    return;

  const int sy_src = width * channels;
  const int sx_src = channels;
  const int sy_dst = width;
  int value = 0;
  for (int i = 0; i < channels; i++)
    value += src[y * sy_src + x * sx_src + i];
  dst[y * sy_dst + x] = value / channels;
}

void GrayGPU(cv::Mat & dst, cv::Mat & src)
{
  const int width = dst.cols;
  const int height = dst.rows;
  const int channels = 3;
  uchar * d_dst, * d_src;
  const size_t dst_size = width * height * sizeof(uchar);
  const size_t src_size = channels * dst_size;
  cudaMalloc(&d_dst, dst_size);
  cudaMalloc(&d_src, src_size);
  cudaMemcpy(d_src, src.data, src_size, cudaMemcpyHostToDevice);
  const size_t thread_dim = 32;
  dim3 blocks(1 + width / thread_dim, 1 + height / thread_dim); 
  dim3 threads(thread_dim, thread_dim);
  GrayKernel<<<blocks, threads>>>(d_dst, d_src, width, height, channels);
  cudaMemcpy(dst.data, d_dst, dst_size, cudaMemcpyDeviceToHost);
}

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
  cv::Mat gray;

  stringstream ss;
  ss << argset.GetArgument("ksize")->GetValue();
  int ksize;
  ss >> ksize;
  cout << "Kernel size: " << ksize << endl;
  cv::namedWindow("Image", 1);
  cv::namedWindow("Gray", 1);


  while (true) 
  {
    if (!cap.read(img))
      break;
   
    blurred.create(img.rows, img.cols, img.type());
    gray.create(img.rows, img.cols, CV_8UC1);

    GrayGPU(gray, img);
    BlurGPU(blurred, img, ksize);
    // Blur(blurred.data, img.data, img.cols, img.rows, 3, 10);

    cv::putText(gray, "Grayscale Image", cv::Point(10, 10), cv::FONT_HERSHEY_PLAIN,
        1, cv::Scalar(150));
    cv::imshow("Image", img);
    // cv::imshow("Blurred", blurred);
    cv::imshow("Gray", gray);

    if (cv::waitKey(1) == 'q')
      break;
  }

  cap.release();
  cv::destroyAllWindows();

  return 0;
}
