#include <iostream>
#include <vector>
#include "src/cuda_sort.h"

using namespace std;

int main() {
  int nums[5] = {5, 3, 2, 1, 4};
  int sorted[5];
  cuda_merge_sort(nums, sorted, 5);
  Print(sorted, 5);

  return 0;
}
