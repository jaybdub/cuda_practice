#include <iostream>
#include <fstream>
#include <NvInfer.h>
#include <sstream>

#define INPUT_BLOB_NAME "data"
#define OUTPUT_BLOB_NAME "prob"

using namespace std;
using namespace nvinfer1;


void LoadModel(void ** modelMem, size_t & modelSize, char * filepath)
{
  ifstream model(filepath, ifstream::ate);
  modelSize = model.tellg();
  *modelMem = malloc(modelSize); 
  model.seekg(0, ios::beg);
  model.read((char*)*modelMem, modelSize);
}


class Logger : public ILogger
{
  void log(Severity severity, const char * msg) override
  {
    if (severity != Severity::kINFO)
      cout << msg << endl;
  }
} gLogger;


size_t GetBindingSize(ICudaEngine * engine, int index)
{ 
  static int dtype_sizes[3];
  dtype_sizes[(int)DataType::kFLOAT] = 4;
  dtype_sizes[(int)DataType::kHALF] = 2;
  dtype_sizes[(int)DataType::kINT8] = 1;

  Dims dims = engine->getBindingDimensions(index);
  size_t bufsize = 1;
  for (int i = 0; i < dims.nbDims; i++)
    bufsize *= dims.d[i];

  bufsize *= dtype_sizes[(int)engine->getBindingDataType(index)];
  return bufsize;
}


int main(int argc, char * argv[])
{
  if (argc != 2) 
  { 
    cout << "Usage: classifier <execution_model>" << endl;
    return 0;
  }

  void * modelMem;
  size_t modelSize;

  cout << "Loading model at " << argv[1];
  LoadModel(&modelMem, modelSize, argv[1]);
  cout << " (done)" << endl;

  cout << "Deserializing cuda engine...";
  IRuntime * runtime = createInferRuntime(gLogger);
  ICudaEngine * engine = runtime->deserializeCudaEngine(modelMem, modelSize,
      nullptr);
  cout << " (done)" << endl;

  IExecutionContext * context = engine->createExecutionContext();

  int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
  int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
  size_t inputSize = GetBindingSize(engine, inputIndex);
  size_t outputSize = GetBindingSize(engine, outputIndex);

  if (inputIndex >= 0)
    cout << "Found blob '" << INPUT_BLOB_NAME << "' at index " << inputIndex <<
      " of size " << GetBindingSize(engine, inputIndex) << endl;
  if (outputIndex >= 0)
    cout << "Found blob '" << OUTPUT_BLOB_NAME << "' at index " << outputIndex <<
      " of size " << GetBindingSize(engine, outputIndex) << endl;

  void * inputDataDevice, * outputDataDevice;

  cudaMalloc(&inputDataDevice, inputSize);
  cudaMalloc(&outputDataDevice, outputSize);

  void * inferenceBuffers[] = { inputDataDevice, outputDataDevice };
  context->execute(1, inferenceBuffers);

  cudaFree(inputDataDevice);
  cudaFree(outputDataDevice);
  free(modelMem);

  return 0;
}
