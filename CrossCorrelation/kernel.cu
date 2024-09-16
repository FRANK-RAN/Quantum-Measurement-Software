
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <Windows.h>
#include <stdlib.h>
#include <time.h>
#include <cublas_v2.h>
#include <iostream>


#define PIPE_NAME "\\\\.\\pipe\\DataPipe"




void checkCuda(cudaError_t result, const char* msg) {
	if (result != cudaSuccess) {
		std::cerr << "CUDA Error: " << msg << " (" << cudaGetErrorString(result) << ")\n";
		exit(EXIT_FAILURE);
	}
}

void checkCublas(cublasStatus_t result, const char* msg) {
	if (result != CUBLAS_STATUS_SUCCESS) {
		std::cerr << "cuBLAS Error: " << msg << "\n";
		exit(EXIT_FAILURE);
	}
}

bool readFileIntoBuffer(const char* filename, void* buffer, DWORD bufferSize, DWORD segmentIndex) {
	HANDLE hFile = CreateFile(
		filename,            // name of the file
		GENERIC_READ,        // open for reading
		0,                   // do not share
		NULL,                // default security
		OPEN_EXISTING,       // open existing file only
		FILE_ATTRIBUTE_NORMAL, // normal file
		NULL);               // no attribute template

	if (hFile == INVALID_HANDLE_VALUE) {
		std::cerr << "Unable to open file " << filename << "\n";
		return false;
	}

	// Calculate the offset to seek to the correct segment
	LARGE_INTEGER offset;
	offset.QuadPart = segmentIndex * bufferSize;
	if (SetFilePointerEx(hFile, offset, NULL, FILE_BEGIN) == 0) {
		std::cerr << "Failed to set file pointer for segment index " << segmentIndex << "\n";
		CloseHandle(hFile);
		return false;
	}

	DWORD bytesRead;
	BOOL readResult = ReadFile(
		hFile,
		buffer,
		bufferSize,
		&bytesRead,
		NULL);

	CloseHandle(hFile);

	if (!readResult || bytesRead != bufferSize) {
		std::cerr << "Failed to read file " << filename << " completely for segment " << segmentIndex << "\n";
		return false;
	}

	return true;
}

HANDLE createAndConnectPipe(const char* pipeName, DWORD bufferSize) {
	HANDLE hPipe = CreateNamedPipe(
		pipeName,                 // Pipe name passed as argument
		PIPE_ACCESS_DUPLEX,        // Read/Write access
		PIPE_TYPE_BYTE |           // Byte-type pipe
		PIPE_READMODE_BYTE |       // Byte-read mode
		PIPE_WAIT,                 // Blocking mode
		1,                         // Max number of instances
		bufferSize,                // Output buffer size
		bufferSize,                // Input buffer size
		0,                         // Default timeout
		NULL);                     // Default security attributes

	if (hPipe == INVALID_HANDLE_VALUE) {
		std::cerr << "Failed to create named pipe.\n";
		return NULL;
	}

	std::cout << "Waiting for client connection...\n";
	BOOL connected = ConnectNamedPipe(hPipe, NULL) ? TRUE : (GetLastError() == ERROR_PIPE_CONNECTED);

	if (!connected) {
		std::cerr << "Failed to connect to the client.\n";
		CloseHandle(hPipe);
		return NULL;
	}

	return hPipe;
}


bool CheckForRequest(HANDLE hPipe)
{
	DWORD bytesAvailable = 0;
	if (PeekNamedPipe(hPipe, NULL, 0, NULL, &bytesAvailable, NULL) && bytesAvailable > 0)
	{
		return true; // Data is available to read
	}
	return false; // No data available
}


int handleClientRequests(HANDLE hPipe, short* data, int segmentIndex, DWORD bytesToSend)
{
	if (!CheckForRequest(hPipe))
	{
		std::cerr << "No request.\n";
		return 0;  // No data to process
	}

	// Read the client's request
	short request;
	DWORD bytesRead;
	BOOL success = ReadFile(hPipe, &request, sizeof(request), &bytesRead, NULL);
	if (!success || bytesRead != sizeof(request))
	{
		std::cerr << "Failed to read request from client.\n";
		return 1;  // Error reading the request
	}

	std::cout << "Received request from client, sending data...\n";

	// Send a specific number of bytes of a specific segment of `data` to the client
	DWORD bytesWritten;

	// Calculate the starting position for the segment to send
	short* segmentStart = data + (segmentIndex * (bytesToSend / sizeof(short))); // Calculate the starting point

	success = WriteFile(hPipe, segmentStart, bytesToSend, &bytesWritten, NULL);
	if (!success || bytesWritten != bytesToSend)
	{
		std::cerr << "Failed to send data to client.\n";
		return 1;  // Error sending the data
	}

	std::cout << "Sent data to client.\n";
	return 2;  // Successfully processed the request and sent data
}








// Demodulation at 8 correlation matrix with shared memory, light version
__global__ void demodulationCorrelationAt8Shared(short* a, __int64 numElements, double* correlationMatrix, const int sharedSegmentSize) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	// Declare shared memory
	__shared__ double sharedSegment[128]; // 512 bytes

	// Only the first 32 threads in the block load data into shared memory
	if (threadIdx.x < sharedSegmentSize) {
		sharedSegment[threadIdx.x] = static_cast<double>(a[blockIdx.x * sharedSegmentSize + threadIdx.x]);
	}

	__syncthreads(); // Ensure all threads have loaded their data into shared memory

	int segmentStart = threadIdx.x / 64 * 32; // Determine the starting index of the segment in shared memory


	int row = index % 64 / 8;
	int col = index % 8;


	double value1 = sharedSegment[segmentStart + row * 2];
	double value2 = sharedSegment[segmentStart + (row + 8) * 2];
	double value3 = sharedSegment[segmentStart + col * 2 + 1];
	double value4 = sharedSegment[segmentStart + (col + 8) * 2 + 1];

	double corrValue = (value1 - value2) * (value3 - value4);

	correlationMatrix[index] = corrValue; // Correlation matrix, one column is a single correlation matrix

}




__global__ void autoCorrelationAt8(short* a, __int64 numElements, double* correlationMatrixA, double* correlationMatrixB, const int sharedSegmentSize) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	//int matrixSize = numElements / 32; // the number of matrices will generate or the number of segments
	//int elementIndex = index % 64; // Each thread works on one element of the 8x8 correlation matrix
	//int segmentIndex = index / 64; // Determines which 32-element segment we're working on

	//int segment
	// Declare shared memory
	__shared__ float sharedSegment[128]; // 512 bytes

	// Only the first 32 threads in the block load data into shared memory
	if (threadIdx.x < sharedSegmentSize) {
		sharedSegment[threadIdx.x] = static_cast<float>(a[blockIdx.x * sharedSegmentSize + threadIdx.x]);
	}

	__syncthreads(); // Ensure all threads have loaded their data into shared memory

	int segmentStart = threadIdx.x / 64 * 32; // Determine the starting index of the segment in shared memory


	int row = index % 64 / 8;
	int col = index % 8;


	float A_value_1 = sharedSegment[segmentStart + row * 2];
	float A_value_2 = sharedSegment[segmentStart + (row + 8) * 2];
	float A_value_3 = sharedSegment[segmentStart + col * 2];
	float A_value_4 = sharedSegment[segmentStart + (col + 8) * 2];

	float B_value_1 = sharedSegment[segmentStart + row * 2 + 1];
	float B_value_2 = sharedSegment[segmentStart + (row + 8) * 2 + 1];
	float B_value_3 = sharedSegment[segmentStart + col * 2 + 1];
	float B_value_4 = sharedSegment[segmentStart + (col + 8) * 2 + 1];



	double corrValueA = (A_value_1 - A_value_2) * (A_value_3 - A_value_4);
	double corrValueB = (B_value_1 - B_value_2) * (B_value_3 - B_value_4);

	//correlationMatrix[elementIndex * matrixSize + segmentIndex] = corrValue;				//Store the correlation matrix in row-major order
	// Store the correlation matrix in column-major order
	correlationMatrixA[index] = corrValueA; // Correlation matrix, one column is a single correlation matrix
	correlationMatrixB[index] = corrValueB; // Correlation matrix, one column is a single correlation matrix
}



// CUDA kernel to initialize the array
__global__ void initializeArrayKernel(double* array, int size, double value) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) {
		array[idx] = value;
	}
}


// Function to initialize the array with 1/N
extern "C" void initializeArrayWithCuda(double* dev_array, int size, double value) {
	int blockSize = 256;
	int numBlocks = (size + blockSize - 1) / blockSize;
	initializeArrayKernel << <numBlocks, blockSize >> > (dev_array, size, value);
	cudaDeviceSynchronize();
}

// CUDA kernel to divide the matrix by N
__global__ void averageMatrixKernel(double* averageMatrix, int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < 64) {
		averageMatrix[idx] /= N;
	}
}


__global__ void loadMemoryfromSystem(short* a, float* d_host_a, __int64 numElements) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	//int stride = blockDim.x * gridDim.x;
	float loadedData = 0;


	d_host_a[index] = static_cast<float>(a[index]);
	//d_host_a[index] = index;
	//short loadedData = a[index];


}


// benchmark function, memory load from system
cudaError_t benchmark(void* d_a, float* d_host_a, const __int64 numElements) {
	cudaError_t cudaStatus = cudaSuccess;

	// record the time
	LARGE_INTEGER nFreq;
	LARGE_INTEGER nBeginTime;
	LARGE_INTEGER nEndTime;
	float time;

	int blockDim = 256;
	int gridDim = numElements / blockDim;
	// Create CUDA events
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	loadMemoryfromSystem << <gridDim, blockDim >> > ((short*)d_a, d_host_a, numElements);
	// Wait for the GPU to finish
	checkCuda(cudaDeviceSynchronize(), "Kernel execution failed");

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	// 
	printf("load memory kernel time: %f ms\n", time);



	checkCuda(cudaDeviceSynchronize(), "Kernel execution failed");


	return cudaStatus;
}


// Helper function for using CUDA.
cudaError_t GPU_Equation_PlusOne(void* a,
	__int64 size, int blocks, int threads,
	int u32LoopCount, double* h_odata,
	int N, cublasHandle_t handle, double* d_correlationMatrix, double* d_averageMatrix, double* d_scaling_factors,
	FILE* binFile, FILE* AnalysisFile)
{
	cudaError_t cudaStatus = cudaSuccess; // Return status of CUDA functions

	// Kernel launch configuration
	int blockSize = 256;	// Threads per block
	int totalThreads = (size / 32) * 64; // Total number of threads
	int gridSize = (totalThreads + blockSize - 1) / blockSize; // Number of blocks


	// Demodulation at 8 for correlation matrix
	//demodulationCorrelationAt8NoShared << <gridSize, blockSize >> > ((short*)a, size, d_correlationMatrix);

	// Demodulation at 8 for correlation matrix with shared memoryEMORY
	const int sharedSegmentSize = 128;
	demodulationCorrelationAt8Shared << <gridSize, blockSize >> > ((short*)a, size, d_correlationMatrix, sharedSegmentSize);


	// Perform matrix-vector multiplication using cuBLAS
	// 64 x N matrix-vector multiplication
	const int Nrows = 64;
	const int Ncols = N;
	const double alpha = 1.0;
	const double beta = 0.0;

	// d_correlationMatrix is a 64 x N matrix
	// d_scaling_factors is a N x 1 vector
	// d_averageMatrix is a 64 x 1 vector
	cublasStatus_t cublasStatus = cublasDgemv(handle, CUBLAS_OP_N, Nrows, Ncols, &alpha,
		d_correlationMatrix, 64,
		d_scaling_factors, 1,
		&beta, d_averageMatrix, 1);
	checkCublas(cublasStatus, "cuBLAS Dgemv failed");

	averageMatrixKernel << <1, 64 >> > (d_averageMatrix, N);	// Average the reduced matrix

	// Copy the result back to the host
	checkCuda(cudaMemcpy(h_odata, d_averageMatrix, 64 * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy failed");

	// Wait for the GPU to finish
	checkCuda(cudaDeviceSynchronize(), "Kernel execution failed");

	// Write results to Analysis file
	if (AnalysisFile) {
		fprintf(AnalysisFile, "%d\t", u32LoopCount);
		for (int i = 0; i < 64; ++i) {
			fprintf(AnalysisFile, "%.10f\t", h_odata[i]);
		}
		fprintf(AnalysisFile, "\n");
	}

	// Write results to binary file for Matlab use
	if (binFile) {
		fwrite(h_odata, sizeof(double), 64, binFile);
	}

	return cudaStatus;
}



int main() {
	// Set device
	checkCuda(cudaSetDevice(0), "cudaSetDevice failed");

	// Number of elements (modify as needed)
	const __int64 numElements = 32022528; // example size
	//const __int64 numElements = 128; // 
	const int blocks = 256; // number of blocks (modify as needed)
	const int threads = 256; // number of threads per block (modify as needed)
	const int u32LoopCount = 1; // loop count
	const int N = numElements / 32; // number of segments


	// Allocate pinned host memory
	short* h_a;
	float* d_host_a;
	checkCuda(cudaHostAlloc((void**)&h_a, numElements * sizeof(short), cudaHostAllocMapped), "cudaHostAlloc failed for h_a");


	checkCuda(cudaMalloc((void**)&d_host_a, numElements * sizeof(float)), "cudaHostAlloc failed for d_host_a");

	double* h_odata = (double*)malloc(64 * sizeof(double));



	// Allocate device memory
	short* d_a = nullptr;
	checkCuda(cudaHostGetDevicePointer((void**)&d_a, (void*)h_a, 0), "cudaHostGetDevicePointer failed for d_a");




	double* d_correlationMatrix = nullptr;
	double* d_averageMatrix = nullptr;
	double* d_scaling_factors = nullptr;

	checkCuda(cudaMalloc((void**)&d_correlationMatrix, N * 64 * sizeof(double)), "cudaMalloc failed for d_correlationMatrix");
	checkCuda(cudaMalloc((void**)&d_averageMatrix, 64 * sizeof(double)), "cudaMalloc failed for d_averageMatrix");
	checkCuda(cudaMalloc((void**)&d_scaling_factors, N * sizeof(double)), "cudaMalloc failed for d_scaling_factors");


	// Initialize scaling factors with 1/N
	initializeArrayWithCuda(d_scaling_factors, N, 1.0);

	// Create cuBLAS handle
	cublasHandle_t handle;
	checkCublas(cublasCreate(&handle), "cuBLAS initialization failed");

	// Open files for writing results
	FILE* binFile = fopen("cm.bin", "wb");
	FILE* AnalysisFile = fopen("results.txt", "w");
	int numIterations = 2;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	DWORD bufferSize = numElements * 2;  // Specify the buffer size
	// Read data from file into h_a
	const char* filename = "C:\\Users\\jr151\\GageData\\data.dat";

	// create namedpipe and connect to server
	HANDLE hPipe = createAndConnectPipe(PIPE_NAME, 0);  // 0 is default buffer size

	// Call the GPU_Equation_PlusOne function multiple times
	for (int i = 0; i < numIterations; ++i) {
		if (!readFileIntoBuffer(filename, h_a, bufferSize, i)) {
			std::cerr << "Failed to read data from " << filename << " for segment " << i << "\n";
			return 1;
		}
		int result = handleClientRequests(hPipe, h_a, 0, 200);  // 200 is the number of bytes to send, check request from client and send data

		cudaEventRecord(start);
		cudaError_t cudaStatus = GPU_Equation_PlusOne((void*)d_a, numElements, blocks, threads, u32LoopCount + i, h_odata, N, handle, d_correlationMatrix, d_averageMatrix, d_scaling_factors, binFile, AnalysisFile);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "GPU_Equation_PlusOne failed at iteration %d!\n", i);
			return 1;
		}	//cudaEvent_t start, stop;


		cudaEventRecord(start);
		cudaMemcpy((void*)d_a, (void*)h_a, numElements * sizeof(short), cudaMemcpyHostToDevice);

		benchmark((void*)d_a, d_host_a, numElements);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("kernel: %f ms\n", milliseconds);

		

		printf("Iteration %d completed\n", i);
	}

	// Cleanup
	fclose(binFile);
	fclose(AnalysisFile);
	checkCublas(cublasDestroy(handle), "cuBLAS cleanup failed");
	checkCuda(cudaFree(d_correlationMatrix), "cudaFree failed for d_correlationMatrix");
	checkCuda(cudaFree(d_averageMatrix), "cudaFree failed for d_averageMatrix");
	checkCuda(cudaFree(d_scaling_factors), "cudaFree failed for d_scaling_factors");
	checkCuda(cudaFreeHost(h_a), "cudaFreeHost failed for h_a");
	free(h_odata);
	checkCuda(cudaDeviceReset(), "cudaDeviceReset failed");

	fprintf(stdout, "Program completed successfully\n");

	return 0;
}





