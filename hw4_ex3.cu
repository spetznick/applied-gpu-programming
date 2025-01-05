#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cusparse_v2.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>

#include <numeric>

#define gpuCheck(stmt)                                       \
    do {                                                     \
        cudaError_t err = stmt;                              \
        if (err != cudaSuccess) {                            \
            printf("ERROR. Failed to run stmt %s\n", #stmt); \
            break;                                           \
        }                                                    \
    } while (0)

// Macro to check the cuBLAS status
#define cublasCheck(stmt)                                           \
    do {                                                            \
        cublasStatus_t err = stmt;                                  \
        if (err != CUBLAS_STATUS_SUCCESS) {                         \
            printf("ERROR. Failed to run cuBLAS stmt %s\n", #stmt); \
            break;                                                  \
        }                                                           \
    } while (0)

// Macro to check the cuSPARSE status
#define cusparseCheck(stmt)                                           \
    do {                                                              \
        cusparseStatus_t err = stmt;                                  \
        if (err != CUSPARSE_STATUS_SUCCESS) {                         \
            printf("ERROR. Failed to run cuSPARSE stmt %s\n", #stmt); \
            break;                                                    \
        }                                                             \
    } while (0)

struct timeval t_start, t_end;
void cputimer_start() { gettimeofday(&t_start, nullptr); }
void cputimer_stop(const char *info) {
    gettimeofday(&t_end, nullptr);
    double time = (1000000.0 * (t_end.tv_sec - t_start.tv_sec) + t_end.tv_usec -
                   t_start.tv_usec);
    printf("Timing - %s. \t\tElasped %.0f microseconds \n", info, time);
}

double cputimer_stop() {
    gettimeofday(&t_end, nullptr);
    double time = (1000000.0 * (t_end.tv_sec - t_start.tv_sec) + t_end.tv_usec -
                   t_start.tv_usec);
    return time;
}

// Initialize the sparse matrix needed for the heat time step
void matrixInit(double *A, int *ArowPtr, int *AcolIndx, int dimX,
                double alpha) {
    // Stencil from the finete difference discretization of the equation
    double stencil[] = {1, -2, 1};
    // Variable holding the position to insert a new element
    size_t ptr = 0;
    // Insert a row of zeros at the beginning of the matrix
    ArowPtr[1] = ptr;
    // Fill the non zero entries of the matrix
    for (int i = 1; i < (dimX - 1); ++i) {
        // Insert the elements: A[i][i-1], A[i][i], A[i][i+1]
        for (int k = 0; k < 3; ++k) {
            // Set the value for A[i][i+k-1]
            A[ptr] = stencil[k];
            // Set the column index for A[i][i+k-1]
            AcolIndx[ptr++] = i + k - 1;
        }
        // Set the number of newly added elements
        ArowPtr[i + 1] = ptr;
    }
    // Insert a row of zeros at the end of the matrix
    ArowPtr[dimX] = ptr;
}

float calculateFLOPS(int nvz, double time_ms) {
    // nvz: Number of non-zero elements in the sparse matrix
    // time_ms: Kernel execution time in milliseconds
    if (time_ms < 0.) {
        printf("ERROR: time negative");
    }
    double ops = 2.0f * nvz;  // Each 1*A*temp + 0*tmp operation counts as 1 add
                              // and 1 multiply per element
    return ops / (time_ms / 1000.f);
}

int main(int argc, char **argv) {
    int device = 0;           // Device to be used
    int dimX;                 // Dimension of the metal rod
    int nsteps;               // Number of time steps to perform
    double alpha = 0.4;       // Diffusion coefficient
    double *temp;             // Array to store the final time step
    double *A;                // Sparse matrix A values in the CSR format
    int *ARowPtr;             // Sparse matrix A row pointers in the CSR format
    int *AColIndx;            // Sparse matrix A col values in the CSR format
    int nzv;                  // Number of non zero values in the sparse matrix
    double *tmp;              // Temporal array of dimX for computations
    size_t bufferSize = 0;    // Buffer size needed by some routines
    void *buffer = nullptr;   // Buffer used by some routines in the libraries
    int concurrentAccessQ;    // Check if concurrent access flag is set
    double zero = 0;          // Zero constant
    double one = 1;           // One constant
    double norm;              // Variable for norm values
    double error;             // Variable for storing the relative error
    double tempLeft = 200.;   // Left heat source applied to the rod
    double tempRight = 300.;  // Right heat source applied to the rod
    cublasHandle_t cublasHandle;      // cuBLAS handle
    cusparseHandle_t cusparseHandle;  // cuSPARSE handle
    int prefetch;                     // Flag to prefetch memory
    struct timeval t_start_run;

    cusparseSpMatDescr_t Adescriptor;  // Mat descriptor needed by cuSPARSE
    cusparseDnVecDescr_t vecX;         // Input vector descriptor
    cusparseDnVecDescr_t vecY;

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("No CUDA-capable devices found!\n");
        return -1;
    }
    if (device >= deviceCount || device < 0) {
        printf("Invalid device ID: %d\n", device);
        return -1;
    }
    cudaSetDevice(device);

    // Read the arguments from the command line
    dimX = atoi(argv[1]);
    nsteps = atoi(argv[2]);
    prefetch = atoi(argv[3]);

    // Print input arguments
    printf("The X dimension of the grid is %d \n", dimX);
    printf("The number of time steps to perform is %d \n", nsteps);

    // Get if the cudaDevAttrConcurrentManagedAccess flag is set
    gpuCheck(cudaDeviceGetAttribute(
        &concurrentAccessQ, cudaDevAttrConcurrentManagedAccess, device));

    // Calculate the number of non zero values in the sparse matrix. This number
    // is known from the structure of the sparse matrix
    nzv = 3 * dimX - 6;

    //@@ Insert the code to allocate the temp, tmp and the sparse matrix
    //@@ arrays using Unified Memory
    cputimer_start();
    t_start_run = t_start;
    cudaMallocManaged(&temp, sizeof(double) * dimX);
    cudaMallocManaged(&tmp, sizeof(double) * dimX);
    cudaMallocManaged(&A, sizeof(double) * nzv);
    cudaMallocManaged(&ARowPtr, sizeof(int) * (dimX + 1));
    cudaMallocManaged(&AColIndx, sizeof(int) * nzv);
    cputimer_stop("Allocating device memory");

    // Check if concurrentAccessQ is non zero in order to prefetch memory
    if (concurrentAccessQ && prefetch) {
        cputimer_start();
        //@@ Insert code to prefetch in Unified Memory asynchronously to CPU
        cudaMemPrefetchAsync(temp, sizeof(double) * dimX, cudaCpuDeviceId);
        cudaMemPrefetchAsync(tmp, sizeof(double) * dimX, cudaCpuDeviceId);
        cudaMemPrefetchAsync(A, sizeof(double) * nzv, cudaCpuDeviceId);
        cudaMemPrefetchAsync(ARowPtr, sizeof(int) * (dimX + 1),
                             cudaCpuDeviceId);
        cudaMemPrefetchAsync(AColIndx, sizeof(int) * nzv, cudaCpuDeviceId);
        cputimer_stop("Prefetching GPU memory to the host");
    } else {
        printf("No prefetching\n");
    }

    // Initialize the sparse matrix
    cputimer_start();
    matrixInit(A, ARowPtr, AColIndx, dimX, alpha);
    cputimer_stop("Initializing the sparse matrix on the host");

    // Initiliaze the boundary conditions for the heat equation
    cputimer_start();
    memset(temp, 0, sizeof(double) * dimX);
    temp[0] = tempLeft;
    temp[dimX - 1] = tempRight;
    cputimer_stop("Initializing memory on the host");

    if (concurrentAccessQ && prefetch) {
        cputimer_start();
        //@@ Insert code to prefetch in Unified Memory asynchronously to the GPU
        cudaMemPrefetchAsync(temp, sizeof(double) * dimX, device);
        cudaMemPrefetchAsync(tmp, sizeof(double) * dimX, device);
        cudaMemPrefetchAsync(A, sizeof(double) * nzv, device);
        cudaMemPrefetchAsync(ARowPtr, sizeof(int) * (dimX + 1), device);
        cudaMemPrefetchAsync(AColIndx, sizeof(int) * nzv, device);
        cputimer_stop("Prefetching GPU memory to the device");
    } else {
        // printf("No prefetching\n");
    }

    //@@ Insert code to create the cuBLAS handle
    cublasCreate(&cublasHandle);
    //@@ Insert code to create the cuSPARSE handle
    cusparseCreate(&cusparseHandle);
    //@@ Insert code to set the cuBLAS pointer mode to
    // CUSPARSE_POINTER_MODE_HOST
    // cublasSetPointerMode(cublasHandle, CUSPARSE_POINTER_MODE_HOST); // is
    // there a typo in the instructions?
    cusparseSetPointerMode(cusparseHandle, CUSPARSE_POINTER_MODE_HOST);
    //@@ Insert code to call cusparse api to create the mat descriptor used by
    // cuSPARSE
    cusparseCreateCsr(&Adescriptor, dimX, dimX, nzv, ARowPtr, AColIndx, A,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    cusparseCreateDnVec(&vecX, dimX, temp, CUDA_R_64F);
    cusparseCreateDnVec(&vecY, dimX, tmp, CUDA_R_64F);
    //@@ Insert code to call cusparse api to get the buffer size needed by the
    // sparse matrix per
    //@@ vector (SMPV) CSR routine of cuSPARSE
    cusparseSpMV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &one, Adescriptor, vecX, &zero, vecY, CUDA_R_64F,
                            CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);

    //@@ Insert code to allocate the buffer needed by cuSPARSE
    cudaMalloc(&buffer, bufferSize);

    // Perform the time step iterations
    std::vector<double> flops;
    flops.reserve(nsteps);
    std::vector<double> timesPerIteration;
    timesPerIteration.reserve(nsteps);
    double time_ms = 1.;
    for (int it = 0; it < nsteps; ++it) {
        cputimer_start();
        //@@ Insert code to call cusparse api to compute the SMPV (sparse matrix
        // multiplication) for
        //@@ the CSR matrix using cuSPARSE. This calculation corresponds to:
        //@@ tmp = 1 * A * temp + 0 * tmp
        cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
                     Adescriptor, vecX, &zero, vecY, CUDA_R_64F,
                     CUSPARSE_SPMV_ALG_DEFAULT, buffer);
        time_ms = cputimer_stop();
        timesPerIteration.push_back(time_ms);
        // compute flops
        flops.push_back(calculateFLOPS(nzv, time_ms));
        //@@ Insert code to call cublas api to compute the axpy routine using
        // cuBLAS.
        //@@ This calculation corresponds to: temp = alpha * tmp + temp
        cublasDaxpy(cublasHandle, dimX, &alpha, tmp, 1, temp, 1);

        //@@ Insert code to call cublas api to compute the norm of the vector
        // using cuBLAS
        //@@ This calculation corresponds to: ||temp||
        cublasDnrm2(cublasHandle, dimX, temp, 1, &norm);

        // If the norm of A*temp is smaller than 10^-4 exit the loop
        if (norm < 1e-4) break;
    }

    // Calculate the exact solution using thrust
    thrust::device_ptr<double> thrustPtr(tmp);
    thrust::sequence(thrustPtr, thrustPtr + dimX, tempLeft,
                     (tempRight - tempLeft) / (dimX - 1));

    // Calculate the relative approximation error:
    one = -1.;
    //@@ Insert the code to call cublas api to compute the difference between
    // the exact solution
    //@@ and the approximation
    //@@ This calculation corresponds to: temp = -tmp + temp
    cublasDaxpy(cublasHandle, dimX, &one, tmp, 1, temp, 1);

    //@@ Insert the code to call cublas api to compute the norm of the absolute
    // error
    //@@ This calculation corresponds to: || tmp ||
    cublasDnrm2(cublasHandle, dimX, temp, 1, &norm);

    error = norm;
    //@@ Insert the code to call cublas api to compute the norm of temp
    //@@ This calculation corresponds to: || tmp ||
    cublasDnrm2(cublasHandle, dimX, tmp, 1, &norm);

    // Calculate average FLOPS for given size
    double avgLoopTime = std::accumulate(timesPerIteration.begin(),
                                         timesPerIteration.end(), 0.) /
                         static_cast<double>(timesPerIteration.size());
    double avgFlops = std::accumulate(flops.begin(), flops.end(), 0.) /
                      static_cast<double>(flops.size());
    printf("The estimated FLOPS are %f, avg. SPMV is %f\n", avgFlops,
           avgLoopTime);

    // Calculate the relative error
    error = error / norm;
    printf("The relative error of the approximation is %f on nsteps %d\n",
           error, nsteps);

    //@@ Insert the code to destroy the mat descriptor
    cusparseDestroySpMat(Adescriptor);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);

    //@@ Insert the code to destroy the cuSPARSE handle
    cusparseDestroy(cusparseHandle);

    //@@ Insert the code to destroy the cuBLAS handle
    cublasDestroy(cublasHandle);

    //@@ Insert the code for deallocating memory
    cudaFree(temp);
    cudaFree(tmp);
    cudaFree(A);
    cudaFree(ARowPtr);
    cudaFree(AColIndx);
    cudaFree(buffer);

    // stop Timing
    t_start = t_start_run;
    double runtime = cputimer_stop();
    printf("Total runtime: %f ms\n\n", runtime);

    return 0;
}
