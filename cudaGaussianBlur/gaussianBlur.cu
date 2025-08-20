#include <cuda_runtime.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <vector>
using namespace std;
#define PI 3.14159265
const int kernelSize = 5;
const int trials = 10;

__global__ void gaussianBlur(const unsigned char* image, unsigned char* output, int width, int height, const float* kernel, int kernelSize, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }
    int radius = kernelSize / 2;
    float sumR = 0.0f, sumG = 0.0f, sumB = 0.0f;
    for (int ky = -radius; ky <= radius; ky++) {
        for (int kx = -radius; kx <= radius; kx++) {
            //use clamping to make sure that we don't go past the border of the image
            int nx = min(max(x + kx, 0), width - 1);
            int ny = min(max(y + ky, 0), height - 1);
            int nIdx = (ny * width + nx) * channels;

            float weight = kernel[(ky + radius) * kernelSize + (kx + radius)];

            sumR += image[nIdx] * weight;
            sumG += image[nIdx + 1] * weight;
            sumB += image[nIdx + 2] * weight;
        }
    }
    
    int idx = (y * width + x) * channels;
    output[idx] = static_cast<unsigned char>(sumR);
    output[idx + 1] = static_cast<unsigned char>(sumG);
    output[idx + 2] = static_cast<unsigned char>(sumB);
}

//calculate gaussian kernel
void calculateKernel(int kernelSize, float sigma, vector<float>& kernel) {
    int radius = kernelSize / 2;
    kernel.resize(kernelSize * kernelSize);
    float sum = 0.0f;
    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {
            float exponent = -((x * x) + (y * y))/(2*sigma*sigma);
            float n = 1.0f / (2.0f * PI * sigma * sigma);
            float value = n * exp(exponent);
            kernel[(y + radius) * kernelSize + (x + radius)] = value;
            sum += value;
        }
    }
    for (int i = 0; i < kernelSize * kernelSize; i++) {
        kernel[i] /= sum;
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: ./gaussianBlur <input_image> <output_image>" << std::endl;
        return -1;
    }
    cv::Mat input = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (input.empty()) {
        std::cerr << "Error loading image\n";
        return -1;
    }
    std::string outputImagePath = argv[2];
    if (outputImagePath.empty()) {
        std::cerr << "Error: output image path is empty." << std::endl;
        return -1;
    }
    //start timer
    //int64 startTime = cv::getTickCount();
    // cpu gaussian blur conversion
    cv::Mat cpu_output;
    cv::GaussianBlur(input, cpu_output, cv::Size(kernelSize, kernelSize), 10.0f);
    //stop timer
    //int64 end = cv::getTickCount();
    //double elapsed_ms = (end - startTime) * 1000.0 / cv::getTickFrequency();
    //std::cout << "CPU Gaussian Blur took " << elapsed_ms << " ms" << std::endl;
    cv::imwrite("cpu_output.jpg", cpu_output);

    int width = input.cols;
    int height = input.rows;
    int channels = input.channels();
    cv::Mat output(height, width, CV_8UC3);

    //build gaussian kernel
    vector<float> h_kernel;
    calculateKernel(kernelSize, 10.0f, h_kernel);

    //allocate gpu mem
    unsigned char *d_image, *d_output;
    float* d_kernel;
    size_t rgbSize = width * height * channels * sizeof(unsigned char);
    size_t kb = kernelSize * kernelSize * sizeof(float);
    cudaMalloc(&d_kernel, kb);
    cudaMalloc(&d_image, rgbSize);
    cudaMalloc(&d_output, rgbSize);

    //copy to gpu
    cudaMemcpy(d_image, input.data, rgbSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel.data(), kb, cudaMemcpyHostToDevice);
    dim3 block (16, 16);
    dim3 grid ((width + block.x -1 ) / block.x, (height + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // cudaEventRecord(start);
    // gaussianBlur<<<grid, block>>>(d_image, d_output, width, height, d_kernel, kernelSize, channels);
    // cudaEventRecord(stop);

    //cudaEventSynchronize(stop);
    // float ms = 0;
    // cudaEventElapsedTime(&ms, start, stop);
    // std::cout << "GPU kernel time: " << ms << " ms" << std::endl;

    double cpu_total_time = 0;
    printf("Benchmarking CPU... \n");
    printf("=======================\n");
    for (int t = 0; t < trials; t++) {
        int64 start = cv::getTickCount();
        cv::GaussianBlur(input, cpu_output, cv::Size(kernelSize, kernelSize), 10.0f);
        int64 end = cv::getTickCount();
        double elapsed = (end - start) * 1000.0 / cv::getTickFrequency();
        if (t > 0) cpu_total_time += elapsed; // skip first trial
    }
    double cpu_avg = cpu_total_time / (trials - 1);

    float gpu_total_time = 0;
    printf("Benchmarking GPU... \n");
    printf("=======================\n");
    for (int t = 0; t < trials; t++) {
        cudaEventRecord(start);
        gaussianBlur<<<grid, block>>>(d_image, d_output, width, height, d_kernel, kernelSize, channels);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);

        if (t > 0) gpu_total_time += ms; // skip first trial
    }
    float gpu_avg = gpu_total_time / (trials - 1);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }
    cudaDeviceSynchronize();

    cudaMemcpy(output.data, d_output, rgbSize, cudaMemcpyDeviceToHost);
    cv::imwrite(argv[2], output);
    cudaFree(d_kernel);
    cudaFree(d_image);
    cudaFree(d_output);

    std::cout << "Saved blurred image to: " << argv[2] << std::endl;
    //std::cout << "GPU kernel improved performance by " << elapsed_ms/ms << "x" << std::endl;

    std::cout << "Average CPU time: " << cpu_avg << " ms\n";
    std::cout << "Average GPU time: " << gpu_avg << " ms\n";
    std::cout << "Speedup: " << cpu_avg / gpu_avg << "x\n";

    return 0;
}