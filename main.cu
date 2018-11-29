//
//  main.c
//  Fractal
//
//  Created by Sebastián Galguera on 12/13/17.
//  Copyright © 2017 Sebastián Galguera. All rights reserved.
//

// SYSTEM LIBRARIES
#include <stdio.h>
#include <iostream>
#include <cctype>
#include <cuda_runtime.h>

// CUDA CUSTOM LIBRARY
#include "common.h"

// CUSTOM FILES
#include "Fractal.hpp"
#include "Helper.hpp"
#include "RGB.hpp"
#include "Complex.hpp"

// CALCULATE ITERATION
__global__ void calculate_iterations_GPU(int *histogram, int *fractal, double xCenter, double yCenter, int height, int width, double scale){
  unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

  if(ix < width && iy < height){
    double xFractal = ((double)ix - width/2.0f) * scale + xCenter;
    double yFractal = ((double)iy - height/2.0f) * scale + yCenter;
    Complex z;
    Complex c(xFractal, yFractal);
    int iterations = 0;
    while(iterations < 5000){
      z = addComplex(multiplyComplex(z, z), c);
      if (complexAbs(z) > 2)
        break;
      iterations++;
    }

    if ((iy * width + ix) < (height * width)) {
      fractal[iy * width + ix] = iterations;
    }
    
    if (iterations < 5000) {
      histogram[iterations]++;
    }
  }
}

// GET THE CUDA PROPS
int get_cuda_device_props(){
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  return prop.multiProcessorCount;
}

void runParallelKernel(int * histogram, int * fractal, double xCenter, double yCenter, int height, int width, double scale){
  // // SET NUMBER OF BYTES
  size_t nBytes = 5000 * sizeof(int);
  size_t fBytes = width * height * sizeof(int);

  // SET UP DEVICE
  int dev = 0;
  cudaDeviceProp deviceProp;
  SAFE_CALL(cudaGetDeviceProperties(&deviceProp, dev), "Error device prop");
  printf("Using Device %d: %s\n", dev, deviceProp.name);

  // MALLOC DEVICE MEMORY
  int *d_histogram;
  int *d_fractal;


  // ALLOCATE DEVICE MEMORY FOR HISTOGRAM AND FRACTAL ARRAYS
  SAFE_CALL(cudaMalloc<int>(&d_histogram, nBytes), "CUDA Malloc Failed");
  SAFE_CALL(cudaMalloc<int>(&d_fractal, fBytes), "CUDA Malloc Failed");

  // INITIALIZE AT ZERO
  SAFE_CALL(cudaMemset(d_histogram, 0, nBytes), "CUDA Malloc Failed");

  // COPY DATA FROM HOST TO DEVICE
  SAFE_CALL(cudaMemcpy(d_histogram, histogram, nBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed histogram");
  SAFE_CALL(cudaMemcpy(d_fractal, fractal, fBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed fractal");

  // EXECUTE THE KERNEL
  int blocks = get_cuda_device_props();

  dim3 block(64, 4);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  calculate_iterations_GPU<<<grid, block>>>(d_histogram, d_fractal, xCenter, yCenter, height, width, scale);

  SAFE_CALL(cudaDeviceSynchronize(), "Error executing kernel");

  // COPY DATA FROM DEVICE TO HOST
  // for(int i = 0; i < 5000; i++){
  //   printf("%d ", histogram[i]);
  // }
  SAFE_CALL(cudaMemcpy(histogram, d_histogram, nBytes, cudaMemcpyDeviceToHost), "CUDA Memcpy Device To Host Failed d_histogram");
  SAFE_CALL(cudaMemcpy(fractal, d_fractal, fBytes, cudaMemcpyDeviceToHost), "CUDA Memcpy Device To HOST Failed d_fractal");

  // Free device global memory
  SAFE_CALL(cudaFree(d_histogram), "Error freeing memory");
  SAFE_CALL(cudaFree(d_fractal), "Error freeing memory");

}

int main(int argc, const char * argv[]) {
    // PRINT THE WELCOME MESSAGE
    Console::printWelcome();
    // START SELECTING THE SIZE OF THE FRACTAL
    Console::printMessage("Select the size for your fractal");
    int width = Console::getValidPrompt<int>("Width", 1000, "The limit are 1000 px");
    int height = Console::getValidPrompt<int>("Height", 1000, "The limit are 1000 px");

    // INITIALIZE THE FRACTAL
    Fractal f(width, height);
    f.addRange(0, RGB(0,0,0));

    int option = 1;
    int type = 0;
    // WHILE THE OPTION IS NOT 0
    while(option != 0){

        // PRINT MAIN MENU AND PROMPT USER
        Console::printMessage("What would you like to do next?");
        Console::printMenu(MAIN);
        option = Console::getValidPrompt<int>("Option", 2, "The limit is not right");

        // OPTION COMMUTER
        switch(option){
            case 1:{ // FRACTAL CONFIGURATION
                if(f.isConfigured()){

                    // CASE IT'S CONFIGURED ALREADY
                    Console::printMessage("You have a fully configured fractal with 5 color ranges, do you want to overwrite it?");
                    Console::printMenu(CONFIRMATION);
                    option = Console::getValidPrompt<int>("Option", 2, "The limit is not right");

                    // CLEAR CONFIGURATION
                    if(option == 1){
                        f.clearConfiguration();
                        f.setConfiguration();
                    }else{
                        break;
                    }
                }
                // CASE IT'S NOT CONFIGURED
                Console::printMessage("Configure your fractal");
                Console::printMenu(FRACTAL);
                option = Console::getValidPrompt<int>("Option", 3, "The limit is not right");

                if(option == 1){ // SELECT FRACTAL TYPE
                    Console::printMenu(FRACTALTYPE);
                    option = Console::getValidPrompt<int>("Option", 3, "The limit is not right");
                    if(option == 1 || option == 2){
                        type = option;
                    }

                    f.setConfiguration();
                }else if(option == 2){ // SELECT ZOOM
                    Console::printMessage("Insert your Zoom");
                    Zoom zoom = f.promptZoom();
                    f.addZoom(zoom);
                }else if(option == 3){ // SELECT COLOR TYPE

                    // PRINT MENU AND PROMPT USER
                    Console::printMenu(COLORTYPE);
                    option = Console::getValidPrompt<int>("Option", 3, "The limit is not right");

                    if(option == 1){ // HEXADECIMAL INPUT AND LIMIT

                        // PRINT AND PROMPT
                        Console::printMessage("Insert your Hexadecimal color and limit");
                        std::string hex = Console::getValidPrompt<std::string>("Hexadecimal", 7, "The string must be 7 characters or shorter");
                        double limit = Console::getValidPrompt<double>("Limit", 5, "Value must not be bigger than 5");

                        // ADD COLOR RANGE
                        f.addRange(limit, f.hexToRGB(hex));

                    }else if(option == 2){ // RGB INPUT AND LIMIT

                        // PRINT AND PROMPT
                        Console::printMessage("Insert your RGB color and limit");
                        RGB color = f.promptRGB();
                        double limit = Console::getValidPrompt<double>("Limit", 5, "Value must not be bigger than 5");

                        // ADD COLOR RANGE
                        f.addRange(limit, color);
                    }
                    // UPDATE CONFIG
                    f.setConfiguration();
                }else{

                    // END PROGRAM
                    exit(0);
                }
                break;
            }
            case 2:{ // SAVE FRACTAL AND PRINT IT
                // IN CASE THE FRACTAL IS CONFIGURED
                if((f.isConfigured() || (f.getRanges().size() > 1 && f.getColors().size() > 1 && f.getZoomList().getZooms().size() > 1)) && type){
                    // SAVE THE FRACTAL
                    std::string name = Console::getValidPrompt<std::string>("File Name", 15, "The file name is too long, max 10 characters");
                    if(type == 1){
                        Console::printMessage("Sequential Algorithm Running");
                        f.run(name);
                    }else{
                        Console::printMessage("Parallel Algorithm Running");

                        int * histogram = f.m_histogram.get();
                        int * fractal = f.m_fractal.get();
                        runParallelKernel(histogram, fractal, f.m_zoomList.m_xCenter, f.m_zoomList.m_yCenter, f.m_height, f.m_width, f.m_zoomList.m_scale);

                        // CALCULATE TOTAL ITERATIONS
                        Time::calculateTimeCPU<Fractal, &Fractal::calculateTotalIterations>(&f, "calculateTotalIterations");

                        // CALCULATE RANGE TOTALS
                        Time::calculateTimeCPU<Fractal, &Fractal::calculateRangeTotals>(&f, "calculateRangeTotals");

                        // DRAW THE FRACTAL
                        Time::calculateTimeCPU<Fractal, &Fractal::drawFractal>(&f, "drawFractal");

                        // WRITE TO FILE
                        f.writeBitmap(name);
                    }

                }else{ // THE FRACTAL IS NOT CONFIGURED
                    Console::printError("First you have to configure your fractal");
                    if(!(f.getRanges().size() > 1) || !(f.getColors().size() > 1)){
                        std::cout << "  - missing colors" << std::endl;;
                    }
                    if(!(f.getZoomList().getZooms().size() > 1)){
                        std::cout << "  - missing zooms" << std::endl;
                    }
                    if(!type){
                        std::cout << "  - missing type" << std::endl;
                    }
                }
                break;
            }
            case 0:{
                // EXIT THE APPLICATION
                exit(1);
            }
            default:{
                Console::printError("Selected option is not valid");
            }

        }

    }

    return 0;
}
