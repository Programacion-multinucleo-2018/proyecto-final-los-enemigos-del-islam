//
//  Mandelbrot.hpp
//  Fractal
//
//  Created by Sebastián Galguera on 12/14/17.
//  Copyright © 2017 Sebastián Galguera. All rights reserved.
//

// COMPILER INSTRUCTIONS
#ifndef Mandelbrot_hpp
#define Mandelbrot_hpp

// SYSTEM LIBRARIES
#include <stdio.h>
#include <complex>
#include <time.h>
#include <stdlib.h>
#include <cmath>

// MANDELBROT CLASS
class Mandelbrot{
public:
    
    // MAXIMUM ITERATIONS
    static const int MAX_ITERATIONS = 5000;

    // CONSTRUCTOR AND VIRTUAL DESTRUCTOR
    Mandelbrot();
    virtual ~Mandelbrot();

    // FUNCTION TO GET ITERATIONS
    static int getIterations(double x, double y);
};

// FUNCTION GET ITERATIONS
int Mandelbrot::getIterations(double x, double y){
    // USE COMPLEX NUMBER FOR REAL AND IMAGINARY COMPONENTS
    std::complex<double> z = 0;
    std::complex<double> c(x,y);
    int iterations = 0;
    while(iterations < MAX_ITERATIONS){
        z = z*z + c; // COMPUTE THE NEW Z
        if(std::abs(z) > 2){ // CHECK FOR ABSOLUTE VALUE
            break;
        }
        iterations++;
    }
    return iterations;
    
};



#endif /* Mandelbrot_hpp */
