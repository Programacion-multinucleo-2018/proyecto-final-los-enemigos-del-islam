//
//  RGB.hpp
//  Fractal
//
//  Created by Sebastián Galguera on 12/15/17.
//  Copyright © 2017 Sebastián Galguera. All rights reserved.
//

// COMPILER INSTRUCTIONS
#ifndef RGB_hpp
#define RGB_hpp

// SYSTEM LIBRARIES
#include <stdio.h>

// RGB CLASS
class RGB{
public:
    // RGB COMPONENTS
    double m_r;
    double m_g;
    double m_b;
    
    // CONSTRUCTOR
    RGB(double r, double g, double b): m_r(r), m_g(g), m_b(b){};
    
    // OPERATOR OVERLOAD TO SUBSTRACT RGB BY COMPONENTS
    friend RGB operator-(const RGB& first, const RGB& second){
        return RGB(first.m_r - second.m_r, first.m_g - second.m_g, first.m_b - second.m_b);
    };
};

#endif /* RGB_hpp */
