//
//  Fractal.hpp
//  Fractal
//
//  Created by Sebastián Galguera on 12/15/17.
//  Copyright © 2017 Sebastián Galguera. All rights reserved.
//

// COMPILER INSTRUCTIONS
#ifndef Fractal_hpp
#define Fractal_hpp

// SYSTEM LIBRARIES
#include <stdio.h>
#include <string>
#include <cstdint>
#include <math.h>
#include <memory>
#include <vector>
#include <iostream>
#include <assert.h>

// CUSTOM FILES
#include "RGB.hpp"
#include "ZoomList.hpp"
#include "Mandelbrot.hpp"
#include "Bitmap.hpp"
#include "Helper.hpp"

// CONSOLE AND TCHECK HELPERS
typedef Helper::ConsoleHandling Console;
typedef Helper::TypeChecking TCheck;
typedef Helper::TimeHandling Time;

// CLASS TO CREATE FRACTAL
class Fractal{
    
    public:
    // TOTAL VALUES
    int m_total{0};
    int m_width;
    int m_height;
    
    // CONFIGURATION FLAG
    bool configured = 0;
    
    // UNIQUE POINTER TO HISTOGRAM AND FRACTAL
    std::unique_ptr<int []> m_histogram;
    std::unique_ptr<int []> m_fractal;
    
    // BITMAP
    Bitmap m_bitmap;
    // ZOOMLIST
    ZoomList m_zoomList;

    // VECTOR OF RANGES
    std::vector<int> m_ranges;
    // VECTOR OF COLORS
    std::vector<RGB> m_colors;
    // VECTOR OF RANGE TOTALS
    std::vector<int> m_rangeTotals;
    
    // FLAG FIRST RANGE
    bool m_bGotFirstRange{false};

    // CALCULATE ITERATION TOTAL ITERATINS AND RANGE TOTALS
    // PARALLEL
    void calculateIterationCUDA(int * histogram, int * fractal, double xCenter, double yCenter, int height, int width, double scale);
    
    // SEQUENTIAL
    void calculateIteration();
    void calculateTotalIterations();
    void calculateRangeTotals();
    
    // DRAW THE FRACTAL
    // PARALLEL
    void drawFractalCUDA();
    void drawFractal();
    
    // ADD AND POP ZOOM
    void popZoom();
    
    // WRITE BITMAP
    void writeBitmap(std::string name);
    
    // GET RANGE
    int getRange(int iterations) const;
    
    Fractal(int width, int height); // CONSTRUCTOR
    
    virtual ~Fractal(); // VIRTUAL DESTRUCTOR
    void addRange(double rangeEnd, const RGB& rgb); // ADD RANGE
     void addZoom(const Zoom& zoom); // ADD ZOOM
    
    // INITIALIZE
    void initialize();
    // RUN THE FRACTAL
    void run(std::string name);
    
    // CONFIGURATION FLAG
    bool isConfigured();
    void setConfiguration();
    
    // CONVERT HEX TO DEC AND HEX TO RGB
    int hexToDec(std::string hex);
    RGB hexToRGB(std::string hex);
    RGB promptRGB();
    Zoom promptZoom();
    
    // CLEAR CONFIGURATION
    void clearConfiguration();
    
    // GETTERS
    ZoomList getZoomList();
    
    // VECTOR OF RANGES
    std::vector<int> getRanges();
    
    // VECTOR OF COLORS
    std::vector<RGB> getColors();
    
    
};

// FRACTAL CONSTRUCTOR
Fractal::Fractal(int width, int height): m_width(width), m_height(height),
// HISTOGRAM INITIALIZES WITH MAX ITERATIONS INITIALIZED
m_histogram(new int[Mandelbrot::MAX_ITERATIONS]{0}),
// FRACTAL SIZE
m_fractal(new int[m_width*m_height]{0}),
// BITMAP SIZE
m_bitmap(m_width, m_height),
// ZOOMLIST WITH SIZE
m_zoomList(m_width, m_height){
    
    // ADD A ZOOM TO THE ZOOMLIST
    m_zoomList.add(Zoom(m_width/2, m_height/2, 4.0/m_width));
    
}

// DESTRUCTOR
Fractal::~Fractal(){}

// ADD RANGE
void Fractal::addRange(double rangeEnd, const RGB& rgb){
    // PUSH BACK THE RANGE END TIMES THE MAX ITERATIONS
    m_ranges.push_back(rangeEnd*Mandelbrot::MAX_ITERATIONS);
    // PUSH BACK COLOR
    m_colors.push_back(rgb);
    
    // IF YOU HAVE A FIRST RANGE, PUSH A 0 TO RANGE TOTALS
    if(m_bGotFirstRange){ m_rangeTotals.push_back(0); }
    // CHANGE FLAG
    m_bGotFirstRange = true;
}


// CALCULATE RANGE ROTALS
void Fractal::calculateRangeTotals(){
    // RANGE INDEX
    int rangeIndex = 0;
    
    // GET A PIXEL PER HISTOGRAM VALUE
    for(int i = 0; i < Mandelbrot::MAX_ITERATIONS; i++){
        int pixels = m_histogram[i];
        
        // CHECK FOR THE FIRST RANGE WHERE THE ITS VALUE IS BIGGER THAN THE
        // ITERATION
        if(i >= m_ranges[rangeIndex+1]){ rangeIndex++; }
        
        // GET THE RANGE TOTALS
        m_rangeTotals[rangeIndex] += pixels;
    }
    
    // PRINT RANGE TOTALS VALUE
    for(int value: m_rangeTotals)
        std::cout << value << std::endl;
}

// CALCULATE TOTAL ITERATIONS
void Fractal::calculateTotalIterations(){
    for(int i = 0; i < Mandelbrot::MAX_ITERATIONS; i++){
        m_total += m_histogram[i];
    }
}

// ADD A ZOOM WRAPPER
void Fractal::addZoom(const Zoom& zoom){
    m_zoomList.add(zoom);
}


// POP A ZOOM WRAPPER
void Fractal::popZoom(){
    m_zoomList.pop();
}

// WRITE BITMAP WRAPPER
void Fractal::writeBitmap(std::string name){
    m_bitmap.write(name);
}

// GET THE RANGE
int Fractal::getRange(int iterations) const{
    int range = 0;
    
    // RANGE MUST START AT 1, 0 IS RESERVED
    for(int i = 1; i < m_ranges.size(); i++){
        range = i;
        if(m_ranges[i] > iterations)
            break;
    }
    
    // ASSERTION PHASE
    range--;
    assert(range > -1);
    assert(range < m_ranges.size());
    return range;
}

Zoom Fractal::promptZoom(){
    int x = Console::getValidPrompt<int>("x coordinate", m_width, "coordinate cannot be bigger than the file width");
    int y = Console::getValidPrompt<int>("y coordinate", m_height, "coordinate cannot be bigger than the file height");
    double scale = Console::getValidPrompt<double>("scale", 0.13, "scale cannot be bigger than 0.1");
    return Zoom(x, m_height - y, scale);
}

// CALCULATE ITERATION
void Fractal::calculateIteration(){
    // TRAVERSE MATRIX
    for(int y = 0; y < m_height; y++){
        for(int x = 0; x < m_width; x++){
            // GET THE NEW COORDS WITH THE ZOOM LIST
            std::pair<double,double> coords = m_zoomList.doZoom(x, y);
            
            // GET THE ITERATIONS WITH THE COORDINATES OF THE FIRST AND SECOND VALUE
            int iterations = Mandelbrot::getIterations(coords.first, coords.second);
            
            // PUT THE ITERATIONS
            m_fractal[y * m_width + x] = iterations;
            
            // ADD ITERATIONS UNTIL THE MAX ITERATIONS ARE REACHED
            if(iterations != Mandelbrot::MAX_ITERATIONS){
                m_histogram[iterations]++;
            }
            
        }
    }
    
}

// DRAW THE FRACTAL
void Fractal::drawFractal(){
    
    // TRAVERSE THE MATRIX
    for(int y = 0; y < m_height; y++){
        for(int x = 0; x < m_width; x++){
            
            int iterations = m_fractal[y * m_width + x]; // ITERATIONS
            int range = getRange(iterations); // RANGE FOR ITERATIONS
            int rangeTotal = m_rangeTotals[range]; // RANGE TOTAL
            int rangeStart = m_ranges[range]; // RANGE START
            
            RGB& startColor = m_colors[range]; // FIRST COLOR
            RGB& endColor = m_colors[range+1]; // SECOND COLOR
            RGB colorDiff = endColor - startColor; // COLOR DIFFERENCE
            
            std::uint8_t red = 0; // R
            std::uint8_t green = 0; // G
            std::uint8_t blue = 0; // B
            
            // CHECK IF ITERATIONS ARE THE MAX ITERATIONS
            if(iterations != Mandelbrot::MAX_ITERATIONS){
                int totalPixels = 0; // TOTAL PIXELS
                
                for(int i = rangeStart; i <= iterations; i++)
                    totalPixels += m_histogram[i];
                
                // COLORS ARE THE SUM OF EACH CHANNEL + THE DIFFERENCE TIMES THE
                // TOTAL PIXELS AND RANGE TOTAL TO CREATE A GRADIENT
                red = startColor.m_r + colorDiff.m_r*(double)totalPixels/rangeTotal;
                green = startColor.m_g + colorDiff.m_g*(double)totalPixels/rangeTotal;
                blue = startColor.m_b + colorDiff.m_b*(double)totalPixels/rangeTotal;
            }
            // SET THE PIXEL TO THE BITMAP
            m_bitmap.setPixel(x, y, red, green, blue);
        }
    }
    
}



////--------------------------------------------------------------------------------------------------------


////--------------------------------------------------------------------------------------------------------



////--------------------------------------------------------------------------------------------------------



// DRAW THE FRACTAL
void Fractal::drawFractalCUDA(){
    
    // TRAVERSE THE MATRIX
    for(int y = 0; y < m_height; y++){
        for(int x = 0; x < m_width; x++){
            
            int iterations = m_fractal[y * m_width + x]; // ITERATIONS
            int range = getRange(iterations); // RANGE FOR ITERATIONS
            int rangeTotal = m_rangeTotals[range]; // RANGE TOTAL
            int rangeStart = m_ranges[range]; // RANGE START
            
            RGB& startColor = m_colors[range]; // FIRST COLOR
            RGB& endColor = m_colors[range+1]; // SECOND COLOR
            RGB colorDiff = endColor - startColor; // COLOR DIFFERENCE
            
            std::uint8_t red = 0; // R
            std::uint8_t green = 0; // G
            std::uint8_t blue = 0; // B
            
            // CHECK IF ITERATIONS ARE THE MAX ITERATIONS
            if(iterations != 5000){
                int totalPixels = 0; // TOTAL PIXELS
                
                for(int i = rangeStart; i <= iterations; i++)
                    totalPixels += m_histogram[i];
                
                // COLORS ARE THE SUM OF EACH CHANNEL + THE DIFFERENCE TIMES THE
                // TOTAL PIXELS AND RANGE TOTAL TO CREATE A GRADIENT
                red = startColor.m_r + colorDiff.m_r*(double)totalPixels/rangeTotal;
                green = startColor.m_g + colorDiff.m_g*(double)totalPixels/rangeTotal;
                blue = startColor.m_b + colorDiff.m_b*(double)totalPixels/rangeTotal;
            }
            // SET THE PIXEL TO THE BITMAP
            m_bitmap.setPixel(x, y, red, green, blue);
        }
    }
    
}

// RUN FUNCTION
void Fractal::run(std::string name){
    
    // CALCULATE ITERATION
    Time::calculateTimeCPU<Fractal, &Fractal::calculateIteration>(&*this, "calculateIteration");
    
    // CALCULATE TOTAL ITERATIONS
    Time::calculateTimeCPU<Fractal, &Fractal::calculateTotalIterations>(&*this, "calculateTotalIterations");
    
    // CALCULATE RANGE TOTALS
    Time::calculateTimeCPU<Fractal, &Fractal::calculateRangeTotals>(&*this, "calculateRangeTotals");
    
    // DRAW THE FRACTAL
    Time::calculateTimeCPU<Fractal, &Fractal::drawFractal>(&*this, "drawFractal");
    
    // WRITE TO FILE
    writeBitmap(name);
}


////--------------------------------------------------------------------------------------------------------

////--------------------------------------------------------------------------------------------------------

////--------------------------------------------------------------------------------------------------------


// HEXADECIMAL TO DECIMAL
int Fractal::hexToDec(std::string hex) {
    double dec = 0;
    for (int i = 0; i < hex.length(); ++i){
        char b = hex[i];
        if (b >= 48 && b <= 57)
            b -= 48;
        else if (b >= 65 && b <= 70)
            b -= 55;
        dec += b * pow(16, ((hex.length() - i) - 1));
    }
    return (int)dec;
}

// HEXADECIMAL TO RGB
RGB Fractal::hexToRGB(std::string hex){
    if (hex[0] == '#')
        hex = hex.erase(0, 1);
    double r, g, b;
    r = (unsigned char)hexToDec(hex.substr(0, 2));
    g = (unsigned char)hexToDec(hex.substr(2, 2));
    b = (unsigned char)hexToDec(hex.substr(4, 2));
    return RGB(r,g,b);
}

// PROMPT THE RGB
RGB Fractal::promptRGB(){
    double r, g, b;
    r = Console::getValidPrompt<int>("R", 255, "The limit is not right");
    g = Console::getValidPrompt<int>("G", 255, "The limit is not right");
    b = Console::getValidPrompt<int>("B", 255, "The limit is not right");
    return RGB(r,g,b);
}

// CLEAR CONFIGURATION
void Fractal::clearConfiguration(){
    m_colors.clear();
    m_ranges.clear();
    // BASE CASE
    m_colors.push_back(RGB(0,0,0));
    m_ranges.push_back(0);
}

// IS CONFIGURED
bool Fractal::isConfigured(){ return configured; }

// CHECK IF CONFIGURATION MUST CHANGE
void Fractal::setConfiguration(){
    if(m_width && m_height && m_colors.size() > 5 && m_ranges.size() > 5 && m_zoomList.getZooms().size() > 3){
        configured = true;
    }else{
        configured = false;
    }
}

// GET ZOOMLIST
ZoomList Fractal::getZoomList(){
    return m_zoomList;
};

// GET RANGES
std::vector<int> Fractal::getRanges(){
    return m_ranges;
};

// GET COLORS
std::vector<RGB> Fractal::getColors(){
    return m_colors;
};

#endif /* Fractal_hpp */
