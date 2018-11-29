//
//  ZoomList.hpp
//  Fractal
//
//  Created by Sebastián Galguera on 12/14/17.
//  Copyright © 2017 Sebastián Galguera. All rights reserved.
//

// COMPILER INSTRUCTIONS
#ifndef ZoomList_hpp
#define ZoomList_hpp

// SYSTEM LIBRARIES
#include <stdio.h>
#include <vector>
#include <utility>

// PRAGMA FOR PERFORMANCE
#pragma pack(2)
struct Zoom{
    int x{0}; // X COORDINATE
    int y{0}; // Y COORDINATE
    double scale{0.0}; // SCALE
    // CONSTRUCTOR
    Zoom(int x, int y, double scale): x(x), y(y), scale(scale){};
    
};

// CLASS THAT CONTAINS THE CALCULATED Y AND X CENTER
class ZoomList{
public:
    // CALCULATE OVERALL X AND Y CENTER
    double m_xCenter{0};
    double m_yCenter{0};
    double m_scale{1.0};
    
    // SIZE
    int m_width{0};
    int m_height{0};
    
    // LIST OF ZOOMS
    std::vector<Zoom> zooms;

    // ZOOM LIST CONSTRUCTOR
    ZoomList(int width, int height): m_width(width), m_height(height){};
    // PUSH BACK
    void add(const Zoom& zoom);
    // POP BACK
    void pop();
    // CREATE PAIR OF X AND Y COORDINATES
    std::pair<double, double> doZoom(int x, int y);
    std::vector<Zoom> getZooms();

};

// ADD ZOOM TO ZOOMLIST
void ZoomList::add(const Zoom& zoom){
    
    // ADD THE ZOOM AND CALCULATE OVERALL CENTERS
    zooms.push_back(zoom);
    m_xCenter += (zoom.x - m_width/2)*m_scale;
    m_yCenter += (zoom.y - m_height/2)*m_scale;
    m_scale *= zoom.scale;
}

// POP THE ZOOM LIST
void ZoomList::pop(){
    zooms.pop_back();
}

// PAIR CONSTRUCTION
std::pair<double, double> ZoomList::doZoom(int x, int y){
    double xFractal = (x - m_width/2)*m_scale + m_xCenter;
    double yFractal = (y - m_height/2)*m_scale + m_yCenter;
    
    return std::pair<double,double>(xFractal,yFractal);
}

std::vector<Zoom> ZoomList::getZooms(){
    return zooms;
}


#endif /* ZoomList_hpp */
