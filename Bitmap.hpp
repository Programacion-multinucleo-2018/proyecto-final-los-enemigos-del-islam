//
//  Bitmap.hpp
//  Fractal
//
//  Created by Sebastián Galguera on 12/13/17.
//  Copyright © 2017 Sebastián Galguera. All rights reserved.
//

// COMPILER INSTRUCTIONS
#ifndef Bitmap_hpp
#define Bitmap_hpp

// SYSTEM LIBRARIES
#include <fstream>
#include <stdio.h>
#include <string>
#include <cstdint>

// PRAGMA PACK FOR PERFORMANCE
#pragma pack(2)
// FILE HEADER STRUCT
struct BitmapFileHeader{
    char header[2]{'B','M'}; // FLAGS
    std::int32_t fileSize; // FILESIZE
    std::int32_t reserved{0}; // RESERVED BITS
    std::int32_t dataOffset; // DATA OFFSET
};

// PRAGMA PACK FOR PERFORMANCE
#pragma pack(2)
// INFO HEADER STRUCT
struct BitmapInfoHeader{
    std::int32_t headerSize{40}; // SIZE OF HEADER
    std::int32_t width;
    std::int32_t height;
    std::int16_t planes{1}; // PLANES, DON'T CHANGE IT
    std::int16_t bitsPerPixel{24}; // NO. BITS PER PIXEL
    std::int32_t compression{0}; // NO COMPRESSION
    std::int32_t dataSize{0};
    std::int32_t horizontalResolution{2400}; // RESOLUTION
    std::int32_t verticalResolution{2400}; // RESOLUTION
    std::int32_t colors{0};
    std::int32_t importantColors{0};
};

// BITMAP CLASS
class Bitmap{

    // WIDTH AND HEIGHT
    int m_width{0};
    int m_height{0};

    // UNIQUE POINTER TO UINT8_T TO HANDLE SCOPE AND OWNERSHIP
    std::unique_ptr<std::uint8_t[]> m_pPixels{nullptr};
public:

    // CONSTRUCTOR
    Bitmap(int width, int height);
    // FUNCTION TO SET PIXEL
    void setPixel(int x, int y, std::uint8_t red, std::uint8_t green, std::uint8_t blue);
    // FUNCTION TO WRITE BITMAP
    bool write(std::string filename);
    // DESTRUCTOR
    virtual ~Bitmap();
};

// CONSTRUCTOR
Bitmap::Bitmap(int width, int height): m_width(width), m_height(height), m_pPixels(new std::uint8_t[width*height*3]{}){

}

// FUNCTION TO WRITE BITMAP
bool Bitmap::write(std::string filename){

    // CREATE HEADER FILE AND INFO
    BitmapFileHeader fileHeader;
    BitmapInfoHeader infoHeader;

    // CALCULATE FILE SIZE
    fileHeader.fileSize = sizeof(BitmapFileHeader) + sizeof(BitmapFileHeader) + m_width*m_height*3;

    // CALCULATE DATA OFFSET
    fileHeader.dataOffset = sizeof(BitmapFileHeader) + sizeof(BitmapInfoHeader);

    // CONFIGURE HEADER'S WIDTH AND HEIGHT
    infoHeader.width = m_width;
    infoHeader.height = m_height;

    // OPEN FILE
    std::ofstream file;
    file.open(filename + ".bmp", std::ios::out|std::ios::binary);

    if(!file){ return false; }

    // WRITE THE HEADER WITH CAST TO CHAR *
    file.write((char *)&fileHeader, sizeof(fileHeader));
    file.write((char *)&infoHeader, sizeof(infoHeader));
    file.write((char *)m_pPixels.get(), m_width*m_height*3);

    // CLOSE FILE
    file.close();

    if(!file){ return false; }
    return true; // CASE OF FILE CORRUPTION
}

// SET PIXEL
void Bitmap::setPixel(int x, int y, std::uint8_t red, std::uint8_t green, std::uint8_t blue){
    // POINTER TO UINT 8 PIXEL WITH SHARED POINTER
    std::uint8_t *pPixel = m_pPixels.get();
    // SET PIXELS FOR EACH CHANNEL
    pPixel += (y * 3) * m_width + ( x * 3);
    pPixel[0] = blue;
    pPixel[1] = green;
    pPixel[2] = red;
}

Bitmap::~Bitmap(){}


#endif /* Bitmap_hpp */

