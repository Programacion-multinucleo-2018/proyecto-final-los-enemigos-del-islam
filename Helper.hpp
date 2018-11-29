//
//  Helper.hpp
//  HelperTools
//
//  Created by Sebastián Galguera on 3/12/17.
//  Copyright © 2017 Sebastián Galguera. All rights reserved.
//

#ifndef Helper_hpp
#define Helper_hpp

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <list>
#include <sstream>
#include <climits>
#include <string>
#include <typeinfo>
#include <cstdlib>
#include <memory>
#include <cxxabi.h>
#include <iomanip>
#include <chrono>


// ENUM FOR THE MENU PRINTING
enum Menu { MAIN = 1, FRACTAL, FRACTALTYPE, COLORTYPE, CONFIRMATION};

// HELPER NAMESPACE
namespace Helper{

    // MENU STRINGS TO POPULATE MENU
    std::vector<std::string> menuString = {
        "configure fractal", "save fractal", // 0 1
        "select type of fractal", "select zoom", "select colors", // 2 4
        "mandelbrot sequential", "mandelbrot parallel", // 5 6
        "Hexadecimal", "RGB", // 7 8
        "overwrite", "cancel", // 9 10
        "exit"
    };

    // CONSOLE HANDLING FOR PROMPTS AND PRINTING ERRORS / MESSAGES / MENUS
    class ConsoleHandling{
    public:

        // PRINT WELCOME
        static void printWelcome(){
            std::cout << "-------------------------------" << std::endl;
            std::cout << "Welcome to the fractal creator" << std::endl;
            std::cout << "-------------------------------" << std::endl;
        }

        // PRINT DYNAMIC MENU
        static void printMenu(int mType){
            int left, right;

            // SET MENU LIMITS
            // TO DO: MAKE IT A DYNAMIC VALUE
            if(mType == 1){
                left = 0;
                right = 1;
            }else if (mType == 2){
                left = 2;
                right = 4;
            }else if (mType == 3){
                left = 5;
                right = 6;
            }else if (mType == 4){
                left = 7;
                right = 8;
            }else{
                left = 9;
                right = 10;
            }


            int offset = 0;
            for(int i = left, label = 1; i <= right; i++, label++){
                // PRINT MENU WITH OFFSET
                offset = int(5 + menuString.at(i).size());
                std::cout << std::setw(6) << label << " : " << std::setw(offset) << menuString.at(i) << std::endl;
            }

            // PRINT EXIT OPTION AVAILABLE IN ALL MENUS
            offset = int(5 + menuString.back().size());
            std::cout << std::setw(9) << "0 : " << std::setw(offset) << menuString.back() << std::endl;
        }

        // PRINT ERROR
        static int printError(std::string e){
            std::cout << "-------------------------------" << std::endl;
            std::cout << "Error: " << e << std::endl;
            std::cout << std::endl;
            return 1;
        }

        // PRINT MESSAGE
        static int printMessage(std::string m){
            std::cout << "\n  > " << m << std::endl;
            std::cout << std::endl;
            return 1;
        }

        // GET PROMPT
        template <typename T>
        static T getPrompt(std::string label){
            T prompt;
            std::cout << "\n" + label + ": ";
            std::cin >> prompt;
            std::cin.ignore();
            return prompt;
        }

        // GET VALID PROMPT WITH TEMPLATE SPECIFICATION
        template <typename T>
        static T getValidPrompt(std::string label, double limit, std::string e){
            T prompt{};
            bool flag = false;
            while(!flag){
                prompt = getPrompt<T>(label);
                validatePrompt<T>(prompt, limit) ? flag = true : printError(e);
            };
            return prompt;
        }

        // VALIDATE PROMPT FOR SPECIFICATION
        // TO DO: CUSTOM TYPE SPECIFICATION WITH LAMBDA OR FUNCTION POINTER
        template <typename T>
        static bool validatePrompt(T prompt, double limit){}

    };

    // STRING SPECIFICATION
    template <>
    bool ConsoleHandling::validatePrompt<std::string>(std::string prompt, double limit){
        if(prompt.size() > limit)
            return 0;
        return 1;
    }

    // INT SPECIFICATION
    template <>
    bool ConsoleHandling::validatePrompt<int>(int prompt, double limit){
        if(prompt > limit || prompt < 0)
            return 0;
        return 1;
    }

    // DOUBLE SPECIFICATION
    template <>
    bool ConsoleHandling::validatePrompt<double>(double prompt, double limit){
        if(prompt > limit || prompt < 0)
            return 0;
        return 1;
    }

    // TYPE CHECKING TOOLS
    class TypeChecking{
    public:

        // CHECK TYPE
        template <typename T>
        static std::string checkType(const T& t) {
            return demangle(typeid(t).name());
        }

        // DEMANGLE TYPE TO GET CUSTOM NAME
        static std::string demangle(const char* name) {
            int status = -10;
            std::unique_ptr<char, void(*)(void*)> res {
                abi::__cxa_demangle(name, NULL, NULL, &status),
                std::free
            };
            return (status==0) ? res.get() : name ;
        }

        // ASSERT TYPE WITH STRING LITERAL
        template <typename T>
        static bool assertType(T variable, std::string typeName){
            if(typeName == checkType(variable))
                return 1;
            return 0;
        }

        // COMPARE TYPES VIA THEIR CUSTOM NAMES
        template <typename T, typename V>
        static bool compareTypes(T variable, V comparingVariable){
            if(checkType(comparingVariable) == checkType(variable))
                return 1;
            return 0;
        }

    };

    // STRING HANDLING
    class StringHandling{
    public:
        static std::string toUpper(std::string str){
            for(auto& charUpper: str){
                charUpper = toupper(charUpper);
            }
            return str;
        };

        static std::string toLower(std::string str){
            for(auto& charUpper: str){
                charUpper = tolower(charUpper);
            }
            return str;
        };
    };

    // CONTAINER HANDLING
    class ContainerHandling{
    public:

        // CHECK SIZE OF ALLOCATED ARRAY
        template <typename T>
        static int getSizeOfAllocatedArray(T arr){
            int count = 0;
            while(*(arr+count) != NULL){
                count++;
            }

            return count;
        }

        // CHECK SIZE OF UNALLOCATED ARRAY
        template <typename T, size_t N>
        static int getSizeOfUnAllocatedArray(T (&arr)[N]){
            size_t size = N;
            return size & INT_MAX;
        }
    };

    // TIME HANDLING
    class TimeHandling{
    public:
        template <typename T, void(T::*foo)()>
        static void calculateTimeCPU(T *p, std::string name){
            auto start_cpu = std::chrono::high_resolution_clock::now();
            (p->*foo)();
            auto end_cpu =  std::chrono::high_resolution_clock::now();
            std::chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;
            std::string mainFoo = __FUNCTION__;
            Helper::ConsoleHandling::printMessage(mainFoo + " " + name + " " + std::to_string(duration_ms.count()));
        }

    };

};

#endif /* helper_hpp */

