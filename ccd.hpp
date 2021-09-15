#ifndef CCD_H
#define CCD_H

#ifdef _MSC_VER                                     // Conditional compilation in case we are using Visual C/C++
    #define EXPORT_SYMBOL __declspec(dllexport)     // We are creating DLL's, which does not export any symbols
#else                                               // by default (unlike static library). In Visual C++,
    #define EXPORT_SYMBOL                           // __declspec(dllexport) is the easiest way to export symbols.
#endif                                              // In any case we should EXPORT_SYMBOL for use with bindings

#include <new> //For std::nothrow

                                                    // We must check to see if we are compiling with g++
#ifdef __cplusplus                                  // C++ does name mangling to enable overloading, and we thus
    extern "C" {                                        // need the extern "C" statement, for the linker to be able to
    #endif                                              // find the proper function to link. In addition it is needed
                                                        // since name mangling can cause issues when using C-bindings

        class ccd {
            private:

            public:
                EXPORT_SYMBOL double sum_array(int array_length, void* array);
        };

        EXPORT_SYMBOL void* constructor_wrapper(void);

        EXPORT_SYMBOL void destructor_wrapper(void* ptr);

        EXPORT_SYMBOL double sum_array_wrapper(void* ptr, int array_length, void* array);


    #ifdef __cplusplus
    }
    #endif



#endif