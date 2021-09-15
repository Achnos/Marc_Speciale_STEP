#ifdef _MSC_VER                                     // Conditional compilation in case we are using Visual C/C++
    #define EXPORT_SYMBOL __declspec(dllexport)     // We are creating DLL's, which does not export any symbols
#else                                               // by default (unlike static library). In Visual C++,
    #define EXPORT_SYMBOL                           // __declspec(dllexport) is the easiest way to export symbols.
#endif                                              // In any case we should EXPORT_SYMBOL for use with bindings

EXPORT_SYMBOL float cmult(int int_param, float float_param);