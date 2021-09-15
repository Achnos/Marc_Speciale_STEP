#include <iostream>
#include <iomanip>
#include <new>      //For std::nothrow
#include "ccd.hpp"

extern "C"  //Tells the compile to use C-linkage for the next scope.
{
    //Note: The interface this linkage region needs to use C only.
    void* constructor_wrapper( void )
    {
        // Note: Inside the function body, we can use C++.
        return new(std::nothrow) ccd;
    }

    void destructor_wrapper (void *self_ptr)
    {
        free(self_ptr);
    }

    double ccd::sum_array(int array_length, void *array)
    {
        int* int_array = reinterpret_cast<int*>(array); // this is a pointer to arr, reintepreted in intarr
        double sum = 0;
        for (int id = 0; id < array_length; id++)
        {
            sum += int_array[id];
        }

        return sum;
    }

    double sum_array_wrapper(void *self_ptr, int array_length, void *array)
    {
        // Note: A downside here is the lack of type safety.
        // You could always internally(in the C++ library) save a reference to all
        // pointers created of type MyClass and verify it is an element in that
        // structure.
        // We should avoid throwing exceptions.
        try
        {
            ccd* ref = reinterpret_cast<ccd*>(self_ptr);
            int res = ref -> sum_array(array_length, array);

            return res;
        }
        catch(...)
        {
            return -1; //assuming -1 is an error condition.
        }
    }

} //End C linkage scope.

