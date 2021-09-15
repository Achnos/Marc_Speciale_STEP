#include <iostream>
#include <iomanip>
#include <new> //For std::nothrow
#include "ccd.hpp"

extern "C"  //Tells the compile to use C-linkage for the next scope.
{
    //Note: The interface this linkage region needs to use C only.
    void * CreateInstanceOfClass( void )
    {
        // Note: Inside the function body, I can use C++.
        return new(std::nothrow) ccd;
    }

    void DeleteInstanceOfClass (void *ptr)
    {
        free(ptr);
    }

    double ccd::test(int arr_len, void *arr) {

        int* intarr = reinterpret_cast<int*>(arr); // this is a pointer to arr, reintepreted in intarr
        double sum = 0;
        for (int id = 0; id < arr_len; id++){
            sum += intarr[id];
        }

        return sum;
    }

    double CallMemberTest(void *ptr, int arr_len, void *arr)
    {

        // Note: A downside here is the lack of type safety.
        // You could always internally(in the C++ library) save a reference to all
        // pointers created of type MyClass and verify it is an element in that
        // structure.
        // We should avoid throwing exceptions.
        try
        {
            ccd* ref = reinterpret_cast<ccd*>(ptr);
            int res = ref->test(arr_len, arr);
            free(ref);
            return res;
        }
        catch(...)
        {
            return -1; //assuming -1 is an error condition.
        }
    }

} //End C linkage scope.

