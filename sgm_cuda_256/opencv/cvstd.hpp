#ifndef CV_CORE_CVSTD_HPP_
#define CV_CORE_CVSTD_HPP_

// reference: include/opencv2/core/cvstd.hpp

#include "cvdef.hpp"

#ifndef __cplusplus
	#error CVstd.hpp header must be compiled as C++
#endif

/* the alignment of all the allocated buffers */
#define  CV_MALLOC_ALIGN    16

// Allocates an aligned memory buffer
CV_EXPORTS void* fastMalloc(size_t size);
// Deallocates a memory buffer
CV_EXPORTS void fastFree(void* ptr);
void* cvAlloc(size_t size);
void cvFree_(void* ptr);
#define cvFree(ptr) (cvFree_(*(ptr)), *(ptr)=0)

#endif // CV_CORE_CVSTD_HPP_
