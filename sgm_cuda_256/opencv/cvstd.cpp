// reference: modules/core/src/alloc.cpp

#include <malloc.h>
#include <stdio.h>
#include "cvstd.hpp"
#include "interface.hpp"
#include "utility.hpp"
#include "base.hpp"

// Allocates an aligned memory buffer
void* fastMalloc(size_t size)
{
	uchar* udata = (uchar*)malloc(size + sizeof(void*) + CV_MALLOC_ALIGN);
	if (!udata) {
		fprintf(stderr, "failed to allocate %lu bytes\n", (unsigned long)size);
		return NULL;
	}
	uchar** adata = alignPtr((uchar**)udata + 1, CV_MALLOC_ALIGN);
	adata[-1] = udata;
	return adata;
}

// Deallocates a memory buffer
void fastFree(void* ptr)
{
	if (ptr) {
		uchar* udata = ((uchar**)ptr)[-1];
		CV_Assert(udata < (uchar*)ptr && ((uchar*)ptr - udata) <= (ptrdiff_t)(sizeof(void*) + CV_MALLOC_ALIGN));
		free(udata);
	}
}

void* cvAlloc(size_t size)
{
	return fastMalloc(size);
}

void cvFree_(void* ptr)
{
	fastFree(ptr);
}
