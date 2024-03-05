#ifndef CV_CORE_CVDEF_HPP_
#define CV_CORE_CVDEF_HPP_

/* reference: include/opencv2/core/cvdef.h
              include/opencv2/core/typedef_c.h
*/

#include "interface.hpp"

#ifdef _MSC_VER
	#define CV_EXPORTS __declspec(dllexport)
	#define CV_DECL_ALIGNED(x) __declspec(align(x))
#else
	#define CV_EXPORTS __attribute__((visibility("default")))
	#define CV_DECL_ALIGNED(x) __attribute__((aligned(x)))
#endif

#define CV_CN_MAX		512
#define CV_CN_SHIFT		3
#define CV_DEPTH_MAX		(1 << CV_CN_SHIFT)

#define CV_MAT_TYPE_MASK	(CV_DEPTH_MAX*CV_CN_MAX - 1)
#define CV_MAT_TYPE(flags)	((flags) & CV_MAT_TYPE_MASK)

#ifndef MIN
	#define MIN(a,b)  ((a) > (b) ? (b) : (a))
#endif

#ifndef MAX
	#define MAX(a,b)  ((a) < (b) ? (b) : (a))
#endif

#define CV_CN_MAX  512

// Common macros and inline functions
#define CV_SWAP(a,b,t) ((t) = (a), (a) = (b), (b) = (t))

/** min & max without jumps */
#define  CV_IMIN(a, b)  ((a) ^ (((a)^(b)) & (((a) < (b)) - 1)))
#define  CV_IMAX(a, b)  ((a) ^ (((a)^(b)) & (((a) > (b)) - 1)))

// fundamental constants
#define CV_PI 3.1415926535897932384626433832795

// Note: No practical significance
class dump {};

typedef union Cv32suf {
	int i;
	unsigned u;
	float f;
} Cv32suf;

typedef union Cv64suf {
	int64 i;
	uint64 u;
	double f;
} Cv64suf;

#endif // CV_CORE_CVDEF_HPP_
