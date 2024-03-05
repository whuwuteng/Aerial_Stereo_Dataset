#ifndef CV_CORE_FAST_MATH_HPP_
#define CV_CORE_FAST_MATH_HPP_

// reference: include/opencv2/core/fast_math.hpp

#include "cvdef.hpp"

// Rounds floating-point number to the nearest integer
static inline int CVRound(double value)
{
	// it's ok if round does not comply with IEEE754 standard;
	// it should allow +/-1 difference when the other functions use round
	return (int)(value + (value >= 0 ? 0.5 : -0.5));
}

static inline int CVRound(float value)
{
	// it's ok if round does not comply with IEEE754 standard;
	// it should allow +/-1 difference when the other functions use round
	return (int)(value + (value >= 0 ? 0.5f : -0.5f));
}

static inline int CVRound(int value)
{
	return value;
}

// Rounds floating-point number to the nearest integer not larger than the original
static inline int CVFloor(double value)
{
	int i = CVRound(value);
	float diff = (float)(value - i);
	return i - (diff < 0);
}

static inline int CVFloor(float value)
{
	int i = CVRound(value);
	float diff = (float)(value - i);
	return i - (diff < 0);
}

static inline int CVFloor(int value)
{
	return value;
}

// Rounds floating-point number to the nearest integer not smaller than the original
static inline int CVCeil(double value)
{
	int i = CVRound(value);
	float diff = (float)(i - value);
	return i + (diff < 0);
}

static inline int CVCeil(float value)
{
	int i = CVRound(value);
	float diff = (float)(i - value);
	return i + (diff < 0);
}

static inline int CVCeil(int value)
{
	return value;
}

#endif // CV_CORE_FAST_MATH_HPP_
