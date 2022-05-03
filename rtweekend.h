#ifndef RTWEEKEND_H
#define RTWEEKEND_H

#include <cmath>
#include <limits>
#include <memory>
#include <random>

// Usings

using std::make_shared;
using std::shared_ptr;
using std::sqrt;

// CUDA hackery

// set this via -DFP_T=double
//#ifndef USE_FLOAT_NOT_DOUBLE
//#define FP_T double
//#else
//#define FP_T float
//#endif

#ifndef USE_CUDA
#define HOSTDEV
#define DEV
#define CURAND_STATE_DEF
#define CURAND_STATE_DEF_COMMA
#define CURAND_STATE
#define CURAND_STATE_COMMA
#else
#include <curand_kernel.h>
#define HOSTDEV __host__ __device__
#define DEV __device__
#define CURAND_STATE_DEF curandState *state
#define CURAND_STATE_DEF_COMMA curandState *state,
#define CURAND_STATE state
#define CURAND_STATE_COMMA state,
#endif

// Constants

const FP_T infinity = std::numeric_limits<double>::infinity();
const FP_T pi = 3.1415926535897932385;

// Utility Functions

HOSTDEV inline FP_T degrees_to_radians(FP_T degrees) { return degrees * pi / 180.0; }

// we handle this via other means when using CUDA
#ifndef USE_CUDA
inline FP_T random_uniform()
{ // !!! changed from random_double !!!
    static std::uniform_real_distribution<FP_T> distribution(0.0, 1.0);
    static std::mt19937 generator;
    return distribution(generator);
}
inline FP_T random_uniform(FP_T min, FP_T max)
{ // !!! changed from random_double !!!
    // Returns a random real in [min,max).
    return min + (max - min) * random_uniform();
}
#else
DEV inline FP_T random_uniform(curandState *state) { return curand_uniform(state); }
DEV inline FP_T random_uniform(curandState *state, FP_T min, FP_T max)
{
    return min + (max - min) * curand_uniform(state);
}
#endif

inline double clamp(FP_T x, FP_T min, FP_T max)
{
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

// Common Headers

#include "ray.h"
#include "vec3.h"

// Common pointer defines

class material;
class hittable;

#ifndef USE_CUDA
#include <memory>
using std::shared_ptr;
#define hittable_ptr_t shared_ptr<hittable>
#define material_ptr_t shared_ptr<material>
#else
#define hittable_ptr_t hittable *
#define material_ptr_t material *
#endif

#endif