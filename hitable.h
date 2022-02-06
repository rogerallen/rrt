#ifndef HITABLEH
#define HITABLEH

#include "ray.h"

class material;

struct hit_record {
    FP_T t;
    vec3 p;
    vec3 normal;
    material *mat_ptr;
};

class hitable {
  public:
    __device__ virtual bool hit(const ray &r, FP_T t_min, FP_T t_max, hit_record &rec) const = 0;
    __device__ virtual void print(int i) const = 0;
};

#endif
