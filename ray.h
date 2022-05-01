#ifndef RAY_H
#define RAY_H

#include "vec3.h"

class ray {
  public:
    DEV ray() {}
    DEV ray(const point3 &origin, const vec3 &direction, FP_T time = 0.0) : orig(origin), dir(direction), tm(time) {}

    DEV point3 origin() const { return orig; }
    DEV vec3 direction() const { return dir; }
    DEV FP_T time() const { return tm; }

    DEV point3 at(FP_T t) const { return orig + t * dir; }

  public:
    point3 orig;
    vec3 dir;
    FP_T tm;
};

#endif
