#ifndef AABB_H
#define AABB_H

#include "rtweekend.h"

class aabb {
  public:
    HOSTDEV aabb() {}
    HOSTDEV aabb(const point3 &a, const point3 &b)
    {
        minimum = a;
        maximum = b;
    }

    HOSTDEV point3 min() const { return minimum; }
    HOSTDEV point3 max() const { return maximum; }

    HOSTDEV inline bool hit(const ray &r, FP_T t_min, FP_T t_max) const
    {
#if 0
        // Andrew Kensler variant
        for (int a = 0; a < 3; a++) {
            auto invD = 1.0f / r.direction()[a];
            auto t0 = (min()[a] - r.origin()[a]) * invD;
            auto t1 = (max()[a] - r.origin()[a]) * invD;
            if (invD < 0.0f) {
#ifndef USE_CUDA
                std::swap(t0, t1);
#else
                auto tmp = t0;
                t0 = t1;
                t1 = tmp;
#endif
            }
            t_min = t0 > t_min ? t0 : t_min;
            t_max = t1 < t_max ? t1 : t_max;
            if (t_max <= t_min) return false;
        }
#else
        // rallen-Manually unrolled a=0,1,2 since this was
        // a hotspot in nsight-compute
        auto invD = 1.0f / r.direction()[0];
        auto t0 = (min()[0] - r.origin()[0]) * invD;
        auto t1 = (max()[0] - r.origin()[0]) * invD;
        if (invD < 0.0f) {
#ifndef USE_CUDA
            std::swap(t0, t1);
#else
            auto tmp = t0;
            t0 = t1;
            t1 = tmp;
#endif
        }
        t_min = t0 > t_min ? t0 : t_min;
        t_max = t1 < t_max ? t1 : t_max;
        if (t_max <= t_min) return false;
        // a = 1
        invD = 1.0f / r.direction()[1];
        t0 = (min()[1] - r.origin()[1]) * invD;
        t1 = (max()[1] - r.origin()[1]) * invD;
        if (invD < 0.0f) {
#ifndef USE_CUDA
            std::swap(t0, t1);
#else
            auto tmp = t0;
            t0 = t1;
            t1 = tmp;
#endif
        }
        t_min = t0 > t_min ? t0 : t_min;
        t_max = t1 < t_max ? t1 : t_max;
        if (t_max <= t_min) return false;
        // a = 2
        invD = 1.0f / r.direction()[2];
        t0 = (min()[2] - r.origin()[2]) * invD;
        t1 = (max()[2] - r.origin()[2]) * invD;
        if (invD < 0.0f) {
#ifndef USE_CUDA
            std::swap(t0, t1);
#else
            auto tmp = t0;
            t0 = t1;
            t1 = tmp;
#endif
        }
        t_min = t0 > t_min ? t0 : t_min;
        t_max = t1 < t_max ? t1 : t_max;
        if (t_max <= t_min) return false;

#endif
        return true;
    }
    HOSTDEV void print(int i) const
    {
        printf("aabb %i ", i);
        minimum.print();
        printf(" - ");
        maximum.print();
    }

    point3 minimum;
    point3 maximum;
};

HOSTDEV aabb surrounding_box(aabb box0, aabb box1)
{
    point3 small(fmin(box0.min().x(), box1.min().x()), fmin(box0.min().y(), box1.min().y()),
                 fmin(box0.min().z(), box1.min().z()));

    point3 big(fmax(box0.max().x(), box1.max().x()), fmax(box0.max().y(), box1.max().y()),
               fmax(box0.max().z(), box1.max().z()));

    return aabb(small, big);
}

#endif