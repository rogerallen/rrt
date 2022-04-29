#ifndef HITABLELISTH
#define HITABLELISTH

#include "hitable.h"

class hitable_list : public hitable {
  public:
    __device__ hitable_list() {}
    __device__ hitable_list(hitable **l, int n)
    {
        list = l;
        list_size = n;
    }
    __device__ virtual bool hit(const ray &r, FP_T tmin, FP_T tmax, hit_record &rec, bool debug) const;
    __device__ virtual void print(int i) const;
    hitable **list;
    int list_size;
};

__device__ bool hitable_list::hit(const ray &r, FP_T t_min, FP_T t_max, hit_record &rec, bool debug) const
{
    hit_record temp_rec;
    bool hit_anything = false;
    FP_T closest_so_far = t_max;
    for (int i = 0; i < list_size; i++) {
        if (debug) printf("DEBUG hit test %d\n", i);
        if (list[i]->hit(r, t_min, closest_so_far, temp_rec, debug)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
            if (debug) printf("DEBUG ! hit !\n");
        }
    }
    return hit_anything;
}

__device__ void hitable_list::print(int i) const { printf("hitable_list print %d?\n", i); }

#endif
