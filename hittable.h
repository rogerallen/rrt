#ifndef HITTABLE_H
#define HITTABLE_H

#include "aabb.h"
#include "ray.h"

class material;

struct hit_record {
    point3 p;
    vec3 normal;
    material_ptr_t mat_ptr;
    FP_T t;
    bool front_face;

    HOSTDEV inline void set_face_normal(const ray &r, const vec3 &outward_normal)
    {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

class hittable {
  public:
    HOSTDEV virtual bool hit(const ray &r, FP_T t_min, FP_T t_max, hit_record &rec, bool debug) const = 0;
    HOSTDEV virtual bool bounding_box(FP_T time0, FP_T time1, aabb &output_box) const = 0;
    HOSTDEV virtual void print(int i) const = 0;
};

#endif