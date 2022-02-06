#ifndef SPHEREH
#define SPHEREH

#include "hitable.h"

class sphere : public hitable {
  public:
    __device__ sphere() {}
    __device__ sphere(vec3 cen, FP_T r, material *m) : center(cen), radius(r), mat_ptr(m){};
    __device__ virtual bool hit(const ray &r, FP_T tmin, FP_T tmax, hit_record &rec) const;
    __device__ virtual void print(int i) const;
    vec3 center;
    FP_T radius;
    material *mat_ptr;
};

__device__ bool sphere::hit(const ray &r, FP_T t_min, FP_T t_max, hit_record &rec) const
{
    vec3 oc = r.origin() - center;
    FP_T a = dot(r.direction(), r.direction());
    FP_T b = dot(oc, r.direction());
    FP_T c = dot(oc, oc) - radius * radius;
    FP_T discriminant = b * b - a * c;
    if (discriminant > 0) {
        FP_T temp = (-b - sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center) / radius;
            rec.mat_ptr = mat_ptr;
            return true;
        }
        temp = (-b + sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center) / radius;
            rec.mat_ptr = mat_ptr;
            return true;
        }
    }
    return false;
}

__device__ void sphere::print(int i) const
{
    mat_ptr->print(i);
    printf("sphere ");
    center.print();
    printf(" %f m%d\n", radius, i);
}

#endif
