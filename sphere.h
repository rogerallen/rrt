#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.h"
#include "material.h"
#include "vec3.h"

class sphere : public hittable {
  public:
    DEV sphere() {}
    DEV sphere(point3 cen, FP_T r, material_ptr_t m) : center(cen), radius(r), mat_ptr(m){};
#ifdef USE_CUDA
    DEV ~sphere()
    {
        // NOTE -- this is deleting materials passed to the sphere
        // this works for our use case, but might not be best practice
        // materials can be shared so code defensively
        if (mat_ptr != nullptr) delete mat_ptr;
        mat_ptr = nullptr;
    }
#endif
    DEV virtual bool hit(const ray &r, FP_T t_min, FP_T t_max, hit_record &rec, bool debug) const override;
    DEV virtual bool bounding_box(FP_T time0, FP_T time1, aabb &output_box) const override;

    DEV virtual void print(int i) const;

  public:
    point3 center;
    FP_T radius;
    material_ptr_t mat_ptr;
};

DEV bool sphere::hit(const ray &r, FP_T t_min, FP_T t_max, hit_record &rec, bool debug) const
{
    vec3 oc = r.origin() - center;
    auto a = r.direction().length_squared();
    auto half_b = dot(oc, r.direction());
    auto c = oc.length_squared() - radius * radius;

    auto discriminant = half_b * half_b - a * c;
    if (discriminant < 0) return false;
    auto sqrtd = sqrt(discriminant);

    // Find the nearest root that lies in the acceptable range.
    auto root = (-half_b - sqrtd) / a;
    if (root < t_min || t_max < root) {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || t_max < root) return false;
    }

    rec.t = root;
    rec.p = r.at(rec.t);
    vec3 outward_normal = (rec.p - center) / radius;
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = mat_ptr;

    return true;
}

DEV bool sphere::bounding_box(FP_T time0, FP_T time1, aabb &output_box) const
{
    output_box = aabb(center - vec3(radius, radius, radius), center + vec3(radius, radius, radius));
    return true;
}

DEV void sphere::print(int i) const
{
    mat_ptr->print(i);
    printf("sphere ");
    center.print();
    printf(" %f m%d\n", radius, i);
}

#endif