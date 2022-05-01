#ifndef MOVING_SPHERE_H
#define MOVING_SPHERE_H

#include "rtweekend.h"

#include "hittable.h"

class moving_sphere : public hittable {
  public:
    DEV moving_sphere() {}
    DEV moving_sphere(point3 cen0, point3 cen1, FP_T _time0, FP_T _time1, FP_T r, material_ptr_t m)
        : center0(cen0), center1(cen1), time0(_time0), time1(_time1), radius(r), mat_ptr(m){};

    DEV virtual bool hit(const ray &r, FP_T t_min, FP_T t_max, hit_record &rec, bool debug) const override;
    DEV virtual void print(int i) const;

    DEV point3 center(FP_T time) const;

  public:
    point3 center0, center1;
    FP_T time0, time1;
    FP_T radius;
    material_ptr_t mat_ptr;
};

DEV point3 moving_sphere::center(FP_T time) const
{
    return center0 + ((time - time0) / (time1 - time0)) * (center1 - center0);
}

DEV bool moving_sphere::hit(const ray &r, FP_T t_min, FP_T t_max, hit_record &rec, bool debug) const
{
    if (debug) printf("time=%f\n", r.time());
    vec3 oc = r.origin() - center(r.time());
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
    auto outward_normal = (rec.p - center(r.time())) / radius;
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = mat_ptr;

    return true;
}

DEV void moving_sphere::print(int i) const
{
    mat_ptr->print(i);
    printf("moving sphere ");
    center0.print();
    printf(" - ");
    center1.print();
    printf("\n %f - %f\n", time0, time1);
    printf(" %f m%d\n", radius, i);
}

#endif