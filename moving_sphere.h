#ifndef MOVING_SPHERE_H
#define MOVING_SPHERE_H

#include "rtweekend.h"

#include "hittable.h"

class moving_sphere : public hittable {
  public:
    HOSTDEV moving_sphere() {}
    HOSTDEV moving_sphere(point3 cen0, point3 cen1, FP_T _time0, FP_T _time1, FP_T r, material_ptr_t m)
        : center0(cen0), center1(cen1), time0(_time0), time1(_time1), radius(r), mat_ptr(m){};

    HOSTDEV virtual bool hit(const ray &r, FP_T t_min, FP_T t_max, hit_record &rec, bool debug) const override;
    HOSTDEV virtual bool bounding_box(FP_T _time0, FP_T _time1, aabb &output_box) const override;
    HOSTDEV virtual void print(int i) const;

    HOSTDEV point3 center(FP_T time) const;

  public:
    point3 center0, center1;
    FP_T time0, time1;
    FP_T radius;
    material_ptr_t mat_ptr;
};

HOSTDEV point3 moving_sphere::center(FP_T time) const
{
    return center0 + ((time - time0) / (time1 - time0)) * (center1 - center0);
}

HOSTDEV bool moving_sphere::hit(const ray &r, FP_T t_min, FP_T t_max, hit_record &rec, bool debug) const
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

HOSTDEV bool moving_sphere::bounding_box(FP_T _time0, FP_T _time1, aabb &output_box) const
{
    aabb box0(center(_time0) - vec3(radius, radius, radius), center(_time0) + vec3(radius, radius, radius));
    aabb box1(center(_time1) - vec3(radius, radius, radius), center(_time1) + vec3(radius, radius, radius));
    output_box = surrounding_box(box0, box1);
    return true;
}

HOSTDEV void moving_sphere::print(int i) const
{
    printf("moving sphere %d c ", i);
    center0.print();
    printf(" - ");
    center1.print();
    printf(" t %f - %f r %f ", time0, time1, radius);
    mat_ptr->print(i);
}

#endif