#ifndef MATERIAL_H
#define MATERIAL_H

#include "rtweekend.h"

struct hit_record;

class material {
  public:
    DEV virtual bool scatter(CURAND_STATE_DEF_COMMA const ray &r_in, const hit_record &rec, color &attenuation,
                             ray &scattered, bool debug) const = 0;
    HOSTDEV virtual void print(int i) const = 0;
};

// ==========================================================================================

class lambertian : public material {
  public:
    HOSTDEV lambertian(const color &a) : albedo(a) {}

    DEV virtual bool scatter(CURAND_STATE_DEF_COMMA const ray &r_in, const hit_record &rec, color &attenuation,
                             ray &scattered, bool debug) const override
    {
        auto scatter_direction = rec.normal + random_unit_vector(CURAND_STATE);

        // Catch degenerate scatter direction
        if (scatter_direction.near_zero()) scatter_direction = rec.normal;

        scattered = ray(rec.p, scatter_direction, r_in.time());
        attenuation = albedo;
        return true;
    }
    HOSTDEV virtual void print(int i) const
    {
        printf("material m%d lambertian ", i);
        albedo.print();
        printf("\n");
    }

  public:
    color albedo;
};

// ==========================================================================================

class metal : public material {
  public:
    HOSTDEV metal(const color &a, FP_T f) : albedo(a), fuzz(f < 1 ? f : 1) {}

    DEV virtual bool scatter(CURAND_STATE_DEF_COMMA const ray &r_in, const hit_record &rec, color &attenuation,
                             ray &scattered, bool debug) const override
    {
        vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere(CURAND_STATE), r_in.time());
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0);
    }
    HOSTDEV virtual void print(int i) const
    {
        printf("material m%d metal ", i);
        albedo.print();
        printf(" %f\n", fuzz);
    }

  public:
    color albedo;
    FP_T fuzz;
};

// ==========================================================================================

class dielectric : public material {
  public:
    HOSTDEV dielectric(FP_T index_of_refraction) : ir(index_of_refraction) {}

    DEV virtual bool scatter(CURAND_STATE_DEF_COMMA const ray &r_in, const hit_record &rec, color &attenuation,
                             ray &scattered, bool debug) const override
    {
        attenuation = color(1.0, 1.0, 1.0);
        FP_T refraction_ratio = rec.front_face ? (1.0 / ir) : ir;

        vec3 unit_direction = unit_vector(r_in.direction());
        FP_T cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0);
        FP_T sin_theta = sqrt(1.0 - cos_theta * cos_theta);

        bool cannot_refract = refraction_ratio * sin_theta > 1.0;
        vec3 direction;

        if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random_uniform(CURAND_STATE))
            direction = reflect(unit_direction, rec.normal);
        else
            direction = refract(unit_direction, rec.normal, refraction_ratio);

        scattered = ray(rec.p, direction, r_in.time());
        return true;
    }
    HOSTDEV virtual void print(int i) const { printf("material m%d dielectric %f\n", i, ir); }

  public:
    FP_T ir; // Index of Refraction

  private:
    HOSTDEV static FP_T reflectance(FP_T cosine, FP_T ref_idx)
    {
        // Use Schlick's approximation for reflectance.
        auto r0 = (1 - ref_idx) / (1 + ref_idx);
        r0 = r0 * r0;
        return r0 + (1 - r0) * pow((1 - cosine), 5);
    }
};

#endif
