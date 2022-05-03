#ifndef CAMERA_H
#define CAMERA_H

#include "rtweekend.h"

class camera {
  public:
    HOSTDEV camera(point3 lookfrom, point3 lookat, vec3 vup,
                   FP_T vfov, // vertical field-of-view in degrees
                   FP_T aspect_ratio, FP_T aperture, FP_T focus_dist, FP_T _time0 = 0, FP_T _time1 = 0)
    {
        auto theta = degrees_to_radians(vfov);
        auto h = tan(theta / 2);
        auto viewport_height = 2.0 * h;
        auto viewport_width = aspect_ratio * viewport_height;

        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        origin = lookfrom;
        horizontal = focus_dist * viewport_width * u;
        vertical = focus_dist * viewport_height * v;
        lower_left_corner = origin - horizontal / 2 - vertical / 2 - focus_dist * w;

        lens_radius = aperture / 2;
        time0 = _time0;
        time1 = _time1;
    }

    DEV ray get_ray(CURAND_STATE_DEF_COMMA FP_T s, FP_T t) const
    {
        vec3 rd = lens_radius * random_in_unit_disk(CURAND_STATE);
        vec3 offset = u * rd.x() + v * rd.y();

        return ray(origin + offset, lower_left_corner + s * horizontal + t * vertical - origin - offset,
                   random_uniform(CURAND_STATE_COMMA time0, time1));
    }
    HOSTDEV FP_T t0() { return time0; }
    HOSTDEV FP_T t1() { return time1; }

  private:
    point3 origin;
    point3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
    vec3 u, v, w;
    FP_T lens_radius;
    FP_T time0, time1; // shutter open/close times
};
#endif