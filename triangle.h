#ifndef TRIANGLEH
#define TRIANGLEH

#include "hitable.h"

__device__ vec3 get_normal(vec3 v0, vec3 v1, vec3 v2)
{
    vec3 v01 = v1 - v0;
    vec3 v02 = v2 - v0;
    return cross(v01, v02);
}

class triangle : public hitable {
  public:
    __device__ triangle() {}
    __device__ triangle(vec3 v0, vec3 v1, vec3 v2, material *m) : mat_ptr(m)
    {
        vertices[0] = v0;
        vertices[1] = v1;
        vertices[2] = v2;
        normal = get_normal(v0, v1, v2);
    };
    __device__ virtual bool hit(const ray &r, FP_T tmin, FP_T tmax, hit_record &rec, bool debug) const;
    __device__ virtual void print(int i) const;
    vec3 vertices[3];
    vec3 normal;
    material *mat_ptr;
};

__device__ bool triangle::hit(const ray &r, FP_T t_min, FP_T t_max, hit_record &rec, bool debug) const
{
    // https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
    const FP_T EPSILON = 0.0000001;
    vec3 edge1 = vertices[1] - vertices[0];
    vec3 edge2 = vertices[2] - vertices[0];
    vec3 h = cross(r.direction(), edge2);
    FP_T a = dot(edge1, h);
    if (a > -EPSILON && a < EPSILON) {
        return false; // This ray is parallel to this triangle.
    }
    FP_T f = 1.0 / a;
    vec3 s = r.origin() - vertices[0];
    FP_T u = dot(f * s, h);
    if (u < 0.0 || u > 1.0) {
        return false;
    }
    vec3 q = cross(s, edge1);
    FP_T v = dot(f * r.direction(), q);
    if (v < 0.0 || u + v > 1.0) {
        return false;
    }
    if (debug) printf("DEBUG tri u=%f v=%f\n", u, v);
    // At this stage we can compute t to find out where the intersection point is on the line.
    float t = dot(f * edge2, q);
    if (t > EPSILON) { // ray intersection
        // outIntersectionPoint = rayOrigin + rayVector * t;
        // need to handle the t_min, t_max bit...
        if ((t > t_min) && (t < t_max)) {
            rec.t = t;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = normal;
            rec.mat_ptr = mat_ptr;
            return true;
        }
        else
            return false;
    }
    // This means that there is a line intersection but not a ray intersection.
    return false;
}

__device__ void triangle::print(int i) const
{
    mat_ptr->print(i);
    printf("triangle ");
    vertices[0].print();
    vertices[1].print();
    vertices[2].print();
    normal.print();
    printf(" m%d\n", i);
}

#endif
