#ifndef TRIANGLEH
#define TRIANGLEH

#include "hittable.h"

//   2 - 1    indices should go around in CCW direction
//   |  /     so that the normal is cross(0->1,0->2)
//   0
HOSTDEV vec3 get_normal(vec3 v0, vec3 v1, vec3 v2)
{
    vec3 v01 = unit_vector(v1 - v0);
    vec3 v02 = unit_vector(v2 - v0);
    vec3 c = unit_vector(cross(v01, v02));
    return c;
}

class triangle : public hittable {
  public:
    HOSTDEV triangle() {}
    HOSTDEV triangle(vec3 v0, vec3 v1, vec3 v2, material_ptr_t m) : mat_ptr(m)
    {
        vertices[0] = v0;
        vertices[1] = v1;
        vertices[2] = v2;
        normal = get_normal(v0, v1, v2);
    };
    HOSTDEV virtual bool hit(const ray &r, FP_T tmin, FP_T tmax, hit_record &rec, bool debug) const;
    HOSTDEV virtual bool bounding_box(FP_T time0, FP_T time1, aabb &output_box) const override;
    HOSTDEV virtual void print(int i) const;
    vec3 vertices[3];
    vec3 normal;
    material_ptr_t mat_ptr;
};

HOSTDEV bool triangle::hit(const ray &r, FP_T t_min, FP_T t_max, hit_record &rec, bool debug) const
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
            rec.p = r.at(rec.t);
            rec.set_face_normal(r, normal);
            rec.mat_ptr = mat_ptr;
            return true;
        }
        else
            return false;
    }
    // This means that there is a line intersection but not a ray intersection.
    return false;
}

HOSTDEV bool triangle::bounding_box(FP_T _time0, FP_T _time1, aabb &output_box) const
{
    point3 small(fmin(fmin(vertices[0].x(), vertices[1].x()), vertices[2].x()),
                 fmin(fmin(vertices[0].y(), vertices[1].y()), vertices[2].y()),
                 fmin(fmin(vertices[0].z(), vertices[1].z()), vertices[2].z()));
    point3 big(fmax(fmax(vertices[0].x(), vertices[1].x()), vertices[2].x()),
               fmax(fmax(vertices[0].y(), vertices[1].y()), vertices[2].y()),
               fmax(fmax(vertices[0].z(), vertices[1].z()), vertices[2].z()));
    output_box = aabb(small, big);
    return true;
}

HOSTDEV void triangle::print(int i) const
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
