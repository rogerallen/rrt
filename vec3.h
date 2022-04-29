#ifndef VEC3_H
#define VEC3_H
//==============================================================================================
// Originally written in 2016 by Peter Shirley <ptrshrl@gmail.com>
//
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is
// distributed without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication
// along with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==============================================================================================

#include <cmath>
#include <iostream>

using std::fabs;
using std::fmin; // requires --expt-relaxed-constexpr for CUDA
using std::sqrt;

class vec3 {
  public:
    HOSTDEV vec3() : e{0, 0, 0} {}
    HOSTDEV vec3(FP_T e0, FP_T e1, FP_T e2) : e{e0, e1, e2} {}

    HOSTDEV FP_T x() const { return e[0]; }
    HOSTDEV FP_T y() const { return e[1]; }
    HOSTDEV FP_T z() const { return e[2]; }

    HOSTDEV FP_T r() const { return e[0]; }
    HOSTDEV FP_T g() const { return e[1]; }
    HOSTDEV FP_T b() const { return e[2]; }

    HOSTDEV vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    HOSTDEV FP_T operator[](int i) const { return e[i]; }
    HOSTDEV FP_T &operator[](int i) { return e[i]; }

    HOSTDEV void print() const;

    HOSTDEV vec3 &operator+=(const vec3 &v)
    {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    HOSTDEV vec3 &operator*=(const FP_T t)
    {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    HOSTDEV vec3 &operator/=(const FP_T t) { return *this *= 1 / t; }

    HOSTDEV FP_T length() const { return sqrt(length_squared()); }

    HOSTDEV FP_T length_squared() const { return e[0] * e[0] + e[1] * e[1] + e[2] * e[2]; }

    HOSTDEV bool near_zero() const
    {
        // Return true if the vector is close to zero in all dimensions.
        const auto s = 1e-8;
        return (fabs(e[0]) < s) && (fabs(e[1]) < s) && (fabs(e[2]) < s);
    }
    DEV inline static vec3 random(CURAND_STATE_DEF)
    {
        return vec3(random_uniform(CURAND_STATE), random_uniform(CURAND_STATE), random_uniform(CURAND_STATE));
    }

    DEV inline static vec3 random(CURAND_STATE_DEF_COMMA FP_T min, FP_T max)
    {
        return vec3(random_uniform(CURAND_STATE_COMMA min, max), random_uniform(CURAND_STATE_COMMA min, max),
                    random_uniform(CURAND_STATE_COMMA min, max));
    }

  public:
    FP_T e[3];
};

// Type aliases for vec3
using point3 = vec3; // 3D point
using color = vec3;  // RGB color

// vec3 Utility Functions

inline std::ostream &operator<<(std::ostream &out, const vec3 &v)
{
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

HOSTDEV inline vec3 operator+(const vec3 &u, const vec3 &v)
{
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

HOSTDEV inline vec3 operator-(const vec3 &u, const vec3 &v)
{
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

HOSTDEV inline vec3 operator*(const vec3 &u, const vec3 &v)
{
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

HOSTDEV inline vec3 operator*(FP_T t, const vec3 &v) { return vec3(t * v.e[0], t * v.e[1], t * v.e[2]); }

HOSTDEV inline vec3 operator*(const vec3 &v, FP_T t) { return t * v; }

HOSTDEV inline vec3 operator/(vec3 v, FP_T t) { return (1 / t) * v; }

HOSTDEV inline FP_T dot(const vec3 &u, const vec3 &v) { return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2]; }

HOSTDEV inline vec3 cross(const vec3 &u, const vec3 &v)
{
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1], u.e[2] * v.e[0] - u.e[0] * v.e[2],
                u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

HOSTDEV void vec3::print() const { printf("%f %f %f", e[0], e[1], e[2]); }

HOSTDEV inline vec3 unit_vector(vec3 v) { return v / v.length(); }

DEV inline vec3 random_in_unit_disk(CURAND_STATE_DEF)
{
    while (true) {
        auto p = vec3(random_uniform(CURAND_STATE_COMMA - 1, 1), random_uniform(CURAND_STATE_COMMA - 1, 1), 0);
        if (p.length_squared() >= 1) continue;
        return p;
    }
}

DEV inline vec3 random_in_unit_sphere(CURAND_STATE_DEF)
{
    while (true) {
        auto p = vec3::random(CURAND_STATE_COMMA - 1, 1);
        if (p.length_squared() >= 1) continue;
        return p;
    }
}

DEV inline vec3 random_unit_vector(CURAND_STATE_DEF) { return unit_vector(random_in_unit_sphere(CURAND_STATE)); }

DEV inline vec3 random_in_hemisphere(CURAND_STATE_DEF_COMMA const vec3 &normal)
{
    vec3 in_unit_sphere = random_in_unit_sphere(CURAND_STATE);
    if (dot(in_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
        return in_unit_sphere;
    else
        return -in_unit_sphere;
}

HOSTDEV inline vec3 reflect(const vec3 &v, const vec3 &n) { return v - 2 * dot(v, n) * n; }

HOSTDEV inline vec3 refract(const vec3 &uv, const vec3 &n, FP_T etai_over_etat)
{
    auto cos_theta = fmin(dot(-uv, n), 1.0);
    vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    vec3 r_out_parallel = -sqrt(fabs(1.0 - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

#endif