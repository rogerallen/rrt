#ifndef VEC3H
#define VEC3H

#include <iostream>
#include <math.h>
#include <stdlib.h>

class vec3 {

  public:
    __host__ __device__ vec3() {}
    __host__ __device__ vec3(FP_T e0, FP_T e1, FP_T e2)
    {
        e[0] = e0;
        e[1] = e1;
        e[2] = e2;
    }
    __host__ __device__ inline FP_T x() const { return e[0]; }
    __host__ __device__ inline FP_T y() const { return e[1]; }
    __host__ __device__ inline FP_T z() const { return e[2]; }
    __host__ __device__ inline FP_T r() const { return e[0]; }
    __host__ __device__ inline FP_T g() const { return e[1]; }
    __host__ __device__ inline FP_T b() const { return e[2]; }

    __host__ __device__ inline const vec3 &operator+() const { return *this; }
    __host__ __device__ inline vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    __host__ __device__ inline FP_T operator[](int i) const { return e[i]; }
    __host__ __device__ inline FP_T &operator[](int i) { return e[i]; };

    __host__ __device__ void print() const;

    __host__ __device__ inline vec3 &operator+=(const vec3 &v2);
    __host__ __device__ inline vec3 &operator-=(const vec3 &v2);
    __host__ __device__ inline vec3 &operator*=(const vec3 &v2);
    __host__ __device__ inline vec3 &operator/=(const vec3 &v2);
    __host__ __device__ inline vec3 &operator*=(const FP_T t);
    __host__ __device__ inline vec3 &operator/=(const FP_T t);

    __host__ __device__ inline FP_T length() const { return sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]); }
    __host__ __device__ inline FP_T squared_length() const { return e[0] * e[0] + e[1] * e[1] + e[2] * e[2]; }
    __host__ __device__ inline void make_unit_vector();

    FP_T e[3];
};

inline std::istream &operator>>(std::istream &is, vec3 &t)
{
    is >> t.e[0] >> t.e[1] >> t.e[2];
    return is;
}

inline std::ostream &operator<<(std::ostream &os, const vec3 &t)
{
    os << t.e[0] << " " << t.e[1] << " " << t.e[2];
    return os;
}

__host__ __device__ void vec3::print() const { printf("%f %f %f", e[0], e[1], e[2]); }

__host__ __device__ inline void vec3::make_unit_vector()
{
    FP_T k = 1.0 / sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
    e[0] *= k;
    e[1] *= k;
    e[2] *= k;
}

__host__ __device__ inline vec3 operator+(const vec3 &v1, const vec3 &v2)
{
    return vec3(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3 &v1, const vec3 &v2)
{
    return vec3(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3 &v1, const vec3 &v2)
{
    return vec3(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
}

__host__ __device__ inline vec3 operator/(const vec3 &v1, const vec3 &v2)
{
    return vec3(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
}

__host__ __device__ inline vec3 operator*(FP_T t, const vec3 &v) { return vec3(t * v.e[0], t * v.e[1], t * v.e[2]); }

__host__ __device__ inline vec3 operator/(vec3 v, FP_T t) { return vec3(v.e[0] / t, v.e[1] / t, v.e[2] / t); }

__host__ __device__ inline vec3 operator*(const vec3 &v, FP_T t) { return vec3(t * v.e[0], t * v.e[1], t * v.e[2]); }

__host__ __device__ inline FP_T dot(const vec3 &v1, const vec3 &v2)
{
    return v1.e[0] * v2.e[0] + v1.e[1] * v2.e[1] + v1.e[2] * v2.e[2];
}

__host__ __device__ inline vec3 cross(const vec3 &v1, const vec3 &v2)
{
    return vec3((v1.e[1] * v2.e[2] - v1.e[2] * v2.e[1]), (-(v1.e[0] * v2.e[2] - v1.e[2] * v2.e[0])),
                (v1.e[0] * v2.e[1] - v1.e[1] * v2.e[0]));
}

__host__ __device__ inline vec3 &vec3::operator+=(const vec3 &v)
{
    e[0] += v.e[0];
    e[1] += v.e[1];
    e[2] += v.e[2];
    return *this;
}

__host__ __device__ inline vec3 &vec3::operator*=(const vec3 &v)
{
    e[0] *= v.e[0];
    e[1] *= v.e[1];
    e[2] *= v.e[2];
    return *this;
}

__host__ __device__ inline vec3 &vec3::operator/=(const vec3 &v)
{
    e[0] /= v.e[0];
    e[1] /= v.e[1];
    e[2] /= v.e[2];
    return *this;
}

__host__ __device__ inline vec3 &vec3::operator-=(const vec3 &v)
{
    e[0] -= v.e[0];
    e[1] -= v.e[1];
    e[2] -= v.e[2];
    return *this;
}

__host__ __device__ inline vec3 &vec3::operator*=(const FP_T t)
{
    e[0] *= t;
    e[1] *= t;
    e[2] *= t;
    return *this;
}

__host__ __device__ inline vec3 &vec3::operator/=(const FP_T t)
{
    FP_T k = 1.0 / t;

    e[0] *= k;
    e[1] *= k;
    e[2] *= k;
    return *this;
}

__host__ __device__ inline vec3 unit_vector(vec3 v) { return v / v.length(); }

#endif
