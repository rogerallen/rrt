#ifndef RRT_H
#define RRT_H

#include "rtweekend.h"

#include "scene.h"

#ifdef USE_CUDA
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line);
#endif

class Rrt {
  public:
    Rrt(int image_width, int image_height, int samples_per_pixel, int max_depth
#ifdef USE_CUDA
        ,
        int threads_x, int threads_y
#else
        ,
        bool use_bvh
#endif
        )
        : image_width(image_width), image_height(image_height), samples_per_pixel(samples_per_pixel),
          max_depth(max_depth)
#ifdef USE_CUDA
          ,
          num_threads_x(threads_x), num_threads_y(threads_y)
#else
          ,
          bvh(use_bvh)
#endif
    {
        aspect_ratio = 1.0 * image_width / image_height;
        fb = nullptr;
    }
    ~Rrt();

    vec3 *render(scene *the_scene);

  private:
    int image_width;
    int image_height;
    int samples_per_pixel;
    int max_depth;
#ifdef USE_CUDA
    int num_threads_x;
    int num_threads_y;
#else
    bool bvh;
#endif
    FP_T aspect_ratio;
    vec3 *fb;
};

#endif