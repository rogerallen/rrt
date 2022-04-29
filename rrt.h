#ifndef RRT_H
#define RRT_H

#include "rtweekend.h"

#include "scene.h"

class Rrt {
  public:
    Rrt(int image_width, int image_height, int samples_per_pixel, int max_depth)
        : image_width(image_width), image_height(image_height), samples_per_pixel(samples_per_pixel),
          max_depth(max_depth)
    {
        aspect_ratio = 1.0 * image_width / image_height;
        fb = nullptr;
    }
    ~Rrt()
    {
        if (fb != nullptr) {
            delete[] fb;
        }
    }

    vec3 *render(scene *the_scene);

  private:
    int image_width;
    int image_height;
    int samples_per_pixel;
    int max_depth;
    FP_T aspect_ratio;
    vec3 *fb;
};

#endif