#ifndef COLOR_H
#define COLOR_H

#include "vec3.h"

#include <iostream>

void convert_color(color pixel_color, int samples_per_pixel, int *red, int *grn, int *blu)
{
    auto r = pixel_color.x();
    auto g = pixel_color.y();
    auto b = pixel_color.z();

    // Divide the color by the number of samples and gamma-correct for gamma=2.0.
    auto scale = (FP_T)1.0 / samples_per_pixel;
    r = SQRT(scale * r);
    g = SQRT(scale * g);
    b = SQRT(scale * b);

    *red = static_cast<int>(256 * clamp(r, (FP_T)0.0, (FP_T)0.999));
    *grn = static_cast<int>(256 * clamp(g, (FP_T)0.0, (FP_T)0.999));
    *blu = static_cast<int>(256 * clamp(b, (FP_T)0.0, (FP_T)0.999));
}

void write_color(std::ostream &out, color pixel_color, int samples_per_pixel)
{
    int red, grn, blu;
    convert_color(pixel_color, samples_per_pixel, &red, &grn, &blu);

    // Write the translated [0,255] value of each color component.
    out << red << ' ' << grn << ' ' << blu << '\n';
}

#endif