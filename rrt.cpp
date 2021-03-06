#include "rrt.h"

#include "bvh.h"
#include "camera.h"
#include "hittable_list.h"
#include "material.h"
#include "moving_sphere.h"
#include "sphere.h"
#include "triangle.h"

#include <ctime>
#include <iomanip>
#include <iostream>
#include <limits.h>
#ifdef __linux__
#include <unistd.h>
#endif
#include <vector>

#include <omp.h>

#define Q(x) #x
#define QUOTE(x) Q(x)

color ray_color(const ray &r, const hittable *world, int depth, bool debug)
{
    hit_record rec;

    // If we've exceeded the ray bounce limit, no more light is gathered.
    if (depth <= 0) return color(0, 0, 0);

    if (world->hit(r, (FP_T)0.001, (FP_T)infinity, rec, debug)) {
        ray scattered;
        color attenuation;
        if (rec.mat_ptr->scatter(r, rec, attenuation, scattered, debug)) {
            if (debug)
                printf("DEBUG hit scat=%f %f %f -> %f %f %f\n", scattered.orig.x(), scattered.orig.y(),
                       scattered.orig.z(), scattered.dir.x(), scattered.dir.y(), scattered.dir.z());
            auto c = ray_color(scattered, world, depth - 1, debug);
            if (debug)
                printf("DEBUG hit c=%f %f %f att=%f %f %f\n", c.x(), c.y(), c.z(), attenuation.x(), attenuation.y(),
                       attenuation.z());
            return attenuation * c;
        }
        return color(0, 0, 0);
    }
    vec3 unit_direction = unit_vector(r.direction());
    auto t = (FP_T)0.5 * (unit_direction.y() + (FP_T)1.0);
    auto c = ((FP_T)1.0 - t) * color((FP_T)1.0, (FP_T)1.0, (FP_T)1.0) + t * color((FP_T)0.5, (FP_T)0.7, (FP_T)1.0);
    if (debug) printf("DEBUG sky c=%f %f %f\n", c.x(), c.y(), c.z());
    return c;
}

hittable *create_world(scene *the_scene, bool bvh)
{
    auto world_list = new hittable_list();

    std::vector<material_ptr_t> materials;
    for (auto m : the_scene->materials) {
        if (m->type == LAMBERTIAN) {
            materials.push_back(make_shared<lambertian>(m->mat.lambertian.albedo));
        }
        else if (m->type == METAL) {
            materials.push_back(make_shared<metal>(m->mat.metal.albedo, m->mat.metal.fuzz));
        }
        else if (m->type == DIELECTRIC) {
            materials.push_back(make_shared<dielectric>(m->mat.dielectric.ref_idx));
        }
    }

    for (auto s : the_scene->spheres) {
        world_list->add(make_shared<sphere>(s->center, s->radius, materials[s->material_idx]));
    }
    for (auto s : the_scene->moving_spheres) {
        world_list->add(make_shared<moving_sphere>(s->center0, s->center1, s->time0, s->time1, s->radius,
                                                   materials[s->material_idx]));
    }
    scene_instance_triangle *instance_triangles = new scene_instance_triangle[the_scene->num_triangles()];
    the_scene->fill_instance_triangles(instance_triangles);
    for (int i = 0; i < the_scene->num_triangles(); ++i) {
        scene_instance_triangle t = instance_triangles[i];
        world_list->add(make_shared<triangle>(t.vertices[0], t.vertices[1], t.vertices[2], materials[t.material_idx]));
    }

    int num_hittables = (int)(the_scene->num_triangles() + the_scene->spheres.size() + the_scene->moving_spheres.size());
    std::cerr << "num_hittables = " << num_hittables << "\n";

    hittable *world = world_list;

    if (bvh) {
        auto world_bvh = new bvh_node(*world_list, the_scene->cam->t0(), the_scene->cam->t1());
        if (0) world_bvh->print(0);
        world = world_bvh;
    }

    return world;
}

vec3 *Rrt::render(scene *the_scene)
{

    std::time_t render_time = std::time(nullptr);
    std::tm render_tm = *std::localtime(&render_time);

    std::cerr << "Rendering a " << image_width << "x" << image_height << " image with " << samples_per_pixel
              << " samples per pixel\n";

    // World
    auto world = create_world(the_scene, bvh);

    // Camera
    camera cam(*(the_scene->cam));

    fb = new vec3[image_width * image_height];

#ifndef _OPENMP
    clock_t start, stop;
    start = clock();
#else
    double start, stop;
    start = omp_get_wtime();
#endif
    // Render
#pragma omp parallel for
    for (int j = image_height - 1; j >= 0; --j) {
#ifndef _OPENMP
        if (j % 10 == 0) std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
#endif
#pragma omp parallel for
        for (int i = 0; i < image_width; ++i) {
            bool debug = false;
            color pixel_color(0, 0, 0);
#if 0
            if ((i == 100) && (j == 75)) {
                printf("DEBUG ij %d %d\n", i, j);
                debug = true;
            }
#endif
            for (int s = 0; s < samples_per_pixel; ++s) {
                auto u = (i + random_uniform()) / (image_width - 1);
                auto v = (j + random_uniform()) / (image_height - 1);
                ray r = cam.get_ray(u, v);
                pixel_color += ray_color(r, world, max_depth, debug);
            }
            if (debug) {
                printf("DEBUG ij %d %d rgb %f %f %f\n", i, j, pixel_color.x(), pixel_color.y(), pixel_color.z());
            }
            fb[j * image_width + i] = pixel_color;
        }
    }
#ifndef _OPENMP
    std::cerr << "\n";
#endif

#ifndef _OPENMP
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
#else
    stop = omp_get_wtime();
    double timer_seconds = stop - start;
#endif
    std::cerr << "took " << timer_seconds << " seconds.\n";

#ifndef _OPENMP
    std::string cpu_version = "SingleThread";
    std::string num_threads = "1";
#else
    std::string cpu_version = "OpenMP";
    std::string num_threads = std::to_string(omp_get_max_threads());
#endif

#ifdef __linux__
    char hostname[HOST_NAME_MAX];
    gethostname(hostname, HOST_NAME_MAX);
#endif
#ifdef WIN32
    char hostname[] = "n/a";
#endif

    std::cerr << "stats," << std::put_time(&render_tm, "%c %Z,") << std::string(hostname) << "," << cpu_version << ","
              << QUOTE(FP_T) << "," << image_width << "," << image_height << "," << samples_per_pixel << ","
              << num_threads << ","
              << "n/a,n/a," << timer_seconds << "\n";

    return fb;
}

Rrt::~Rrt()
{
    if (fb != nullptr) {
        delete[] fb;
    }
}