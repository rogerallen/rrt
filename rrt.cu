#include "rrt.h"

#include "bvh.h"
#include "camera.h"
#include "hittable_list.h"
#include "material.h"
#include "moving_sphere.h"
#include "sphere.h"
#include "triangle.h"

#include <iostream>

#ifdef COMPILING_FOR_WSL
#define SUPPORTS_CUDA_MEM_PREFETCH_ASYNC 0
#else
#define SUPPORTS_CUDA_MEM_PREFETCH_ASYNC 1
#endif

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
// #define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line)
{
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '"
                  << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__device__ color ray_color(const ray &r, hittable **world, int max_depth, curandState *local_rand_state, bool debug)
{
    // update cur_ray & cur_attenuation in your loop
    ray cur_ray = r;
    color cur_attenuation(1, 1, 1);
    for (int i = 0; i < max_depth; i++) {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, infinity, rec, debug)) {
            // hit something, adjust cur_attenuation
            // and update cur_ray per the diffuse calculation
            ray scattered;
            color attenuation;
            if (rec.mat_ptr->scatter(local_rand_state, cur_ray, rec, attenuation, scattered, debug)) {
                if (debug)
                    printf("DEBUG hit scat=%f %f %f -> %f %f %f\n", scattered.orig.x(), scattered.orig.y(),
                           scattered.orig.z(), scattered.dir.x(), scattered.dir.y(), scattered.dir.z());
                cur_attenuation = cur_attenuation * attenuation;
                cur_ray = scattered;
                if (debug)
                    printf("DEBUG hit c=%f %f %f att=%f %f %f\n", cur_attenuation.x(), cur_attenuation.y(),
                           cur_attenuation.z(), attenuation.x(), attenuation.y(), attenuation.z());
            }
            else {
                return color(0, 0, 0);
            }
        }
        else {
            // hit sky, attenuate the sky color and return
            vec3 unit_direction = unit_vector(cur_ray.direction());
            auto t = 0.5 * (unit_direction.y() + 1.0);
            vec3 c = (1.0 - t) * color(1, 1, 1) + t * color(0.5, 0.7, 1.0);
            c = cur_attenuation * c;
            if (debug) printf("DEBUG sky t=%f c=%f %f %f\n", t, c.x(), c.y(), c.z());
            return c;
        }
    }
    return color(0, 0, 0); // exceeded recursion
}

__global__ void render_init(int image_width, int image_height, curandState *rand_state)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= image_width) || (j >= image_height)) return;
    int pixel_index = j * image_width + i;
    // Each thread gets same seed, a different sequence number, no offset
    curand_init(1984 + pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void
// may be useful
//__launch_bounds__(64, 12) // maxThreadsPerBlock, minBlocksPerMultiprocessor
cuda_render(vec3 *fb, int image_width, int image_height, int samples_per_pixel, camera **cam, hittable **world,
            int max_depth, curandState *d_rand_state)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int pixel_index = j * image_width + i;
    if ((i >= image_width) || (j >= image_height)) return;
    curandState local_rand_state = d_rand_state[pixel_index];
    bool debug = false;
#if 0
    if ((i == 600) && (j == 400)) {
        printf("DEBUG ij %d %d\n", i, j);
        debug = true;
    }
#endif
    //  from C++ Render inner loop
    color pixel_color(0, 0, 0);
    for (int s = 0; s < samples_per_pixel; ++s) {
        auto u = FP_T(i + random_uniform(&local_rand_state)) / (image_width - 1);
        auto v = FP_T(j + random_uniform(&local_rand_state)) / (image_height - 1);
        ray r = (*cam)->get_ray(&local_rand_state, u, v);
        pixel_color += ray_color(r, world, max_depth, &local_rand_state, debug);
    }
    if (debug) {
        printf("DEBUG ij %d %d rgb %f %f %f\n", i, j, pixel_color.x(), pixel_color.y(), pixel_color.z());
    }
    d_rand_state[pixel_index] = local_rand_state;
    fb[pixel_index] = pixel_color;
}

__global__ void create_world(hittable **d_world, camera *d_scene_camera, camera **d_camera, int num_materials,
                             scene_material *d_scene_materials, material **d_materials, int num_spheres,
                             scene_sphere *d_scene_spheres, int num_moving_spheres,
                             scene_moving_sphere *d_scene_moving_spheres, int num_triangles,
                             scene_instance_triangle *d_scene_triangles, FP_T aspect_ratio)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {

        *d_world = new hittable_list();

        // we have to copy-construct since the camera has a DEV-only get_ray function that depends
        // on curand state.
        *d_camera = new camera(*d_scene_camera);

        for (int i = 0; i < num_materials; ++i) {
            scene_material *m = &(d_scene_materials[i]);
            if (m->type == LAMBERTIAN) {
                d_materials[i] = new lambertian(m->mat.lambertian.albedo);
            }
            else if (m->type == METAL) {
                d_materials[i] = new metal(m->mat.metal.albedo, m->mat.metal.fuzz);
            }
            else if (m->type == DIELECTRIC) {
                d_materials[i] = new dielectric(m->mat.dielectric.ref_idx);
            }
        }

        for (int i = 0; i < num_spheres; ++i) {
            scene_sphere *s = &(d_scene_spheres[i]);
            ((hittable_list *)*d_world)->add(new sphere(s->center, s->radius, d_materials[s->material_idx]));
        }
        for (int i = 0; i < num_moving_spheres; ++i) {
            scene_moving_sphere *s = &(d_scene_moving_spheres[i]);
            ((hittable_list *)*d_world)
                ->add(new moving_sphere(s->center0, s->center1, s->time0, s->time1, s->radius,
                                        d_materials[s->material_idx]));
        }
        for (int i = 0; i < num_triangles; ++i) {
            scene_instance_triangle *t = &(d_scene_triangles[i]);
            ((hittable_list *)*d_world)
                ->add(new triangle(t->vertices[0], t->vertices[1], t->vertices[2], d_materials[t->material_idx]));
        }
    }
}
__global__ void free_world(int num_materials, material **d_materials, int num_spheres, hittable **d_world,
                           camera **d_camera)
{
    // ??? who frees these? FIXME  hmm delete[] d_materials?
    for (int i = 0; i < num_materials; i++) {
        delete d_materials[i];
    }
    delete *d_world;
    delete *d_camera;
}

vec3 *Rrt::render(scene *the_scene)
{
    int num_blocks_x = image_width / num_threads_x + 1;
    int num_blocks_y = image_height / num_threads_y + 1;

    int cuda_runtime_version = -1;
    checkCudaErrors(cudaRuntimeGetVersion(&cuda_runtime_version));

    std::cerr << "CUDA Runtime Version " << cuda_runtime_version << "\n";
    std::cerr << "Rendering a " << image_width << "x" << image_height << " image with " << samples_per_pixel
              << " samples per pixel ";
    std::cerr << "in " << num_blocks_x << "x" << num_blocks_y << " = " << num_blocks_x * num_blocks_y << " blocks of "
              << num_threads_x << "x" << num_threads_y << " threads each.\n";

    int num_pixels = image_width * image_height;
    size_t fb_size = num_pixels * sizeof(vec3);

    // allocate FB
    vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    // allocate random state
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels * samples_per_pixel * sizeof(curandState)));

    // make our world of hittables & the camera
    // create & populate scene data that create_world with use to make the scene.
    camera *d_scene_camera;
    checkCudaErrors(cudaMallocManaged((void **)&d_scene_camera, sizeof(camera)));
    *d_scene_camera = *(the_scene->cam);

    scene_material *d_scene_materials;
    int num_materials = the_scene->materials.size();
    checkCudaErrors(cudaMallocManaged((void **)&d_scene_materials, num_materials * sizeof(scene_material)));
    for (int i = 0; i < num_materials; ++i) {
        d_scene_materials[i] = *(the_scene->materials[i]);
    }

    scene_sphere *d_scene_spheres;
    int num_spheres = the_scene->spheres.size();
    checkCudaErrors(cudaMallocManaged((void **)&d_scene_spheres, num_spheres * sizeof(scene_sphere)));
    for (int i = 0; i < num_spheres; ++i) {
        d_scene_spheres[i] = *(the_scene->spheres[i]);
    }

    scene_moving_sphere *d_scene_moving_spheres;
    int num_moving_spheres = the_scene->moving_spheres.size();
    checkCudaErrors(
        cudaMallocManaged((void **)&d_scene_moving_spheres, num_moving_spheres * sizeof(scene_moving_sphere)));
    for (int i = 0; i < num_moving_spheres; ++i) {
        d_scene_moving_spheres[i] = *(the_scene->moving_spheres[i]);
    }

    scene_instance_triangle *d_instance_triangles;
    int num_instance_triangles = the_scene->num_triangles();
    checkCudaErrors(
        cudaMallocManaged((void **)&d_instance_triangles, num_instance_triangles * sizeof(scene_instance_triangle)));
    the_scene->fill_instance_triangles(d_instance_triangles);

    // now create the data that will contain the world.
    // create_world populates these vars
    camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));

    material **d_materials;
    checkCudaErrors(cudaMalloc((void **)&d_materials, num_materials * sizeof(material *)));

    hittable **d_world;
    checkCudaErrors(cudaMallocManaged((void **)&d_world, sizeof(hittable *)));

    int num_hittables = num_instance_triangles + num_spheres;
    std::cerr << "num_hittables = " << num_hittables << "\n";

    create_world<<<1, 1>>>(d_world, d_scene_camera, d_camera, num_materials, d_scene_materials, d_materials,
                           num_spheres, d_scene_spheres, num_moving_spheres, d_scene_moving_spheres,
                           num_instance_triangles, d_instance_triangles, aspect_ratio);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

#if SUPPORTS_CUDA_MEM_PREFETCH_ASYNC == 1
    // Prefetch the FB to the GPU
    int device = -1;
    checkCudaErrors(cudaGetDevice(&device));
    std::cerr << "CUDA Device: " << device << std::endl;
    checkCudaErrors(cudaMemPrefetchAsync(fb, fb_size, device, NULL));
    checkCudaErrors(cudaGetLastError());
#endif

    clock_t start, stop;
    start = clock();
    // Render our buffer
    dim3 blocks(num_blocks_x, num_blocks_y);
    dim3 threads(num_threads_x, num_threads_y);
    render_init<<<blocks, threads>>>(image_width, image_height, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    cuda_render<<<blocks, threads>>>(fb, image_width, image_height, samples_per_pixel, d_camera, d_world, max_depth,
                                     d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    std::cerr << "stats:" << cuda_runtime_version << "," << image_width << "," << image_height << ","
              << samples_per_pixel << "," << (num_blocks_x * num_blocks_y) << "," << num_threads_x << ","
              << num_threads_y << "," << timer_seconds << "\n";

#if SUPPORTS_CUDA_MEM_PREFETCH_ASYNC == 1
    // Prefetch the FB back to the CPU
    checkCudaErrors(cudaMemPrefetchAsync(fb, fb_size, cudaCpuDeviceId, NULL));
    checkCudaErrors(cudaGetLastError());
#endif

    // clean up
    free_world<<<1, 1>>>(num_materials, d_materials, num_spheres, d_world, d_camera);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    // checkCudaErrors(cudaFree(d_scene_camera));
    checkCudaErrors(cudaFree(d_scene_materials));
    checkCudaErrors(cudaFree(d_scene_spheres));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_rand_state));

    return fb;
}

Rrt::~Rrt()
{
    if (fb != nullptr) {
        checkCudaErrors(cudaFree(fb));
    }
    cudaDeviceReset();
}