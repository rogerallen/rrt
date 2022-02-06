#include "camera.h"
#include "hitable_list.h"
#include "material.h"
#include "ray.h"
#include "scene.h"
#include "sphere.h"
#include "vec3.h"
#include <assert.h>
#include <curand_kernel.h>
#include <float.h>
#include <iostream>
#include <time.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#ifdef COMPILING_FOR_WSL
#define SUPPORTS_CUDA_MEM_PREFETCH_ASYNC 0
#else
#define SUPPORTS_CUDA_MEM_PREFETCH_ASYNC 1
#endif

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

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

// Matching the C++ code would recurse enough into color() calls that
// it was blowing up the stack, so we have to turn this into a
// limited-depth loop instead.  Later code in the book limits to a max
// depth of 50, so we adapt this a few chapters early on the GPU.
__device__ vec3 color(const ray &r, hitable **world, curandState *local_rand_state)
{
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);
    for (int i = 0; i < 50; i++) {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            vec3 attenuation;
            if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else {
                return vec3(0.0, 0.0, 0.0);
            }
        }
        else {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f * (unit_direction.y() + 1.0f);
            vec3 c = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    return vec3(0.0, 0.0, 0.0); // exceeded recursion
}

__global__ void rand_init(curandState *rand_state)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
    }
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    // Original: Each thread gets same seed, a different sequence number, no
    // offset curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
    // BUGFIX, see Issue#2: Each thread gets different seed, same sequence for
    // performance improvement of about 2x!
    curand_init(1984 + pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void __launch_bounds__(64, 12) // maxThreadsPerBlock, minBlocksPerMultiprocessor
    render(vec3 *fb, int max_x, int max_y, int num_samples, camera **cam, hitable **world, curandState *rand_state)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    vec3 col(0, 0, 0);
    for (int s = 0; s < num_samples; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += color(r, world, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(num_samples);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel_index] = col;
}

#define RND (curand_uniform(&local_rand_state))

#if 0
__global__ void create_world(hitable **d_hitables, hitable **d_world,
                             camera **d_camera, int image_width,
                             int image_height, curandState *rand_state)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        d_hitables[0] = new sphere(vec3(0, -1000.0, -1), 1000,
                               new lambertian(vec3(0.5, 0.5, 0.5)));
        int i = 1;
        for (int a = -11; a < 11; a++) {
            for (int b = -11; b < 11; b++) {
                float choose_mat = RND;
                vec3 center(a + RND, 0.2, b + RND);
                if (choose_mat < 0.8f) {
                    d_hitables[i++] = new sphere(
                        center, 0.2,
                        new lambertian(vec3(RND * RND, RND * RND, RND * RND)));
                }
                else if (choose_mat < 0.95f) {
                    d_hitables[i++] = new sphere(
                        center, 0.2,
                        new metal(vec3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND),
                                       0.5f * (1.0f + RND)),
                                  0.5f * RND));
                }
                else {
                    d_hitables[i++] = new sphere(center, 0.2, new dielectric(1.5));
                }
            }
        }
        d_hitables[i++] = new sphere(vec3(0, 1, 0), 1.0, new dielectric(1.5));
        d_hitables[i++] = new sphere(vec3(-4, 1, 0), 1.0,
                                 new lambertian(vec3(0.4, 0.2, 0.1)));
        d_hitables[i++] =
            new sphere(vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
        *rand_state = local_rand_state;
        *d_world = new hitable_list(d_hitables, 22 * 22 + 1 + 3);

        for(int j = 0; j < i; j++) {
            d_hitables[j]->print(j);
        }

        vec3 lookfrom(13, 2, 3);
        vec3 lookat(0, 0, 0);
        float dist_to_focus = 10.0;
        (lookfrom - lookat).length();
        float aperture = 0.1;
        *d_camera = new camera(lookfrom, lookat, vec3(0, 1, 0), 30.0,
                               float(image_width) / float(image_height),
                               aperture, dist_to_focus);
    }
}

__global__ void free_world(hitable **d_hitables, hitable **d_world,
                           camera **d_camera)
{
    for (int i = 0; i < 22 * 22 + 1 + 3; i++) {
        delete ((sphere *)d_hitables[i])->mat_ptr;
        delete d_hitables[i];
    }
    delete *d_world;
    delete *d_camera;
}
#else
__global__ void create_world(hitable **d_world, scene_camera *d_scene_camera, camera **d_camera, int num_materials,
                             scene_material *d_scene_materials, material **d_materials, int num_spheres,
                             scene_sphere *d_scene_spheres, hitable **d_hitables, int image_width, int image_height)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {

        *d_camera = new camera(d_scene_camera->lookfrom, d_scene_camera->lookat, d_scene_camera->vup,
                               (float)d_scene_camera->vfov, float(image_width) / float(image_height),
                               (float)d_scene_camera->aperture, (float)d_scene_camera->focus);

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
            d_hitables[i] = new sphere(s->center, s->radius, d_materials[s->material_index]);
        }

        *d_world = new hitable_list(d_hitables, num_spheres);
    }
}
__global__ void free_world(int num_materials, material **d_materials, int num_spheres, hitable **d_hitables,
                           hitable **d_world, camera **d_camera)
{
    for (int i = 0; i < num_materials; i++) {
        delete d_materials[i];
    }
    for (int i = 0; i < num_spheres; i++) {
        delete d_hitables[i];
    }
    delete *d_world;
    delete *d_camera;
}
#endif

void usage(char *argv)
{
    std::cerr << "Unexpected argument: " << argv << "\n\n";
    std::cerr << "Usage: rrt [options]\n";
    std::cerr << "  -i file.txt         : input scene file\n";
    std::cerr << "  -o file.png         : output raytraced PNG image (default is PPM to stdout)\n";
    std::cerr << "  -w <width>          : output image width. (default = 1200)\n";
    std::cerr << "  -h <height>         : output image height. (800)\n";
    std::cerr << "  -s <samples>        : number of samples per pixel. (10)\n";
    std::cerr << "  -tx <num_threads_x> : number of threads per block in x. (8)\n";
    std::cerr << "  -ty <num_threads_y> : number of threads per block in y. (8)\n";
    std::exit(1);
}

int main(int argc, char *argv[])
{

    int image_width = 1200;
    int image_height = 800;
    int num_samples = 10;
    int num_threads_x = 8;
    int num_threads_y = 8;
    scene *the_scene = nullptr;
    char *png_filename = nullptr;

    for (int i = 1; i < argc; ++i) {
        if (argv[i][0] == '-') {
            if (argv[i][1] == 'w') {
                image_width = atoi(argv[++i]);
            }
            else if (argv[i][1] == 'h') {
                image_height = atoi(argv[++i]);
            }
            else if (argv[i][1] == 's') {
                num_samples = atoi(argv[++i]);
            }
            else if (argv[i][1] == 't') {
                if (argv[i][2] == 'x') {
                    num_threads_x = atoi(argv[++i]);
                }
                else if (argv[i][2] == 'y') {
                    num_threads_y = atoi(argv[++i]);
                }
                else {
                    usage(argv[i]);
                }
            }
            else if (argv[i][1] == 'i') {
                the_scene = new scene(argv[++i]);
            }
            else if (argv[i][1] == 'o') {
                png_filename = argv[++i];
            }
            else {
                usage(argv[i]);
            }
        }
        else {
            usage(argv[i]);
        }
    }
    int num_blocks_x = image_width / num_threads_x + 1;
    int num_blocks_y = image_height / num_threads_y + 1;

    int cuda_runtime_version = -1;
    checkCudaErrors(cudaRuntimeGetVersion(&cuda_runtime_version));

    std::cerr << "CUDA Runtime Version " << cuda_runtime_version << "\n";
    std::cerr << "Rendering a " << image_width << "x" << image_height << " image with " << num_samples
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
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels * sizeof(curandState)));
    curandState *d_rand_state2;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1 * sizeof(curandState)));

    // we need that 2nd random state to be initialized for the world creation
    rand_init<<<1, 1>>>(d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // make our world of hitables & the camera
#if 0
    hitable **d_hitables;
    int num_hitables = 22 * 22 + 1 + 3;
    checkCudaErrors(
        cudaMalloc((void **)&d_hitables, num_hitables * sizeof(hitable *)));
    hitable **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable *)));
    camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));
    create_world<<<1, 1>>>(d_hitables, d_world, d_camera, image_width, image_height,
                           d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
#else
    // create & populate scene data that create_world with use to make the
    // scene.
    scene_camera *d_scene_camera;
    checkCudaErrors(cudaMallocManaged((void **)&d_scene_camera, sizeof(scene_camera)));
    *d_scene_camera = the_scene->camera;

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

    // now create the data that will contain the world.  create_world populates
    // these
    camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));
    material **d_materials;
    checkCudaErrors(cudaMalloc((void **)&d_materials, num_materials * sizeof(material *)));
    hitable **d_hitables;
    checkCudaErrors(cudaMalloc((void **)&d_hitables, num_spheres * sizeof(hitable *)));
    hitable **d_world;
    checkCudaErrors(cudaMallocManaged((void **)&d_world, sizeof(hitable *)));

    create_world<<<1, 1>>>(d_world, d_scene_camera, d_camera, num_materials, d_scene_materials, d_materials,
                           num_spheres, d_scene_spheres, d_hitables, image_width, image_height);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
#endif

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
    // note that render has __launch_bounds__ of maxThreadsPerBlock=64, minBlocksPerMultiprocessor=12
    render<<<blocks, threads>>>(fb, image_width, image_height, num_samples, d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    std::cerr << "stats:" << cuda_runtime_version << "," << image_width << "," << image_height << "," << num_samples
              << "," << num_threads_x << "," << num_threads_y << "," << timer_seconds << "\n";

#if SUPPORTS_CUDA_MEM_PREFETCH_ASYNC == 1
    // Prefetch the FB back to the CPU
    checkCudaErrors(cudaMemPrefetchAsync(fb, fb_size, cudaCpuDeviceId, NULL));
    checkCudaErrors(cudaGetLastError());
#endif

    // Output FB as Image
    if (png_filename == nullptr) {
        // default to PPM to stdout
        std::cout << "P3\n" << image_width << " " << image_height << "\n255\n";
        for (int j = image_height - 1; j >= 0; j--) {
            for (int i = 0; i < image_width; i++) {
                size_t pixel_index = j * image_width + i;
                int ir = int(255.99 * fb[pixel_index].r());
                int ig = int(255.99 * fb[pixel_index].g());
                int ib = int(255.99 * fb[pixel_index].b());
                std::cout << ir << " " << ig << " " << ib << "\n";
            }
        }
    }
    else {
        // write fb to png_filename
        uint8_t *cpu_fb = new uint8_t[image_width * image_height * 3];
        for (int j = image_height - 1, k = 0; j >= 0; j--, k++) {
            for (int i = 0; i < image_width; i++) {
                size_t fb_idx = j * image_width + i;
                size_t cpu_fb_idx = k * image_width * 3 + i * 3;
                cpu_fb[cpu_fb_idx + 0] = uint8_t(255.99 * fb[fb_idx].r());
                cpu_fb[cpu_fb_idx + 1] = uint8_t(255.99 * fb[fb_idx].g());
                cpu_fb[cpu_fb_idx + 2] = uint8_t(255.99 * fb[fb_idx].b());
            }
        }
        stbi_write_png(png_filename, image_width, image_height, 3, (const void *)cpu_fb,
                       image_width * 3 * sizeof(uint8_t));
        delete[] cpu_fb;
    }

    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
#if 0
    free_world<<<1, 1>>>(d_hitables, d_world, d_camera);
#else
    free_world<<<1, 1>>>(num_materials, d_materials, num_spheres, d_hitables, d_world, d_camera);
#endif
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_scene_camera));
    checkCudaErrors(cudaFree(d_scene_materials));
    checkCudaErrors(cudaFree(d_scene_spheres));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_rand_state2));
    checkCudaErrors(cudaFree(fb));

    if (the_scene) {
        delete the_scene;
    }

    cudaDeviceReset();
}
