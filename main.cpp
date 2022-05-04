// REQUIRED:
// -DFP_T=float or double
// -DUSE_CUDA (or not)

#include "rrt.h"

#include "color.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#ifdef USE_CUDA
void query_cuda_info()
{
    int count;
    checkCudaErrors(cudaGetDeviceCount(&count));
    for (int i = 0; i < count; i++) {
        cudaDeviceProp prop;
        checkCudaErrors(cudaGetDeviceProperties(&prop, i));
        std::cout << "cudaGetDeviceProperties #" << i << "\n";
        std::cout << "  name                        " << prop.name << "\n";
        std::cout << "  major.minor                 " << prop.major << "." << prop.minor << "\n";
        std::cout << "  multiProcessorCount         " << prop.multiProcessorCount << "\n";
        std::cout << "  sharedMemPerBlock           " << prop.sharedMemPerBlock << "\n";
        std::cout << "  maxThreadsPerBlock          " << prop.maxThreadsPerBlock << "\n";
        std::cout << "  maxThreadsPerMultiProcessor " << prop.maxThreadsPerMultiProcessor << "\n";
        std::cout << "  unifiedAddressing           " << prop.unifiedAddressing << "\n";
        std::cout << "  l2CacheSize                 " << prop.l2CacheSize << "\n";
    }
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
    std::cerr << "  -d <max_depth>      : may ray recursion depth. (50)\n";
    std::cerr << "  -b                  : disable bvh acceleration (enabled).\n";
#ifdef USE_CUDA
    std::cerr << "  -tx <num_threads_x> : number of threads per block in x. (8)\n";
    std::cerr << "  -ty <num_threads_y> : number of threads per block in y. (8)\n";
    std::cerr << "  -q                  : query devices & cuda info\n";
    std::cerr << "  -D <device number>  : use this cuda device (0)\n";
#endif
    std::exit(1);
}

int main(int argc, char *argv[])
{

    int image_width = 1200;
    int image_height = 800;
    int num_samples = 10;
#ifdef USE_CUDA
    int num_threads_x = 8;
    int num_threads_y = 8;
#endif
    std::string the_scene_filename;
    scene *the_scene = nullptr;
    char *png_filename = nullptr;
    int max_depth = 50;
    bool use_bvh = true;

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
            else if (argv[i][1] == 'd') {
                max_depth = atoi(argv[++i]);
            }
#ifdef USE_CUDA
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
#endif
            else if (argv[i][1] == 'i') {
                the_scene_filename = argv[++i];
            }
            else if (argv[i][1] == 'o') {
                png_filename = argv[++i];
            }
#ifdef USE_CUDA
            else if (argv[i][1] == 'q') {
                query_cuda_info();
            }
            else if (argv[i][1] == 'D') {
                int device = atoi(argv[++i]);
                checkCudaErrors(cudaSetDevice(device));
            }
#endif
            else if (argv[i][1] == 'b') {
                use_bvh = false;
            }
            else {
                usage(argv[i]);
            }
        }
        else {
            usage(argv[i]);
        }
    }

    // Create Scene
    if (the_scene_filename != "") {
        the_scene = new scene(the_scene_filename.c_str(), image_width, image_height);
    }
    else {
        std::cerr << "ERROR: no scene loaded." << std::endl;
        std::exit(1);
    }

    // Render Scene to Framebuffer
    Rrt rrt = Rrt(image_width, image_height, num_samples, max_depth
#ifdef USE_CUDA
                  ,
                  num_threads_x, num_threads_y
#else
                  ,
                  use_bvh
#endif
    );
    vec3 *fb = rrt.render(the_scene);

    // Output Framebuffer as Image
    if (png_filename == nullptr) {
        // PPM to stdout
        std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";
        for (int j = image_height - 1; j >= 0; --j) {
            for (int i = 0; i < image_width; ++i) {
                size_t pixel_index = j * image_width + i;
                write_color(std::cout, fb[pixel_index], num_samples);
            }
        }
    }
    else {
        // PNG to png_filename
        uint8_t *cpu_fb = new uint8_t[image_width * image_height * 3];
        for (int j = image_height - 1, k = 0; j >= 0; j--, k++) {
            for (int i = 0; i < image_width; i++) {
                size_t fb_idx = j * image_width + i;
                size_t cpu_fb_idx = k * image_width * 3 + i * 3;
                int red, grn, blu;
                convert_color(fb[fb_idx], num_samples, &red, &grn, &blu);
                cpu_fb[cpu_fb_idx + 0] = uint8_t(red);
                cpu_fb[cpu_fb_idx + 1] = uint8_t(grn);
                cpu_fb[cpu_fb_idx + 2] = uint8_t(blu);
            }
        }
        stbi_write_png(png_filename, image_width, image_height, 3, (const void *)cpu_fb,
                       image_width * 3 * sizeof(uint8_t));
        delete[] cpu_fb;
    }

    // Cleanup
    if (the_scene) {
        delete the_scene;
    }
}
