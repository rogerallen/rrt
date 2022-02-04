#ifndef SCENEH
#define SCENEH

#include "vec3.h"
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

struct scene_camera {
    vec3 lookfrom;
    vec3 lookat;
    vec3 vup;
    double vfov;
    double aperture;
    double focus;
};

struct scene_sphere {
    vec3 center;
    double radius;
    std::string material_name;
};

enum scene_material_t { LAMBERTIAN, METAL, DIELECTRIC };

union scene_material {

    scene_material(){};
    ~scene_material(){};

    scene_material_t type;

    struct {
        vec3 albedo;
    } lambertian;

    struct {
        vec3 albedo;
        double fuzz;
    } metal;

    struct {
        double ref_idx;
    } dielectric;
};

class scene {
  public:
    scene(char *filename)
    {
        std::string line;
        std::ifstream fl(filename);
        if (!fl.good()) {
            std::cerr << "ERROR: problem with opening file: " << filename
                      << "\n";
            exit(2);
        }
        while (std::getline(fl, line)) {
            if (line.find("camera") == 0) {
                std::istringstream iss(line);
                std::string camera_str;
                std::string lfx_str, lfy_str, lfz_str;
                std::string lax_str, lay_str, laz_str;
                std::string vux_str, vuy_str, vuz_str;
                std::string vfo_str, ap_str, foc_str;
                iss >> camera_str;
                iss >> lfx_str;  // FIXME try multiple per line
                iss >> lfy_str;
                iss >> lfz_str;
                iss >> lax_str;
                iss >> lay_str;
                iss >> laz_str;
                iss >> vux_str;
                iss >> vuy_str;
                iss >> vuz_str;
                iss >> vfo_str;
                iss >> ap_str;
                iss >> foc_str;
                camera.lookfrom = vec3(std::stod(lfx_str), std::stod(lfy_str),
                                       std::stod(lfz_str));
                camera.lookat = vec3(std::stod(lax_str), std::stod(lay_str),
                                     std::stod(laz_str));
                camera.vup = vec3(std::stod(vux_str), std::stod(vuy_str),
                                  std::stod(vuz_str));
                camera.vfov = std::stod(vfo_str);
                camera.aperture = std::stod(ap_str);
                camera.focus = std::stod(foc_str);
            }
            else if (line.find("material") == 0) {
                scene_material *new_material = new scene_material;
                std::istringstream iss(line);
                std::string material_str, name_str, type_str;
                iss >> material_str;
                iss >> name_str;
                iss >> type_str;
                if (type_str == "lambertian") {
                    std::string r_str, g_str, b_str;
                    iss >> r_str;
                    iss >> g_str;
                    iss >> b_str;
                    new_material->type = LAMBERTIAN;
                    new_material->lambertian.albedo = vec3(
                        std::stod(r_str), std::stod(g_str), std::stod(b_str));
                }
                else if (type_str == "metal") {
                    // FIXME
                }
                else if (type_str == "dielectric") {
                    // FIXME
                }
                else {
                    std::cerr << "ERROR: unknown material type: " << type_str
                              << std::endl;
                    exit(3);
                }
                materials.insert(std::pair<std::string, scene_material *>(
                    name_str, new_material));
            }
            else if (line.find("sphere") == 0) {
                std::cout << "!sphere\n";
                    // FIXME
            }
        }
        fl.close();
    }

    scene_camera camera;
    std::map<std::string, scene_material *> materials;
    std::vector<scene_sphere> spheres;
};

#endif
