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
    int material_index;
};

enum scene_material_t { LAMBERTIAN = 0, METAL, DIELECTRIC };

struct scene_material {

    scene_material(){};
    ~scene_material(){};

    scene_material_t type;

    union U {
        U(){};
        ~U(){};
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
    } mat;
};

class scene {
  public:
    scene(char *filename)
    {
        bool got_camera = false;
        std::string line;
        std::ifstream fl(filename);
        if (!fl.good()) {
            std::cerr << "ERROR: problem with opening file: " << filename << "\n";
            exit(2);
        }
        while (std::getline(fl, line)) {
            if (line.find("camera") == 0) {
                got_camera = true;
                std::istringstream iss(line);
                std::string camera_str;
                std::string lfx_str, lfy_str, lfz_str;
                std::string lax_str, lay_str, laz_str;
                std::string vux_str, vuy_str, vuz_str;
                std::string vfo_str, ap_str, foc_str;
                iss >> camera_str;
                iss >> lfx_str; // FIXME try multiple per line
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
                camera.lookfrom = vec3(std::stod(lfx_str), std::stod(lfy_str), std::stod(lfz_str));
                camera.lookat = vec3(std::stod(lax_str), std::stod(lay_str), std::stod(laz_str));
                camera.vup = vec3(std::stod(vux_str), std::stod(vuy_str), std::stod(vuz_str));
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
                    new_material->mat.lambertian.albedo = vec3(std::stod(r_str), std::stod(g_str), std::stod(b_str));
                }
                else if (type_str == "metal") {
                    std::string r_str, g_str, b_str, f_str;
                    iss >> r_str;
                    iss >> g_str;
                    iss >> b_str;
                    iss >> f_str;
                    new_material->type = METAL;
                    new_material->mat.metal.albedo = vec3(std::stod(r_str), std::stod(g_str), std::stod(b_str));
                    new_material->mat.metal.fuzz = std::stod(f_str);
                }
                else if (type_str == "dielectric") {
                    std::string r_str;
                    iss >> r_str;
                    new_material->type = DIELECTRIC;
                    new_material->mat.dielectric.ref_idx = std::stod(r_str);
                }
                else {
                    std::cerr << "ERROR: unknown material type: " << type_str << std::endl;
                    exit(3);
                }
                int next_index = materials.size();
                materials.push_back(new_material);
                material_names_to_index.insert(std::pair<std::string, int>(name_str, next_index));
            }
            else if (line.find("sphere") == 0) {
                std::istringstream iss(line);
                std::string sphere_str;
                std::string cx_str, cy_str, cz_str;
                std::string r_str;
                std::string mat_str;
                iss >> sphere_str;
                iss >> cx_str;
                iss >> cy_str;
                iss >> cz_str;
                iss >> r_str;
                iss >> mat_str;

                scene_sphere *new_sphere = new scene_sphere;

                new_sphere->center = vec3(std::stod(cx_str), std::stod(cy_str), std::stod(cz_str));
                new_sphere->radius = std::stod(r_str);
                new_sphere->material_index = material_names_to_index[mat_str];
                spheres.push_back(new_sphere);
            }
        }
        fl.close();
        // check scene
        if (!got_camera) {
            std::cerr << "ERROR: Scene did not have a camera." << std::endl;
            std::exit(4);
        }
        if (materials.size() == 0) {
            std::cerr << "ERROR: Scene did not have any materials." << std::endl;
            std::exit(4);
        }
        if (spheres.size() == 0) {
            std::cerr << "ERROR: Scene did not have any spheres." << std::endl;
            std::exit(4);
        }
        std::cerr << "read scene file: " << filename << "\n";
        std::cerr << "material count:  " << materials.size() << "\n";
        std::cerr << "sphere count:    " << spheres.size() << std::endl;
    }
    // FIXME -- add a destructor

    scene_camera camera;
    std::map<std::string, int> material_names_to_index;
    std::vector<scene_material *> materials;
    std::vector<scene_sphere *> spheres;
};

#endif
