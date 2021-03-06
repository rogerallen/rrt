#ifndef SCENEH
#define SCENEH

#include "vec3.h"
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "camera.h"

//   2 - 1    indices should go around in CCW direction
//   |  /     so that the normal is cross(0->1,0->2)
//   0
struct scene_triangle {
    int vertex_idx[3]; // 3 vert indices
    scene_triangle() { vertex_idx[0] = vertex_idx[1] = vertex_idx[2] = -1; }
    scene_triangle(int i0, int i1, int i2)
    {
        vertex_idx[0] = i0;
        vertex_idx[1] = i1;
        vertex_idx[2] = i2;
    }
};

struct scene_instance_triangle {
    vec3 vertices[3]; // 3 verts
    int material_idx;
    scene_instance_triangle() {}
    scene_instance_triangle(vec3 v0, vec3 v1, vec3 v2, int mi)
    {
        vertices[0] = v0;
        vertices[1] = v1;
        vertices[2] = v2;
        material_idx = mi;
        // printf("sit v0=%f %f %f v1=%f %f %f v2=%f %f %f \n", vertices[0].e[0], vertices[0].e[1], vertices[0].e[2],
        //        vertices[1].e[0], vertices[1].e[1], vertices[1].e[2], vertices[2].e[0], vertices[2].e[1],
        //        vertices[2].e[2]);
    }
};

struct scene_sphere {
    vec3 center;
    double radius;
    int material_idx;
};

struct scene_moving_sphere {
    vec3 center0, center1;
    double time0, time1;
    double radius;
    int material_idx;
};

struct scene_obj {
    int num_vertices;
    int num_triangles;
    vec3 *vertices;
    scene_triangle *triangles;
    int cur_vertex_idx;
    int cur_triangle_idx;
    scene_obj() { num_vertices = num_triangles = 0; }
    scene_obj(int num_verts, int num_tris)
    {
        num_vertices = num_verts;
        num_triangles = num_tris;
        vertices = new vec3[num_vertices];
        triangles = new scene_triangle[num_triangles];
        cur_vertex_idx = 0;
        cur_triangle_idx = 0;
    };
    ~scene_obj()
    {
        delete[] vertices;
        delete[] triangles;
    };
    void add_vertex(vec3 v)
    {
        if (cur_vertex_idx == num_vertices) {
            std::cerr << std::string("ERROR: only expected " + std::to_string(num_vertices) + " vertices.");
            std::exit(1);
        }
        vertices[cur_vertex_idx++] = v;
    };
    void add_triangle(int i, int j, int k)
    {
        if (cur_triangle_idx == num_triangles) {
            std::cerr << std::string("ERROR: only expected " + std::to_string(num_triangles) + " triangles.");
            std::exit(1);
        }
        scene_triangle t = scene_triangle(i, j, k);
        triangles[cur_triangle_idx++] = t;
    };
    void finish()
    {
        if (cur_vertex_idx != num_vertices) {
            std::cerr << std::string("ERROR: expected " + std::to_string(num_vertices) + " vertices, got " +
                        std::to_string(cur_vertex_idx) + ".");
            std::exit(1);
        }
        if (cur_triangle_idx != num_triangles) {
            std::cerr << std::string("ERROR: expected " + std::to_string(num_triangles) + " triangles, got " +
                        std::to_string(cur_triangle_idx) + ".");
            std::exit(1);
        }
    }
};

class xform {
  public:
    virtual vec3 transform(vec3 v) = 0;
};

class xf_translation : public xform {
    vec3 t;

  public:
    xf_translation(vec3 v) : t(v) {}
    virtual vec3 transform(vec3 v) { return v + t; }
};
class xf_scale : public xform {
    vec3 s;

  public:
    xf_scale(vec3 v) : s(v) {}
    virtual vec3 transform(vec3 v) { return v * s; }
};
class xf_rotate : public xform {
    double angle; // angle in degrees
    vec3 axis;

  public:
    xf_rotate(FP_T a, vec3 v) : angle(a), axis(v) {}
    // https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    virtual vec3 transform(vec3 v)
    {
        FP_T theta = (FP_T)(angle * (pi / 180));
        FP_T cos_theta = cos(theta);
        FP_T sin_theta = sin(theta);
        vec3 rotated = (v * cos_theta) + (cross(axis, v) * sin_theta) + (axis * dot(axis, v)) * ((FP_T)1 - cos_theta);
        return rotated;
    }
};

struct scene_obj_inst {
    int obj_idx;
    int material_idx;
    std::vector<xform *> transforms;

    scene_obj_inst(int oi, int mi)
    {
        obj_idx = oi;
        material_idx = mi;
    }
    int num_triangles(std::vector<scene_obj *> &objs) { return objs[obj_idx]->num_triangles; }
    scene_instance_triangle *fill_instance_triangles(std::vector<scene_obj *> &objs,
                                                     scene_instance_triangle *instance_triangles)
    {
        for (int i = 0; i < objs[obj_idx]->num_triangles; i++) {
            vec3 v0 = objs[obj_idx]->vertices[objs[obj_idx]->triangles[i].vertex_idx[0]];
            vec3 v1 = objs[obj_idx]->vertices[objs[obj_idx]->triangles[i].vertex_idx[1]];
            vec3 v2 = objs[obj_idx]->vertices[objs[obj_idx]->triangles[i].vertex_idx[2]];
            v0 = transform(v0);
            v1 = transform(v1);
            v2 = transform(v2);
            *instance_triangles++ = scene_instance_triangle(v0, v1, v2, material_idx);
        }
        return instance_triangles;
    }
    vec3 transform(vec3 v)
    {
        for (auto xf : transforms) {
            v = xf->transform(v);
        }
        return v;
    }
    void add_translate(vec3 v) { transforms.push_back(new xf_translation(v)); }
    void add_scale(vec3 v) { transforms.push_back(new xf_scale(v)); }
    void add_rotate(FP_T angle, vec3 v) { transforms.push_back(new xf_rotate(angle, v)); }
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
    scene(const char *filename, const int image_width, const int image_height)
    {
        cam = nullptr;
        bool got_camera = false;
        std::string line;
        std::ifstream fl(filename);
        scene_obj *new_obj = nullptr;
        bool adding_obj = false;
        if (!fl.good()) {
            std::cerr << "ERROR: problem with opening file: " << filename << "\n";
            exit(2);
        }
        while (std::getline(fl, line)) {
            // std::cout << "LINE: " << line << "\n";
            if (line.find("camera") == 0) {
                got_camera = true;
                std::istringstream iss(line);
                std::vector<std::string> words;
                while (iss) {
                    std::string s;
                    iss >> s;
                    words.push_back(s);
                }
                double time0 = 0.0, time1 = 0.0;
                int cur_idx = 1; // skip camera
                vec3 lookfrom = vec3((FP_T)std::stod(words[cur_idx]), (FP_T)std::stod(words[cur_idx + 1]),
                                     (FP_T)std::stod(words[cur_idx + 2]));
                cur_idx += 3;
                vec3 lookat = vec3((FP_T)std::stod(words[cur_idx]), (FP_T)std::stod(words[cur_idx + 1]),
                                   (FP_T)std::stod(words[cur_idx + 2]));
                cur_idx += 3;
                vec3 vup = vec3((FP_T)std::stod(words[cur_idx]), (FP_T)std::stod(words[cur_idx + 1]),
                                (FP_T)std::stod(words[cur_idx + 2]));
                cur_idx += 3;
                double vfov = std::stod(words[cur_idx++]);
                double aperture = std::stod(words[cur_idx++]);
                double focus = std::stod(words[cur_idx++]);
                if (cur_idx < words.size() - 1) {
                    // we have time0, time1
                    time0 = std::stod(words[cur_idx++]);
                    time1 = std::stod(words[cur_idx++]);
                }
                double aspect_ratio = double(image_width) / image_height;
                cam = new camera(lookfrom, lookat, vup, (FP_T)vfov, (FP_T)aspect_ratio, (FP_T)aperture, (FP_T)focus,
                                 (FP_T)time0, (FP_T)time1);
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
                    iss >> r_str >> g_str >> b_str;
                    new_material->type = LAMBERTIAN;
                    new_material->mat.lambertian.albedo =
                        vec3((FP_T)std::stod(r_str), (FP_T)std::stod(g_str), (FP_T)std::stod(b_str));
                }
                else if (type_str == "metal") {
                    std::string r_str, g_str, b_str, f_str;
                    iss >> r_str >> g_str >> b_str;
                    iss >> f_str;
                    new_material->type = METAL;
                    new_material->mat.metal.albedo =
                        vec3((FP_T)std::stod(r_str), (FP_T)std::stod(g_str), (FP_T)std::stod(b_str));
                    new_material->mat.metal.fuzz = std::stod(f_str);
                }
                else if (type_str == "dielectric") {
                    std::string r_str;
                    iss >> r_str;
                    new_material->type = DIELECTRIC;
                    new_material->mat.dielectric.ref_idx = (FP_T)std::stod(r_str);
                }
                else {
                    std::cerr << "ERROR: unknown material type: " << type_str << std::endl;
                    exit(3);
                }
                int next_idx = (int)(materials.size());
                materials.push_back(new_material);
                material_names_to_idx.insert(std::pair<std::string, int>(name_str, next_idx));
            }
            else if (line.find("sphere") == 0) {
                std::istringstream iss(line);
                std::string sphere_str;
                std::string cx_str, cy_str, cz_str;
                std::string r_str;
                std::string mat_str;
                iss >> sphere_str;
                iss >> cx_str >> cy_str >> cz_str;
                iss >> r_str;
                iss >> mat_str;

                scene_sphere *new_sphere = new scene_sphere;

                new_sphere->center = vec3((FP_T)std::stod(cx_str), (FP_T)std::stod(cy_str), (FP_T)std::stod(cz_str));
                new_sphere->radius = (FP_T)std::stod(r_str);
                new_sphere->material_idx = material_names_to_idx[mat_str];
                spheres.push_back(new_sphere);
            }
            else if (line.find("msphere") == 0) {
                std::istringstream iss(line);
                std::string sphere_str;
                std::string c0x_str, c0y_str, c0z_str;
                std::string c1x_str, c1y_str, c1z_str;
                std::string t0_str, t1_str;
                std::string r_str;
                std::string mat_str;
                iss >> sphere_str;
                iss >> c0x_str >> c0y_str >> c0z_str;
                iss >> c1x_str >> c1y_str >> c1z_str;
                iss >> t0_str >> t1_str;
                iss >> r_str;
                iss >> mat_str;

                scene_moving_sphere *new_moving_sphere = new scene_moving_sphere;

                new_moving_sphere->center0 =
                    vec3((FP_T)std::stod(c0x_str), (FP_T)std::stod(c0y_str), (FP_T)std::stod(c0z_str));
                new_moving_sphere->center1 =
                    vec3((FP_T)std::stod(c1x_str), (FP_T)std::stod(c1y_str), (FP_T)std::stod(c1z_str));
                new_moving_sphere->time0 = std::stod(t0_str);
                new_moving_sphere->time1 = std::stod(t1_str);
                new_moving_sphere->radius = std::stod(r_str);
                new_moving_sphere->material_idx = material_names_to_idx[mat_str];
                moving_spheres.push_back(new_moving_sphere);
            }
            else if (line.find("obj_beg") == 0) {
                std::istringstream iss(line);
                std::string obj_str;
                std::string nv_str, nt_str;
                iss >> obj_str;
                iss >> nv_str >> nt_str;
                if (new_obj != nullptr) {
                    std::cerr << "ERROR: obj_beg called without prior obj_end.\n";
                    std::exit(1);
                }
                new_obj = new scene_obj(std::stoi(nv_str), std::stoi(nt_str));
                adding_obj = true;
            }
            else if (line.find("obj_vtx") == 0) {
                if (!adding_obj) {
                    std::cerr << "ERROR: obj_vtx called without prior obj_beg\n";
                    std::exit(1);
                }
                std::istringstream iss(line);
                std::string obj_str;
                std::string x_str, y_str, z_str;
                iss >> obj_str;
                iss >> x_str >> y_str >> z_str;
                new_obj->add_vertex(vec3((FP_T)std::stod(x_str), (FP_T)std::stod(y_str), (FP_T)std::stod(z_str)));
            }
            else if (line.find("obj_tri") == 0) {
                if (!adding_obj) {
                    std::cerr << "ERROR: obj_tri called without prior obj_beg.\n";
                    std::exit(1);
                }
                std::istringstream iss(line);
                std::string obj_str;
                std::string i_str, j_str, k_str;
                iss >> obj_str;
                iss >> i_str >> j_str >> k_str;
                new_obj->add_triangle(std::stoi(i_str), std::stoi(j_str), std::stoi(k_str));
            }
            else if (line.find("obj_end") == 0) {
                if (!adding_obj) {
                    std::string("ERROR: obj_end called without prior obj_beg.");
                    std::exit(1);
                }
                new_obj->finish();
                objs.push_back(new_obj);
                new_obj = nullptr;
                adding_obj = true;
            }
            else if (line.find("obj") == 0) {
                std::istringstream iss(line);
                std::vector<std::string> words;
                while (iss) {
                    std::string s;
                    iss >> s;
                    words.push_back(s);
                }
                if (words.size() >= 3) {
                    scene_obj_inst *new_obj_inst =
                        new scene_obj_inst(std::stoi(words[1]), material_names_to_idx[words[2]]);
                    int cur_idx = 3;
                    // words has an extra "blank" at the end
                    while (cur_idx < words.size() - 1) {
                        if (words[cur_idx][0] == 't') {
                            new_obj_inst->add_translate(vec3((FP_T)std::stod(words[cur_idx + 1]),
                                                             (FP_T)std::stod(words[cur_idx + 2]),
                                                             (FP_T)std::stod(words[cur_idx + 3])));
                            cur_idx += 4;
                        }
                        else if (words[cur_idx][0] == 's') {
                            new_obj_inst->add_scale(vec3((FP_T)std::stod(words[cur_idx + 1]),
                                                         (FP_T)std::stod(words[cur_idx + 2]),
                                                         (FP_T)std::stod(words[cur_idx + 3])));
                            cur_idx += 4;
                        }
                        else if (words[cur_idx][0] == 'r') {
                            new_obj_inst->add_rotate((FP_T)std::stod(words[cur_idx + 1]),
                                                     vec3((FP_T)std::stod(words[cur_idx + 2]),
                                                          (FP_T)std::stod(words[cur_idx + 3]),
                                                          (FP_T)std::stod(words[cur_idx + 4])));
                            cur_idx += 5;
                        }
                    }
                    obj_insts.push_back(new_obj_inst);
                }
                else {
                    std::cerr << "ERROR: obj called without enough args (count = " << words.size() << "\n";
                    std::exit(1);
                }
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
        if (spheres.size() + moving_spheres.size() + obj_insts.size() == 0) {
            std::cerr << "ERROR: Scene did not have any objects." << std::endl;
            std::exit(4);
        }
        std::cerr << "read scene file: " << filename << "\n";
        std::cerr << "material count:  " << materials.size() << "\n";
        std::cerr << "sphere count:    " << spheres.size() << std::endl;
        std::cerr << "msphere count:   " << moving_spheres.size() << std::endl;
        std::cerr << "obj count:       " << objs.size() << std::endl;
        std::cerr << "obj_inst count:  " << obj_insts.size() << std::endl;
        if (cam->t0() != cam->t1()) {
            std::cerr << "camera time:     " << cam->t0() << " - " << cam->t1() << std::endl;
        }
    }
    ~scene()
    {
        if (cam != nullptr) {
            delete cam;
        }
    }
    int num_triangles()
    {
        int n = 0;
        for (auto i : obj_insts) {
            n += i->num_triangles(objs);
        }
        return n;
    }
    void fill_instance_triangles(scene_instance_triangle *instance_triangles)
    {
        for (auto i : obj_insts) {
            instance_triangles = i->fill_instance_triangles(objs, instance_triangles);
        }
    }

    camera *cam;
    std::map<std::string, int> material_names_to_idx;
    std::vector<scene_material *> materials;
    std::vector<scene_sphere *> spheres;
    std::vector<scene_moving_sphere *> moving_spheres;
    std::vector<scene_obj *> objs;
    std::vector<scene_obj_inst *> obj_insts;
};

#endif
