#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "hittable.h"

#ifndef USE_CUDA
#include <memory>
#include <vector>
using std::shared_ptr;
#endif

class hittable_list : public hittable {
  public:
    HOSTDEV hittable_list()
    {
#ifdef USE_CUDA
        objects_size = 0;
        objects_reserve = 0;
#endif
    }

#ifdef USE_CUDA
    HOSTDEV ~hittable_list()
    {
        // NOTE -- this is deleting all of the objects passed to you
        // this works for our use case, but might not be best practice
        for (int i = 0; i < objects_size; i++) {
            delete objects[i];
        }
        delete[] objects;
    }
#endif

    HOSTDEV hittable_list(hittable_ptr_t object) { add(object); }

    HOSTDEV void clear()
    {
#ifndef USE_CUDA
        objects.clear();
#else
        for (int i = 0; i < objects_size; i++) {
            delete objects[i];
        }
        objects_size = 0;
#endif
    }

    HOSTDEV void add(hittable_ptr_t object)
    {
#ifndef USE_CUDA
        objects.push_back(object);
#else
        if (objects_size == objects_reserve) {
            increment_reserve();
        }
        objects[objects_size++] = object;
#endif
    }

#ifdef USE_CUDA
    HOSTDEV void increment_reserve()
    {
        // increment reserve
        objects_reserve += OBJECTS_COUNT;
        hittable **new_objects = new hittable *[objects_reserve]; // FIXME new might fail someday
        // copy from old to new
        for (int i = 0; i < objects_size; i++) {
            new_objects[i] = objects[i];
        }
        // get rid of old
        delete[] objects;
        // start using new
        objects = new_objects;
    }
#endif

    HOSTDEV virtual bool hit(const ray &r, FP_T t_min, FP_T t_max, hit_record &rec, bool debug) const override;
    HOSTDEV virtual bool bounding_box(FP_T time0, FP_T time1, aabb &output_box) const override;
    HOSTDEV virtual void print(int i) const override;

  public:
#ifndef USE_CUDA
    std::vector<shared_ptr<hittable>> objects;
#else
    hittable **objects;
    int objects_size;            // how many actual objects
    int objects_reserve;         // how many objects can we handle
    const int OBJECTS_COUNT = 4; // how many objects to allocate and increment
#endif
};

HOSTDEV bool hittable_list::hit(const ray &r, FP_T t_min, FP_T t_max, hit_record &rec, bool debug) const
{
    hit_record temp_rec;
    bool hit_anything = false;
    auto closest_so_far = t_max;

#ifndef USE_CUDA
    for (const auto &object : objects) {
#else
    for (int i = 0; i < objects_size; i++) {
        if (debug) printf("DEBUG hit test %d\n", i);
        hittable *object = objects[i];
#endif
        if (object->hit(r, t_min, closest_so_far, temp_rec, debug)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
            if (debug) printf("DEBUG ! hit !\n");
        }
    }

    return hit_anything;
}

HOSTDEV bool hittable_list::bounding_box(FP_T time0, FP_T time1, aabb &output_box) const
{
#ifndef USE_CUDA
    if (objects.empty()) return false;
#else
    if (objects_size == 0) return false;
#endif

    aabb temp_box;
    bool first_box = true;

#ifndef USE_CUDA
    for (const auto &object : objects) {
#else
    for (int i = 0; i < objects_size; i++) {
        hittable *object = objects[i];
#endif
        if (!object->bounding_box(time0, time1, temp_box)) return false;
        output_box = first_box ? temp_box : surrounding_box(output_box, temp_box);
        first_box = false;
    }

    return true;
}

HOSTDEV void hittable_list::print(int i) const { printf("hitable_list print %d?\n", i); }

#endif