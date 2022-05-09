#ifndef BVH_H
#define BVH_H

#include "rtweekend.h"

#include "hittable.h"
#include "hittable_list.h"

#include <algorithm>

class bvh_node : public hittable {
  public:
    HOSTDEV bvh_node();

    DEV bvh_node(const hittable_list &list, FP_T time0, FP_T time1)
        : bvh_node(list.objects, 0,
#ifndef USE_CUDA
                   list.objects.size(),
#else
                   list.objects_size,
#endif
                   time0, time1)
    {
        // printf("bvh_node entry &list,t0,t1\n");
    }

    DEV bvh_node(
#ifndef USE_CUDA
        const std::vector<shared_ptr<hittable>> &src_objects,
#else
        hittable **src_objects,
#endif
        size_t start, size_t end, FP_T time0, FP_T time1);

    HOSTDEV bvh_node(bvh_node &b); // copy from host to device constructor

    HOSTDEV virtual bool hit(const ray &r, FP_T t_min, FP_T t_max, hit_record &rec, bool debug) const override;
    HOSTDEV virtual bool bounding_box(FP_T time0, FP_T time1, aabb &output_box) const override;
    HOSTDEV virtual void print(int i) const override;

  public:
    hittable_ptr_t left;
    hittable_ptr_t right;
    aabb box;
};

DEV inline bool box_compare(const hittable_ptr_t a, const hittable_ptr_t b, int axis)
{
    aabb box_a;
    aabb box_b;

    if (!a->bounding_box(0, 0, box_a) || !b->bounding_box(0, 0, box_b))
        printf("ERROR: No bounding box in bvh_node constructor.\n");

    return box_a.min().e[axis] < box_b.min().e[axis];
}

DEV bool box_x_compare(const hittable_ptr_t a, const hittable_ptr_t b) { return box_compare(a, b, 0); }

DEV bool box_y_compare(const hittable_ptr_t a, const hittable_ptr_t b) { return box_compare(a, b, 1); }

DEV bool box_z_compare(const hittable_ptr_t a, const hittable_ptr_t b) { return box_compare(a, b, 2); }

#ifdef USE_CUDA
DEV void dev_sort(const hittable_ptr_t *a, const hittable_ptr_t *b,
                  bool (*comparator)(const hittable_ptr_t a, const hittable_ptr_t b))
{
    // simplest sort for now.  FIXME
    for (int i = 0; a + i < b; i++) {
        for (int j = 0; a + j < b; j++) {
            if (!((*comparator)(*(a + i), *(a + j)))) {
                const hittable *tmp = *(a + i);
                *(a + i) = *(a + j);
                *(a + j) = tmp;
            }
        }
    }
}
#endif

DEV bvh_node::bvh_node(
#ifndef USE_CUDA
    const std::vector<shared_ptr<hittable>> &src_objects,
#else
    hittable **src_objects,
#endif
    size_t start, size_t end, FP_T time0, FP_T time1)
{
    // printf("bvh_node %ld %ld\n", start, end);
    //  Create a modifiable array of the source scene objects
    //  FIXME -- this copies ALL of the objects.  Not just start...end.
#ifndef USE_CUDA
    auto objects = src_objects;
#else
    // this creates an array that is larger than necessary,
    // but matches C++ code for the rest of the method.
    hittable **objects = new hittable *[end + 1];
    // just copy the stuff from start..end
    for (int i = start; i < end + 1; i++) {
        objects[i] = *(src_objects + i);
    }
#endif

#ifndef USE_CUDA
    int axis = random_int(0, 2);
#else
    // passing in CURAND state is annoying for bvh construction
    static int axis = 0;
    axis = (axis + 1) % 3;
#endif

    auto comparator = (axis == 0) ? box_x_compare : (axis == 1) ? box_y_compare : box_z_compare;

    size_t object_span = end - start;

    if (object_span == 1) {
        left = right = objects[start];
    }
    else if (object_span == 2) {
        if (comparator(objects[start], objects[start + 1])) {
            left = objects[start];
            right = objects[start + 1];
        }
        else {
            left = objects[start + 1];
            right = objects[start];
        }
    }
    else {
#ifndef USE_CUDA
        std::sort(objects.begin() + start, objects.begin() + end, comparator);
#else
        dev_sort((const hittable **)(objects + start), (const hittable **)(objects + end), comparator);
#endif
        auto mid = start + object_span / 2;
#ifndef USE_CUDA
        left = make_shared<bvh_node>(objects, start, mid, time0, time1);
        right = make_shared<bvh_node>(objects, mid, end, time0, time1);
#else
        // FIXME -- when to delete
        left = new bvh_node(objects, start, mid, time0, time1);
        right = new bvh_node(objects, mid, end, time0, time1);
#endif
    }

#ifdef USE_CUDA
    delete[] objects;
#endif

    aabb box_left, box_right;

    if (!left->bounding_box(time0, time1, box_left) || !right->bounding_box(time0, time1, box_right)) {
        printf("ERROR: No bounding box in bvh_node constructor.\n");
    }

    box = surrounding_box(box_left, box_right);

    // printf("bvh_node %ld %ld EXIT\n", start, end);
}

HOSTDEV bool bvh_node::bounding_box(FP_T time0, FP_T time1, aabb &output_box) const
{
    output_box = box;
    return true;
}

HOSTDEV bool bvh_node::hit(const ray &r, FP_T t_min, FP_T t_max, hit_record &rec, bool debug) const
{
    if (!box.hit(r, t_min, t_max)) return false;

    bool hit_left = left->hit(r, t_min, t_max, rec, debug);
    bool hit_right = right->hit(r, t_min, hit_left ? rec.t : t_max, rec, debug);

    return hit_left || hit_right;
}

HOSTDEV void bvh_node::print(int i) const
{
    printf("bvh_node %d ", i);
    box.print(i);
    printf("\n");
    for (int j = 0; j < i; j++) {
        printf(" ");
    }
    printf("LEFT: ");
    left->print(i + 1);
    for (int j = 0; j < i; j++) {
        printf(" ");
    }
    printf("RIGHT: ");
    right->print(i + 1);
}

#endif