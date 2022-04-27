#!/usr/bin/python3
import math

def print_tetrahedron():
    print("# tetrahedron 4 verts 4 tris. centered at origin")
    print("obj_beg 4 4")
    for deg in [0,120,240]:
        theta = math.radians(deg)
        x = math.cos(theta)
        z = math.sin(theta)
        print(f"obj_vtx {x:5f} -0.5 {z:5f}")
    print("obj_vtx 0.0 0.5 0.0")
    print("obj_tri 0 1 3")
    print("obj_tri 0 3 2")
    print("obj_tri 0 2 1")
    print("obj_tri 1 2 3")
    print("obj_end")

def print_pyramid():
    print("# pyramid 5 verts, 6 tris. centered at origin")
    print("obj_beg 5 6")
    print("obj_vtx -0.5 -0.5  -0.5")
    print("obj_vtx  0.5 -0.5  -0.5")
    print("obj_vtx  0.5 -0.5  0.5")
    print("obj_vtx -0.5 -0.5  0.5")
    print("obj_vtx  0.0  0.5  0.0")
    print("obj_tri 0 1 2")
    print("obj_tri 0 2 3")
    print("obj_tri 0 1 4")
    print("obj_tri 1 2 4")
    print("obj_tri 2 3 4")
    print("obj_tri 3 0 4")
    print("obj_end")

print_tetrahedron()
print_pyramid()