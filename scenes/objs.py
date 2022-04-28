#!/usr/bin/python3
# 
# objects centered at origin, y is up, xz plane is ground
#
#         Y ^
#           |
#           +--> X
#          / 
#       Z v   
#
import math

def print_tetrahedron():
    print("# tetrahedron 4 verts 4 tris")
    print("obj_beg 4 4")
    for deg in [0,120,240]:
        theta = math.radians(deg)
        x = math.cos(theta)
        y = -0.5
        z = math.sin(theta)
        print(f"obj_vtx {x: 3.5f} {y: 3.5f} {z: 3.5f}")
    x,y,z = 0.0,0.5,0.0
    print(f"obj_vtx {x: 3.5f} {y: 3.5f} {z: 3.5f}")
    print("obj_tri 0 1 2")
    print("obj_tri 0 3 1")
    print("obj_tri 1 3 2")
    print("obj_tri 2 3 0")
    print("obj_end")
    print()

def print_pyramid():
    print("# pyramid 5 verts, 6 tris")
    print("obj_beg 5 6")
    print("obj_vtx -0.5 -0.5  -0.5")
    print("obj_vtx  0.5 -0.5  -0.5")
    print("obj_vtx -0.5 -0.5  0.5")
    print("obj_vtx  0.5 -0.5  0.5")
    print("obj_vtx  0.0  0.5  0.0")
    print("obj_tri 0 1 3")
    print("obj_tri 0 3 2")
    print("obj_tri 4 1 0")
    print("obj_tri 4 3 1")
    print("obj_tri 4 2 3")
    print("obj_tri 4 0 2")
    print("obj_end")
    print()

def print_square():
    print("# square 4 verts, 2 tris")
    print("obj_beg 4 2")
    y = 0.0
    for z in [-0.5,0.5]:
        for x in [-0.5,0.5]:
            print(f"obj_vtx {x: 3.5f} {y: 3.5f} {z: 3.5f}")
    print("obj_tri 0 3 1") # normal points in +y
    print("obj_tri 0 2 3")
    print("obj_end")
    print()

def print_cube():
    print("# cube 8 verts, 12 tris")
    print("obj_beg 8 12")
    for y in [-0.5,0.5]:
        for z in [-0.5,0.5]:
            for x in [-0.5,0.5]:
                print(f"obj_vtx {x: 3.5f} {y: 3.5f} {z: 3.5f}")
    print("obj_tri 0 5 1") # sides
    print("obj_tri 0 4 5")
    print("obj_tri 1 5 7")
    print("obj_tri 1 7 3")
    print("obj_tri 3 7 6")
    print("obj_tri 3 6 2")
    print("obj_tri 2 6 4")
    print("obj_tri 2 4 0")
    print("obj_tri 0 3 2") # bot
    print("obj_tri 0 1 3")
    print("obj_tri 4 7 5") # top
    print("obj_tri 4 6 7")
    print("obj_end")
    print()

def print_instance_checkerboard(n,num,mat1,mat2):
    print("# checkerboard")
    for i in range(n):
        for j in range(n):
            x = i - 4
            y = j - 4
            mat = mat1 if ((i ^ j) % 2 == 0) else mat2
            print(f"obj {num} {mat:<10s} t {x: 3.5f} 0.0 {y: 3.5f}")

print_tetrahedron()
print_pyramid()
print_square()
print_cube()
print_instance_checkerboard(8,0,"red","black")
