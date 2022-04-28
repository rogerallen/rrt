#!/usr/bin/python3

import math

def camera(from_x, from_y, from_z):
    return f"""
# camera lookfrom  lookat   vup    vfov  aperture focus 
camera   {from_x} {from_y} {from_z}  0 {from_y} 0  0 1 0  15.0  0.01   6.0
"""

def materials():
    return f"""
# material name type options
material ground lambertian 0.8 0.8 0.8
material white  lambertian 0.95 0.95 0.95
material black  lambertian 0.1 0.1 0.1
material red    lambertian 1.0 0.0 0.0
material orange lambertian 1.0 0.5 0.0
material yellow lambertian 1.0 1.0 0.0
material green  lambertian 0.0 1.0 0.0
material blue   lambertian 0.0 0.0 1.0
material indigo lambertian 0.3 0.0 0.5
material violet lambertian 0.6 0.0 0.8
material mirror metal      0.8 0.8 0.8   0.01
material chrome metal      0.8 0.8 0.8   0.1
material bronze metal      0.8 0.7 0.0   0.1
material glass  dielectric 2.33
"""

def objects():
    return f"""
# 0 square 4 verts, 2 tris
obj_beg 4 2
obj_vtx -0.50000  0.00000 -0.50000
obj_vtx  0.50000  0.00000 -0.50000
obj_vtx -0.50000  0.00000  0.50000
obj_vtx  0.50000  0.00000  0.50000
obj_tri 0 3 1
obj_tri 0 2 3
obj_end

# 1 cube 8 verts, 12 tris
obj_beg 8 12
obj_vtx -0.50000 -0.50000 -0.50000
obj_vtx  0.50000 -0.50000 -0.50000
obj_vtx -0.50000 -0.50000  0.50000
obj_vtx  0.50000 -0.50000  0.50000
obj_vtx -0.50000  0.50000 -0.50000
obj_vtx  0.50000  0.50000 -0.50000
obj_vtx -0.50000  0.50000  0.50000
obj_vtx  0.50000  0.50000  0.50000
obj_tri 0 5 1
obj_tri 0 4 5
obj_tri 1 5 7
obj_tri 1 7 3
obj_tri 3 7 6
obj_tri 3 6 2
obj_tri 2 6 4
obj_tri 2 4 0
obj_tri 0 3 2
obj_tri 0 1 3
obj_tri 4 7 5
obj_tri 4 6 7
obj_end

sphere  0.5 0.25 -0.5  0.25  mirror
sphere  -0.5 0.25 -0.5  0.25  mirror

obj 0 glass r 90 1 0 0 t 0 0.5 2.5

# checkerboard
obj 0 red        t -4.00000 0.0 -4.00000
obj 0 black      t -4.00000 0.0 -3.00000
obj 0 red        t -4.00000 0.0 -2.00000
obj 0 black      t -4.00000 0.0 -1.00000
obj 0 red        t -4.00000 0.0  0.00000
obj 0 black      t -4.00000 0.0  1.00000
obj 0 red        t -4.00000 0.0  2.00000
obj 0 black      t -4.00000 0.0  3.00000
obj 0 black      t -3.00000 0.0 -4.00000
obj 0 red        t -3.00000 0.0 -3.00000
obj 0 black      t -3.00000 0.0 -2.00000
obj 0 red        t -3.00000 0.0 -1.00000
obj 0 black      t -3.00000 0.0  0.00000
obj 0 red        t -3.00000 0.0  1.00000
obj 0 black      t -3.00000 0.0  2.00000
obj 0 red        t -3.00000 0.0  3.00000
obj 0 red        t -2.00000 0.0 -4.00000
obj 0 black      t -2.00000 0.0 -3.00000
obj 0 red        t -2.00000 0.0 -2.00000
obj 0 black      t -2.00000 0.0 -1.00000
obj 0 red        t -2.00000 0.0  0.00000
obj 0 black      t -2.00000 0.0  1.00000
obj 0 red        t -2.00000 0.0  2.00000
obj 0 black      t -2.00000 0.0  3.00000
obj 0 black      t -1.00000 0.0 -4.00000
obj 0 red        t -1.00000 0.0 -3.00000
obj 0 black      t -1.00000 0.0 -2.00000
obj 0 red        t -1.00000 0.0 -1.00000
obj 0 black      t -1.00000 0.0  0.00000
obj 0 red        t -1.00000 0.0  1.00000
obj 0 black      t -1.00000 0.0  2.00000
obj 0 red        t -1.00000 0.0  3.00000
obj 0 red        t  0.00000 0.0 -4.00000
obj 0 black      t  0.00000 0.0 -3.00000
obj 0 red        t  0.00000 0.0 -2.00000
obj 0 black      t  0.00000 0.0 -1.00000
obj 0 red        t  0.00000 0.0  0.00000
obj 0 black      t  0.00000 0.0  1.00000
obj 0 red        t  0.00000 0.0  2.00000
obj 0 black      t  0.00000 0.0  3.00000
obj 0 black      t  1.00000 0.0 -4.00000
obj 0 red        t  1.00000 0.0 -3.00000
obj 0 black      t  1.00000 0.0 -2.00000
obj 0 red        t  1.00000 0.0 -1.00000
obj 0 black      t  1.00000 0.0  0.00000
obj 0 red        t  1.00000 0.0  1.00000
obj 0 black      t  1.00000 0.0  2.00000
obj 0 red        t  1.00000 0.0  3.00000
obj 0 red        t  2.00000 0.0 -4.00000
obj 0 black      t  2.00000 0.0 -3.00000
obj 0 red        t  2.00000 0.0 -2.00000
obj 0 black      t  2.00000 0.0 -1.00000
obj 0 red        t  2.00000 0.0  0.00000
obj 0 black      t  2.00000 0.0  1.00000
obj 0 red        t  2.00000 0.0  2.00000
obj 0 black      t  2.00000 0.0  3.00000
obj 0 black      t  3.00000 0.0 -4.00000
obj 0 red        t  3.00000 0.0 -3.00000
obj 0 black      t  3.00000 0.0 -2.00000
obj 0 red        t  3.00000 0.0 -1.00000
obj 0 black      t  3.00000 0.0  0.00000
obj 0 red        t  3.00000 0.0  1.00000
obj 0 black      t  3.00000 0.0  2.00000
obj 0 red        t  3.00000 0.0  3.00000
"""

for i in range(0,200+1,1):
    from_x = -1 + i/100.0   
    from_y = 0.25
    from_z = 5.0
    with open(f"anim3_{i:03d}.txt", 'w') as f:
        print(camera(from_x,from_y,from_z), file=f)
        print(materials(), file=f)
        print(objects(), file=f)

        