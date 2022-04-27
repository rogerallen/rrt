#!/usr/bin/python3

import math

def camera(from_x, from_y, from_z):
    return f"""
# camera lookfrom  lookat   vup    vfov  aperture focus 
camera   {from_x} {from_y} {from_z}  0 {from_y} 0  0 1 0  24.0  0.1   5.0
"""

def materials():
    return f"""
# material name type options
material ground lambertian 0.8 0.8 0.8
material red    lambertian 1.0 0.0 0.0
material orange lambertian 1.0 0.5 0.0
material yellow lambertian 1.0 1.0 0.0
material green  lambertian 0.0 1.0 0.0
material blue   lambertian 0.0 0.0 1.0
material indigo lambertian 0.3 0.0 0.5
material violet lambertian 0.6 0.0 0.8
material mirror metal      0.8 0.8 0.8   0.01
material glass  dielectric 1.35
"""

def objects():
    return f"""
# obj 0 tetrahedron 4 vert, 4 face
obj_beg 4 4
obj_vtx  0.0  0.5774 -0.2041
obj_vtx -0.5 -0.2887 -0.2041
obj_vtx  0.5 -0.2887 -0.2041
obj_vtx  0.0  0.0     0.6124
obj_tri 0 1 3
obj_tri 0 3 2
obj_tri 0 2 1
obj_tri 1 2 3
obj_end

# sphere x y z radius material
sphere  0.0 -100.0  0.0   100.0  ground
sphere  0.0    1.4  0.0     0.5  mirror

# obj 0 instance (translation) material
obj 0 glass  t 0.0  0.2887 0.0  
"""

def print_ring(f,y):
    colors = "red orange yellow green blue indigo violet".split()
    for i,c in enumerate(colors):
        deg = 360.0/len(colors)
        theta = math.radians(i*deg + y*deg*0.5)
        r = 0.75
        x = r * math.cos(theta)
        z = r * math.sin(theta)
        print(f"sphere {x} {y} {z} 0.1 {c}",file=f)

for i in range(0,360,1):
    deg = i
    theta = math.radians(deg)
    from_x = 5.0 * math.cos(theta)    
    from_y = 0.9
    from_z = 5.0 * math.sin(theta)
    with open(f"anim2_{i:03d}.txt", 'w') as f:
        print(camera(from_x,from_y,from_z), file=f)
        print(materials(), file=f)
        print(objects(), file=f)
        for y in [0.0, 0.5, 1.0, 1.5, 2.0]:
            print_ring(f,y)
        