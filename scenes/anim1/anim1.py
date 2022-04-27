#!/usr/bin/python3

import math

def script(from_x, from_y, from_z):
    return f"""
# camera lookfrom               lookat   vup    vfov  aperture focus 
camera   {from_x} {from_y} {from_z}  0 1.0 0  0 1 0  30.0  0.1   5.0

# material name type options
material ground lambertian 0.8 0.8 0.0
material pinky  lambertian 0.7 0.3 0.3
material mirror metal      0.8 0.8 0.8   0.1
material bronze metal      0.8 0.6 0.2   1.0
material glass  dielectric 1.35

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
sphere  0.0 -100.5  -1.0   100.0  ground
sphere  0.0    1.5 -1.0     0.5  mirror
sphere -1.0    1.5 -0.5     0.5  pinky
sphere  1.0    1.5 -0.5     0.5  bronze

# obj 0 instance (translation) material
obj 0 glass  t  0.0 0.90  0.20     
obj 0 mirror  t -0.65 0.25 -0.75    
obj 0 bronze t  0.65 0.25 -0.75    
"""

for i in range(0,360,1):
    deg = i
    theta = math.radians(deg)
    from_x = 5.0 * math.cos(theta)    
    from_y = 1.0
    from_z = 5.0 * math.sin(theta)
    with open(f"anim1_{i:03d}.txt", 'w') as f:
        print(script(from_x,from_y,from_z), file=f)    