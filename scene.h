#ifndef SCENEH
#define SCENEH

#include "vec3.h"
#include <fstream>
#include <sstream>
#include <string>

class scene {
  public:
    scene(char *filename) { 
        std::string line; 
        std::ifstream fl(filename);
        if(!fl.good()) {
            std::cerr << "ERROR: problem with opening file: " << filename << "\n";
            exit(2);
        }
        while(getline(fl,line)) {
            std::cout << "scene: " << line << "\n";
            if(line[0] == '#') {
                std::cout << "!comment\n";
                continue;
            }
            else if(line.find("camera") == 0) {
                std::cout << "!camera\n";
            }
            else if(line.find("sphere") == 0) {
                std::cout << "!sphere\n";
            }
        }
        fl.close();
    }
};

#endif
