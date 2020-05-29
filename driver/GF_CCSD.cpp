
#include "gfcc/gf_ccsd.hpp"
#include <algorithm>
#undef I

using namespace tamm;

void gfccsd_main_driver(std::string);

// Main 
int main( int argc, char* argv[] ){
    if(argc<2){
        std::cout << "Please provide an input file!" << std::endl;
        return 1;
    }

    std::string filename = std::string(argv[1]);
    std::ifstream testinput(filename); 
    if(!testinput){
        std::cout << "Input file provided [" << filename << "] does not exist!" << std::endl;
        return 1;
    }

    tamm::initialize(argc, argv);

    gfccsd_main_driver(filename);
    
    tamm::finalize();

    return 0;
}
