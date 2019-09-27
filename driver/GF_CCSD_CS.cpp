// #define CATCH_CONFIG_RUNNER

#include "gfcc/gf_ccsd_cs.hpp"
#include <algorithm>
// #include <Eigen/QR>
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

    MPI_Init(&argc,&argv);
    GA_Initialize();
    MA_init(MT_DBL, 8000000, 20000000);
    
    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    #ifdef USE_TALSH
    TALSH talsh_instance;
    talsh_instance.initialize(mpi_rank);
    #endif

    gfccsd_main_driver(filename);
    
    #ifdef USE_TALSH
    //talshStats();
    talsh_instance.shutdown();
    #endif  

    GA_Terminate();
    MPI_Finalize();

    return 0;
}
