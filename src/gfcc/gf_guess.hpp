#ifndef TAMM_TESTS_GFGUESS_HPP_
#define TAMM_TESTS_GFGUESS_HPP_

#include <algorithm>
#include <complex>
using namespace tamm;

template<typename T>
void gf_guess_ip(ExecutionContext& ec, const TiledIndexSpace& MO, const TAMM_SIZE nocc, double omega, 
              double gf_eta, int pi, std::vector<T>& p_evl_sorted_occ,
              Tensor<T>& t2v2_o, Tensor<std::complex<T>>& x1, 
              Tensor<std::complex<T>>& Minv, bool opt=false) {

    using ComplexTensor = Tensor<std::complex<T>>;

    const TiledIndexSpace& O = MO("occ");

    ComplexTensor guessM{O,O};
    ComplexTensor::allocate(&ec, guessM);

    Scheduler sch{ec};
    sch(guessM() = t2v2_o()).execute();

    using CMatrix   = Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    CMatrix guessM_eig(nocc,nocc);
    
    tamm_to_eigen_tensor(guessM, guessM_eig);
    ComplexTensor::deallocate(guessM);

    double denominator = 0.0;
    for(TAMM_SIZE i=0;i<nocc;i++){
        denominator = omega - p_evl_sorted_occ[i];
        if (denominator > 0 && denominator < 0.5) {
            denominator +=  0.5;
        }
        else if (denominator < 0 && denominator > -0.5) {
            denominator += -0.5;
        }
        guessM_eig(i,i) += std::complex<T>(denominator, -1.0*gf_eta); 
    }
    
    CMatrix guessMI = guessM_eig.inverse();

    if(opt){
        Eigen::Tensor<std::complex<T>, 1, Eigen::RowMajor> x1e(nocc);
        for(TAMM_SIZE i=0;i<nocc;i++)
            x1e(i) = guessMI(i,pi);

        eigen_to_tamm_tensor(x1,x1e);
    }
    else {
        Eigen::Tensor<std::complex<T>, 2, Eigen::RowMajor> x1e(nocc,1);
        for(TAMM_SIZE i=0;i<nocc;i++)
            x1e(i,0) = guessMI(i,pi);

        eigen_to_tamm_tensor(x1,x1e);
    }
    eigen_to_tamm_tensor(Minv,guessMI);

}

#endif //TAMM_TESTS_GFGUESS_HPP_
