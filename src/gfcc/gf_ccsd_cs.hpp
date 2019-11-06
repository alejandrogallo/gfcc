// #define CATCH_CONFIG_RUNNER
#ifndef GFCCSD_HPP_
#define GFCCSD_HPP_

#include "gf_diis_cs.hpp"
#include "gf_guess.hpp"
#include "contrib/cd_ccsd_common.hpp"
#include <algorithm>
// #include <Eigen/QR>
#undef I

using namespace tamm;

#include <filesystem>
namespace fs = std::filesystem;

//TODO input file
size_t ndiis;
size_t gf_maxiter;

int gf_nprocs_poi;
double gf_omega;
size_t p_oi; //number of occupied/all MOs
double gf_eta;
double gf_threshold;
double omega_min   ;
double omega_max   ;
double lomega_min  ;
double lomega_max  ;
double omega_delta ;
int64_t omega_npts ;
int64_t lomega_npts;
double omega_delta_e;
double gf_level_shift;
double gf_damping_factor;
int gf_extrapolate_level;

#define GF_PGROUPS 1
#define GF_IN_SG 0
#define GF_GS_SG 0

// Tensor<double> lambda_y1, lambda_y2,
Tensor<double> d_t1_a, d_t1_b,
               d_t2_aaaa, d_t2_bbbb, d_t2_abab,
               v2_aaaa, v2_abab, v2_bbbb,
               t2v2,
               lt12_a,
              //  ix1_1_a, 
               ix1_1_1_a, ix1_1_1_b, 
              //  ix1_4_aaaa, ix1_4_abab,
               ix2_1_aaaa, ix2_1_abab,
               ix2_2_a, ix2_2_b, 
               ix2_3_a, ix2_3_b, 
               ix2_4_aaaa, ix2_4_abab, 
               ix2_5_aaaa, ix2_5_abba, ix2_5_abab, 
               ix2_5_bbbb, ix2_5_baab,
               ix2_6_2_a, ix2_6_2_b,
               ix2_6_3_aaaa, ix2_6_3_abba, ix2_6_3_abab, 
               ix2_6_3_bbbb, ix2_6_3_baab;

TiledIndexSpace diis_tis;

template<typename T>
T find_closest(T w, std::vector<T>& wlist){
  double diff = std::abs(wlist[0]-w);
  int idx = 0;
  for (size_t c=1; c<wlist.size(); c++){
    double cdiff = std::abs(wlist[c]-w);
    if(cdiff < diff) {
      idx = c;
      diff = cdiff;
    }
  }

  return wlist[idx];
}

void write_string_to_disk(ExecutionContext& ec, const std::string& tstring, const std::string& filename) {

    int tstring_len = tstring.length();
    auto rank = ec.pg().rank().value();

    std::vector<int> recvcounts;
    auto size = ec.pg().size().value();

    if (rank == 0)
        recvcounts.resize(size,0); 

    MPI_Gather(&tstring_len, 1, MPI_INT,
              &recvcounts[0], 1, MPI_INT,
              0, ec.pg().comm());

    /*
    * Figure out the total length of string, 
    * and displacements for each rank 
    */

    int totlen = 0;
    std::vector<int> displs;
    char *combined_string = nullptr;

    if (rank == 0) {
        displs.resize(size,0);

        displs[0] = 0;
        totlen += recvcounts[0]+1;

        for (int i=1; i<size; i++) {
          totlen += recvcounts[i]+1;   /* plus one for space or \0 after words */
          displs[i] = displs[i-1] + recvcounts[i-1] + 1;
        }

        /* allocate string, pre-fill with spaces and null terminator */
        combined_string = new char[totlen];           
        for (int i=0; i<totlen-1; i++)
            combined_string[i] = ' ';
        combined_string[totlen-1] = '\0';
    }

    // Gather strings from all ranks in pg
    MPI_Gatherv(tstring.c_str(), tstring_len, MPI_CHAR,
                combined_string, &recvcounts[0], &displs[0], MPI_CHAR,
                0, ec.pg().comm());


    if (rank == 0) {
        cout << combined_string << endl;
        std::ofstream out(filename, std::ios::out);
        if(!out) cerr << "Error opening file " << filename << endl;
        out << combined_string << std::endl;
        out.close();        
        delete combined_string;
    }

}

template<typename T>
void gfccsd_x1(/* ExecutionContext& ec, */
               Scheduler& sch, const TiledIndexSpace& MO,
               Tensor<std::complex<T>>& i0_a,
               const Tensor<T>& t1_a,    const Tensor<T>& t1_b, 
               const Tensor<T>& t2_aaaa, const Tensor<T>& t2_bbbb, const Tensor<T>& t2_abab,
               const Tensor<std::complex<T>>& x1_a,
               const Tensor<std::complex<T>>& x2_aaa, const Tensor<std::complex<T>>& x2_bab,
               const Tensor<T>& f1, //const Tensor<T>& v2,
               const TiledIndexSpace& gf_tis) {

   TiledIndexSpace o_alpha,v_alpha,o_beta,v_beta;

   const TiledIndexSpace &O = MO("occ");
   const TiledIndexSpace &V = MO("virt");
    const int otiles   = O.num_tiles();
    const int vtiles   = V.num_tiles();
    const int oabtiles = otiles/2;
    const int vabtiles = vtiles/2;

    o_alpha = {MO("occ"), range(oabtiles)};
    v_alpha = {MO("virt"),range(vabtiles)};
    o_beta  = {MO("occ"), range(oabtiles,otiles)};
    v_beta  = {MO("virt"),range(vabtiles,vtiles)};   
    auto [p7_va] = v_alpha.labels<1>("all");
    auto [p7_vb] = v_beta.labels<1>("all");
    auto [h1_oa,h6_oa,h8_oa] = o_alpha.labels<3>("all");
    auto [h6_ob,h8_ob] = o_beta.labels<2>("all");
    auto [u1] = gf_tis.labels<1>("all");

    sch
      ( i0_a(h1_oa,u1)  =  0 )
      ( i0_a(h1_oa,u1) += -1   * x1_a(h6_oa,u1) * ix2_2_a(h6_oa,h1_oa) )
      ( i0_a(h1_oa,u1) +=        x2_aaa(p7_va,h1_oa,h6_oa,u1) * ix1_1_1_a(h6_oa,p7_va) )
      ( i0_a(h1_oa,u1) +=        x2_bab(p7_vb,h1_oa,h6_ob,u1) * ix1_1_1_b(h6_ob,p7_vb) )
      ( i0_a(h1_oa,u1) +=  0.5 * x2_aaa(p7_va,h6_oa,h8_oa,u1) * ix2_6_3_aaaa(h6_oa,h8_oa,h1_oa,p7_va) )
      ( i0_a(h1_oa,u1) +=        x2_bab(p7_vb,h6_oa,h8_ob,u1) * ix2_6_3_abab(h6_oa,h8_ob,h1_oa,p7_vb) );
      if(debug) sch.execute();
}

template<typename T>
void gfccsd_x2(/* ExecutionContext& ec, */
               Scheduler& sch, const TiledIndexSpace& MO,
               Tensor<std::complex<T>>& i0_aaa, Tensor<std::complex<T>>& i0_bab,
               const Tensor<T>& t1_a,    const Tensor<T>& t1_b,
               const Tensor<T>& t2_aaaa, const Tensor<T>& t2_bbbb, const Tensor<T>& t2_abab,
               const Tensor<std::complex<T>>& x1_a,   
               const Tensor<std::complex<T>>& x2_aaa, const Tensor<std::complex<T>>& x2_bab,
               const Tensor<T>& f1, //const Tensor<T>& v2,
               const TiledIndexSpace& gf_tis) {

   using ComplexTensor = Tensor<std::complex<T>>;
   TiledIndexSpace o_alpha,v_alpha,o_beta,v_beta;

   const TiledIndexSpace &O = MO("occ");
   const TiledIndexSpace &V = MO("virt");
   auto [u1] = gf_tis.labels<1>("all");

   const int otiles   = O.num_tiles();
   const int vtiles   = V.num_tiles();
   const int oabtiles = otiles/2;
   const int vabtiles = vtiles/2;

   o_alpha = {MO("occ"), range(oabtiles)};
   v_alpha = {MO("virt"),range(vabtiles)};
   o_beta  = {MO("occ"), range(oabtiles,otiles)};
   v_beta  = {MO("virt"),range(vabtiles,vtiles)};

   auto [p3_va,p4_va,p5_va,p8_va,p9_va] = v_alpha.labels<5>("all");
   auto [p3_vb,p4_vb,p5_vb,p8_vb,p9_vb] = v_beta.labels<5>("all");
   auto [h1_oa,h2_oa,h6_oa,h7_oa,h8_oa,h9_oa,h10_oa] = o_alpha.labels<7>("all");
   auto [h1_ob,h2_ob,h6_ob,h7_ob,h8_ob,h10_ob] = o_beta.labels<6>("all");

   ComplexTensor i_6_aaa      {o_alpha,o_alpha,o_alpha,gf_tis};
   ComplexTensor i_6_bab      {o_beta, o_alpha,o_beta, gf_tis};
   ComplexTensor i_10_a       {v_alpha,gf_tis};
   ComplexTensor i_11_aaa     {o_alpha,o_alpha,v_alpha,gf_tis};
   ComplexTensor i_11_bab     {o_beta, o_alpha,v_beta, gf_tis};
   ComplexTensor i_11_bba     {o_beta, o_beta, v_alpha,gf_tis};
   ComplexTensor i0_temp_aaa  {v_alpha,o_alpha,o_alpha,gf_tis};
   ComplexTensor i0_temp_bab  {v_beta, o_alpha,o_beta, gf_tis};
   ComplexTensor i0_temp_bba  {v_beta, o_beta, o_alpha,gf_tis};
   ComplexTensor i_6_temp_aaa {o_alpha,o_alpha,o_alpha,gf_tis};
   ComplexTensor i_6_temp_bab {o_beta, o_alpha,o_beta, gf_tis};
   ComplexTensor i_6_temp_bba {o_beta, o_beta, o_alpha,gf_tis};
   
   sch
     .allocate(i_6_aaa,i_6_bab,
               i_10_a,i_11_aaa,i_11_bab,i_11_bba,
               i0_temp_aaa,i0_temp_bab,i0_temp_bba,
               i_6_temp_aaa,i_6_temp_bab,i_6_temp_bba)
     ( i0_aaa(p4_va,h1_oa,h2_oa,u1)    =  0 )
     ( i0_bab(p4_vb,h1_oa,h2_ob,u1)    =  0 )
  
     ( i0_aaa(p4_va,h1_oa,h2_oa,u1)      +=  x1_a(h9_oa,u1) * ix2_1_aaaa(h9_oa,p4_va,h1_oa,h2_oa) ) 
     ( i0_bab(p4_vb,h1_oa,h2_ob,u1)      +=  x1_a(h9_oa,u1) * ix2_1_abab(h9_oa,p4_vb,h1_oa,h2_ob) )
      
     ( i0_temp_aaa(p3_va,h1_oa,h2_oa,u1)  =       x2_aaa(p3_va,h1_oa,h8_oa,u1) * ix2_2_a(h8_oa,h2_oa) )
     ( i0_temp_bab(p3_vb,h1_oa,h2_ob,u1)  =       x2_bab(p3_vb,h1_oa,h8_ob,u1) * ix2_2_b(h8_ob,h2_ob) )
     ( i0_temp_bba(p3_vb,h1_ob,h2_oa,u1)  =  -1 * x2_bab(p3_vb,h8_oa,h1_ob,u1) * ix2_2_a(h8_oa,h2_oa) )
  
     ( i0_temp_aaa(p4_va,h1_oa,h2_oa,u1) +=       x2_aaa(p8_va,h1_oa,h7_oa,u1) * ix2_5_aaaa(h7_oa,p4_va,h2_oa,p8_va) ) //O3V2
     ( i0_temp_aaa(p4_va,h1_oa,h2_oa,u1) +=       x2_bab(p8_vb,h1_oa,h7_ob,u1) * ix2_5_baab(h7_ob,p4_va,h2_oa,p8_vb) )
     ( i0_temp_bab(p4_vb,h1_oa,h2_ob,u1) +=       x2_bab(p8_vb,h1_oa,h7_ob,u1) * ix2_5_bbbb(h7_ob,p4_vb,h2_ob,p8_vb) )
     ( i0_temp_bab(p4_vb,h1_oa,h2_ob,u1) +=       x2_aaa(p8_va,h1_oa,h7_oa,u1) * ix2_5_abba(h7_oa,p4_vb,h2_ob,p8_va) )
     ( i0_temp_bba(p4_vb,h1_ob,h2_oa,u1) +=  -1 * x2_bab(p8_vb,h7_oa,h1_ob,u1) * ix2_5_abab(h7_oa,p4_vb,h2_oa,p8_vb) )
  
     (   i_11_aaa(h6_oa,h1_oa,p5_va,u1)   =  x2_aaa(p8_va,h1_oa,h7_oa,u1) * v2_aaaa(h6_oa,h7_oa,p5_va,p8_va) )
     (   i_11_aaa(h6_oa,h1_oa,p5_va,u1)  +=  x2_bab(p8_vb,h1_oa,h7_ob,u1) * v2_abab(h6_oa,h7_ob,p5_va,p8_vb) )
     (   i_11_bab(h6_ob,h1_oa,p5_vb,u1)   =  x2_bab(p8_vb,h1_oa,h7_ob,u1) * v2_bbbb(h6_ob,h7_ob,p5_vb,p8_vb) )
     (   i_11_bab(h6_ob,h1_oa,p5_vb,u1)  +=  x2_aaa(p8_va,h1_oa,h7_oa,u1) * v2_abab(h7_oa,h6_ob,p8_va,p5_vb) )
     (   i_11_bba(h6_ob,h1_ob,p5_va,u1)   =  x2_bab(p8_vb,h7_oa,h1_ob,u1) * v2_abab(h7_oa,h6_ob,p5_va,p8_vb) )
     ( i0_temp_aaa(p3_va,h2_oa,h1_oa,u1) += -1 * t2_aaaa(p3_va,p5_va,h1_oa,h6_oa) * i_11_aaa(h6_oa,h2_oa,p5_va,u1) )
     ( i0_temp_aaa(p3_va,h2_oa,h1_oa,u1) += -1 * t2_abab(p3_va,p5_vb,h1_oa,h6_ob) * i_11_bab(h6_ob,h2_oa,p5_vb,u1) )
     ( i0_temp_bab(p3_vb,h2_oa,h1_ob,u1) += -1 * t2_bbbb(p3_vb,p5_vb,h1_ob,h6_ob) * i_11_bab(h6_ob,h2_oa,p5_vb,u1) )
     ( i0_temp_bab(p3_vb,h2_oa,h1_ob,u1) += -1 * t2_abab(p5_va,p3_vb,h6_oa,h1_ob) * i_11_aaa(h6_oa,h2_oa,p5_va,u1) )
     ( i0_temp_bba(p3_vb,h2_ob,h1_oa,u1) +=      t2_abab(p5_va,p3_vb,h1_oa,h6_ob) * i_11_bba(h6_ob,h2_ob,p5_va,u1) )
      
     ( i0_aaa(p3_va,h1_oa,h2_oa,u1) += -1 * i0_temp_aaa(p3_va,h1_oa,h2_oa,u1) )
     ( i0_aaa(p3_va,h2_oa,h1_oa,u1) +=      i0_temp_aaa(p3_va,h1_oa,h2_oa,u1) )
     ( i0_bab(p3_vb,h1_oa,h2_ob,u1) += -1 * i0_temp_bab(p3_vb,h1_oa,h2_ob,u1) )
     ( i0_bab(p3_vb,h2_oa,h1_ob,u1) +=      i0_temp_bba(p3_vb,h1_ob,h2_oa,u1) )
  
     ( i0_aaa(p4_va,h1_oa,h2_oa,u1) +=  x2_aaa(p8_va,h1_oa,h2_oa,u1) * ix2_3_a(p4_va,p8_va) )
     ( i0_bab(p4_vb,h1_oa,h2_ob,u1) +=  x2_bab(p8_vb,h1_oa,h2_ob,u1) * ix2_3_b(p4_vb,p8_vb) )
  
     ( i0_aaa(p3_va,h1_oa,h2_oa,u1) +=  0.5 * x2_aaa(p3_va,h9_oa,h10_oa,u1) * ix2_4_aaaa(h9_oa,h10_oa,h1_oa,h2_oa) ) //O4V
     ( i0_bab(p3_vb,h1_oa,h2_ob,u1) +=        x2_bab(p3_vb,h9_oa,h10_ob,u1) * ix2_4_abab(h9_oa,h10_ob,h1_oa,h2_ob) )
  
     (   i_6_aaa(h10_oa,h1_oa,h2_oa,u1)  = -1 * x1_a(h8_oa,u1) * ix2_4_aaaa(h8_oa,h10_oa,h1_oa,h2_oa) )
     (   i_6_bab(h10_ob,h1_oa,h2_ob,u1)  = -1 * x1_a(h8_oa,u1) * ix2_4_abab(h8_oa,h10_ob,h1_oa,h2_ob) )
      
     (   i_6_aaa(h10_oa,h1_oa,h2_oa,u1) +=  x2_aaa(p5_va,h1_oa,h2_oa,u1) * ix2_6_2_a(h10_oa,p5_va) )
     (   i_6_bab(h10_ob,h1_oa,h2_ob,u1) +=  x2_bab(p5_vb,h1_oa,h2_ob,u1) * ix2_6_2_b(h10_ob,p5_vb) )
      
     (   i_6_temp_aaa(h10_oa,h1_oa,h2_oa,u1)  =  x2_aaa(p9_va,h2_oa,h8_oa,u1) * ix2_6_3_aaaa(h8_oa,h10_oa,h1_oa,p9_va) )
     (   i_6_temp_aaa(h10_oa,h1_oa,h2_oa,u1) +=  x2_bab(p9_vb,h2_oa,h8_ob,u1) * ix2_6_3_baab(h8_ob,h10_oa,h1_oa,p9_vb) ) 
    
     (   i_6_temp_bab(h10_ob,h1_oa,h2_ob,u1)  =  -1 * x2_bab(p9_vb,h8_oa,h2_ob,u1) * ix2_6_3_abab(h8_oa,h10_ob,h1_oa,p9_vb) )
     (   i_6_temp_bba(h10_ob,h1_ob,h2_oa,u1)  =       x2_bab(p9_vb,h2_oa,h8_ob,u1) * ix2_6_3_bbbb(h8_ob,h10_ob,h1_ob,p9_vb) )
     (   i_6_temp_bba(h10_ob,h1_ob,h2_oa,u1) +=       x2_aaa(p9_va,h2_oa,h8_oa,u1) * ix2_6_3_abba(h8_oa,h10_ob,h1_ob,p9_va) )
  
     (   i_6_aaa(h10_oa,h1_oa,h2_oa,u1) += -1 * i_6_temp_aaa(h10_oa,h1_oa,h2_oa,u1) )
     (   i_6_aaa(h10_oa,h2_oa,h1_oa,u1) +=      i_6_temp_aaa(h10_oa,h1_oa,h2_oa,u1) )
     (   i_6_bab(h10_ob,h1_oa,h2_ob,u1) += -1 * i_6_temp_bab(h10_ob,h1_oa,h2_ob,u1) )
     (   i_6_bab(h10_ob,h2_oa,h1_ob,u1) +=      i_6_temp_bba(h10_ob,h1_ob,h2_oa,u1) )
      
     ( i0_aaa(p3_va,h1_oa,h2_oa,u1)  +=  t1_a(p3_va,h10_oa) * i_6_aaa(h10_oa,h1_oa,h2_oa,u1) )
     ( i0_bab(p3_vb,h1_oa,h2_ob,u1)  +=  t1_b(p3_vb,h10_ob) * i_6_bab(h10_ob,h1_oa,h2_ob,u1) )
  
     (   i_10_a(p5_va,u1)  =  0.5 * x2_aaa(p8_va,h6_oa,h7_oa,u1) * v2_aaaa(h6_oa,h7_oa,p5_va,p8_va) )
     (   i_10_a(p5_va,u1) +=        x2_bab(p8_vb,h6_oa,h7_ob,u1) * v2_abab(h6_oa,h7_ob,p5_va,p8_vb) )
     ( i0_aaa(p3_va,h1_oa,h2_oa,u1) +=      t2_aaaa(p3_va,p5_va,h1_oa,h2_oa) * i_10_a(p5_va,u1) )
     ( i0_bab(p3_vb,h1_oa,h2_ob,u1) += -1 * t2_abab(p5_va,p3_vb,h1_oa,h2_ob) * i_10_a(p5_va,u1) )
     .deallocate(i_6_aaa,i_6_bab,
                 i_10_a,i_11_aaa,i_11_bab,i_11_bba,
                 i0_temp_aaa,i0_temp_bab,i0_temp_bba,
                 i_6_temp_aaa,i_6_temp_bab,i_6_temp_bba);
     if(debug) sch.execute();

}

template<typename T>
void gfccsd_driver(ExecutionContext& gec, ExecutionContext& sub_ec, MPI_Comm &subcomm,
                   const TiledIndexSpace& MO, Tensor<T>& t1_a,   Tensor<T>& t1_b, 
                   Tensor<T>& t2_aaaa, Tensor<T>& t2_bbbb, Tensor<T>& t2_abab,
                   Tensor<T>& f1, //Tensor<T>& v2,
                   std::vector<T>& p_evl_sorted_occ, std::vector<T>& p_evl_sorted_virt,
                   long int total_orbitals, const TAMM_SIZE& nocc,const TAMM_SIZE& nvir,
                   size_t& nptsi,
                   const TiledIndexSpace& unit_tis,string files_prefix,string levelstr) {


  using ComplexTensor = Tensor<std::complex<T>>;

  const TiledIndexSpace& O = MO("occ");
  const TiledIndexSpace& V = MO("virt");
  const TiledIndexSpace& N = MO("all");
  auto [u1] = unit_tis.labels<1>("all");

  const int otiles   = O.num_tiles();
  const int vtiles   = V.num_tiles();
  const int oabtiles = otiles/2;
  const int vabtiles = vtiles/2;

  o_alpha = {MO("occ"), range(oabtiles)};
  v_alpha = {MO("virt"),range(vabtiles)};
  o_beta  = {MO("occ"), range(oabtiles,otiles)};
  v_beta  = {MO("virt"),range(vabtiles,vtiles)};

  // auto [p1,p2,p3,p4,p5,p6,p7,p8,p9,p10] = MO.labels<10>("virt");
  // auto [h1,h2,h3,h4,h5,h6,h7,h8,h9,h10] = MO.labels<10>("occ");
  
  // auto [p1,p2] = MO.labels<2>("virt");
  // auto [h1,h2,h3] = MO.labels<3>("occ");

  auto [p1_va] = v_alpha.labels<1>("all");
  auto [p1_vb] = v_beta.labels<1>("all");
  auto [h1_oa,h2_oa] = o_alpha.labels<2>("all");
  auto [h2_ob] = o_beta.labels<1>("all");

  std::cout.precision(15);

  Scheduler gsch{gec};
  auto rank = gec.pg().rank();

  std::stringstream gfo;
  gfo << std::defaultfloat << gf_omega;

  // PRINT THE HEADER FOR GF-CCSD ITERATIONS
  if(rank == 0) {
    std::stringstream gfp;
    gfp << std::endl << std::string(55, '-') << std::endl << "GF-CCSD (omega = " << gfo.str() << ") " << std::endl;
    std::cout << gfp.str() << std::flush;
  }

  ComplexTensor dtmp_aaa{v_alpha,o_alpha,o_alpha};
  ComplexTensor dtmp_bab{v_beta, o_alpha,o_beta};
  ComplexTensor::allocate(&gec,dtmp_aaa,dtmp_bab);

  // double au2ev = 27.2113961;

  std::string dtmp_aaa_file = files_prefix+".W"+gfo.str()+".dtmp_aaa.l"+levelstr;
  std::string dtmp_bab_file = files_prefix+".W"+gfo.str()+".dtmp_bab.l"+levelstr;

  if (fs::exists(dtmp_aaa_file) && fs::exists(dtmp_bab_file)) {
    //read_from_disk(dtmp,dtmpfile);
    read_from_disk(dtmp_aaa,dtmp_aaa_file);
    read_from_disk(dtmp_bab,dtmp_bab_file);
  }
  else {
    double denominator = 0.0;
    const double lshift1 = 0; //0.00000001;
    const double lshift2 = 0.50000000;
    
    ComplexTensor dtmp{V,O,O};
    auto dtmp_lambda = [&](const IndexVector& bid) {
      const IndexVector blockid = internal::translate_blockid(bid, dtmp());
      const TAMM_SIZE size = dtmp.block_size(blockid);
      std::vector<std::complex<T>> buf(size);
      auto block_dims   = dtmp.block_dims(blockid);
      auto block_offset = dtmp.block_offsets(blockid);
      size_t c = 0;
      for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0]; i++) {
          for(size_t j = block_offset[1]; j < block_offset[1] + block_dims[1];
              j++) {
              for(size_t k = block_offset[2]; k < block_offset[2] + block_dims[2];
                  k++, c++) {
                denominator = gf_omega + p_evl_sorted_virt[i] - p_evl_sorted_occ[j] 
                                       - p_evl_sorted_occ[k];
                if (denominator < lshift1 && denominator > -1.0*lshift2){
                    gf_level_shift = -1.0*lshift2;
                }
                else if (denominator > lshift1 && denominator < lshift2) {
                    gf_level_shift = lshift2;
                }
                buf[c] = 1.0/std::complex<T>(denominator + gf_level_shift, gf_eta);
              }
          }
      }
      dtmp.put(blockid,buf);
    };
    
    gsch.allocate(dtmp).execute();

    if(subcomm != MPI_COMM_NULL){
        Scheduler sub_sch{sub_ec};

        sub_sch
        (dtmp()     = 0)
        (dtmp_aaa() = 0)
        (dtmp_bab() = 0)
        .execute();

        block_for(sub_ec, dtmp(), dtmp_lambda);
        
        sub_sch
        (dtmp_aaa(p1_va,h1_oa,h2_oa) = dtmp(p1_va,h1_oa,h2_oa))
        (dtmp_bab(p1_vb,h1_oa,h2_ob) = dtmp(p1_vb,h1_oa,h2_ob))
        .execute();
    }
    gec.pg().barrier();
    gsch.deallocate(dtmp).execute();
  
    write_to_disk(dtmp_aaa,dtmp_aaa_file);
    write_to_disk(dtmp_bab,dtmp_bab_file);
    // if(rank == 0) cout << "Finished [gfccsd] denominator." << std::endl;
  }

  //------------------------
  auto nranks = GA_Nnodes();
  // auto nnodes = GA_Cluster_nnodes();
  // auto ppn = GA_Cluster_nprocs(0);
  
  auto world_comm = gec.pg().comm();
  auto world_rank = gec.pg().rank().value();

  MPI_Group world_group;
  int world_size;
  MPI_Comm_group(world_comm,&world_group);
  MPI_Comm_size(world_comm,&world_size);
  MPI_Comm gf_comm;

  
  //TODO: open shell
  const size_t num_oi = p_oi/2;
  size_t num_pi_processed = 0;
  std::vector<size_t> pi_tbp;
  //Check pi's already processed
  for (size_t pi=0; pi < num_oi; pi++) {
    std::string x1_a_wpi_file   = files_prefix+".x1_a.W"  +gfo.str()+".oi"+std::to_string(pi);
    std::string x2_aaa_wpi_file = files_prefix+".x2_aaa.W"+gfo.str()+".oi"+std::to_string(pi);
    std::string x2_bab_wpi_file = files_prefix+".x2_bab.W"+gfo.str()+".oi"+std::to_string(pi);

    if(fs::exists(x1_a_wpi_file) && fs::exists(x2_aaa_wpi_file) && fs::exists(x2_bab_wpi_file)) 
      num_pi_processed++;
    else pi_tbp.push_back(pi);
  }

  //TODO:if pi's remaining are very few, throttle subranks even further
  size_t num_pi_remain = num_oi-num_pi_processed;
  if(num_pi_remain == 0) num_pi_remain = 1;
  int subranks = std::floor(nranks/num_pi_remain);
  const bool no_pg=(subranks == 0 || subranks == 1);
  if(no_pg) subranks=nranks;
  if(gf_nprocs_poi > 0) subranks = gf_nprocs_poi;

  if(rank==0) cout << "No of processes used to compute each orbital index = " << subranks << endl;

  int color = 0;
  if(subranks > 1) color = world_rank/subranks;  
  
  MPI_Comm_split(world_comm, color, world_rank, &gf_comm);

  //##########################
  //  MAIN ITERATION LOOP 
  //##########################
  auto cc_t1 = std::chrono::high_resolution_clock::now();
  // auto pi_a  = 0;
  // auto pi_b  = 0;
  // auto spin  = 0;
  #if GF_PGROUPS
    ProcGroup pg{gf_comm};
    auto mgr = MemoryManagerGA::create_coll(pg);
    Distribution_NW distribution;
    RuntimeEngine re;
    ExecutionContext ec{pg, &distribution, mgr, &re};
    Scheduler sch{ec};
  #else
      Scheduler& sch = gsch;
      ExecutionContext& ec = gec;    
  #endif

  int64_t root_ppi = 0;
  for (size_t piv=0; piv < pi_tbp.size(); piv++){
    size_t pi = pi_tbp[piv];
    #if GF_PGROUPS
    if( (rank >= piv*subranks && rank < (piv*subranks+subranks) ) || no_pg){
    if(!no_pg) root_ppi = piv*subranks; //root of sub-group
    #endif

    auto gf_t1 = std::chrono::high_resolution_clock::now();
    
    bool gf_conv = false;
    ComplexTensor Hx1_a{o_alpha,unit_tis};
    ComplexTensor Hx2_aaa{v_alpha,o_alpha,o_alpha,unit_tis};
    ComplexTensor Hx2_bab{v_beta, o_alpha,o_beta, unit_tis};
    ComplexTensor d1_a{o_alpha,unit_tis};
    ComplexTensor d2_aaa{v_alpha,o_alpha,o_alpha,unit_tis};
    ComplexTensor d2_bab{v_beta, o_alpha,o_beta, unit_tis};
    ComplexTensor Minv_a{o_alpha,o_alpha};
    ComplexTensor Minv_b{o_beta, o_beta};
    ComplexTensor dx1_a{o_alpha,unit_tis};
    ComplexTensor dx2_aaa{v_alpha,o_alpha,o_alpha,unit_tis};
    ComplexTensor dx2_bab{v_beta, o_alpha,o_beta, unit_tis};
    ComplexTensor Minv{{O,O},{1,1}};
    ComplexTensor x1{O,unit_tis};
    //ComplexTensor x2{V,O,O,unit_tis};
    ComplexTensor x1_a{o_alpha,unit_tis};
    ComplexTensor x2_aaa{v_alpha,o_alpha,o_alpha,unit_tis};
    ComplexTensor x2_bab{v_beta, o_alpha,o_beta, unit_tis};
    Tensor<T> B1{O,unit_tis};
    Tensor<T> B1_a{o_alpha,unit_tis};
    Tensor<T> B1_b{o_beta, unit_tis};
  
    sch.allocate( 
                          Hx1_a, Hx2_aaa, Hx2_bab,
                          d1_a,  d2_aaa,  d2_bab,
                          dx1_a, dx2_aaa, dx2_bab,
                          Minv, x1, x1_a,x2_aaa,x2_bab,
                          Minv_a,Minv_b,B1,B1_a,B1_b).execute();

      // // a function to return spin, sub_index in spin array for orbital pi
      // if(pi<noa){
      //   pi_a = pi;
      // }
      // else{
      //   pi_b = pi - noa;
      //   spin = 1;
      // }

    // std::vector<ComplexTensor> e1(ndiis); //O
    // gf_populate_vector_of_tensors(e1,false);
    // std::vector<ComplexTensor> e2(ndiis); //VOO
    // gf_populate_vector_of_tensors(e2);

    // std::vector<ComplexTensor> xx1(ndiis); //O
    // gf_populate_vector_of_tensors(xx1,false);
    // std::vector<ComplexTensor> xx2(ndiis); //VOO
    // gf_populate_vector_of_tensors(xx2);  


      // sch(x1() = 0)
      // (x2() = 0)
      // .execute();

    int total_iter = 0;
    double gf_t_guess = 0.0;
    double gf_t_x1_tot  = 0.0;
    double gf_t_x2_tot  = 0.0;
    double gf_t_res_tot = 0.0;
    double gf_t_res_tot_1 = 0.0;
    double gf_t_res_tot_2 = 0.0;
    double gf_t_upd_tot = 0.0;
    double gf_t_dis_tot = 0.0;

    std::string x1_a_wpi_file = files_prefix+".x1_a.W"+gfo.str()+".oi"+std::to_string(pi);
    std::string x2_aaa_wpi_file = files_prefix+".x2_aaa.W"+gfo.str()+".oi"+std::to_string(pi);
    std::string x2_bab_wpi_file = files_prefix+".x2_bab.W"+gfo.str()+".oi"+std::to_string(pi);

    if(! (fs::exists(x1_a_wpi_file) && fs::exists(x2_aaa_wpi_file) && fs::exists(x2_bab_wpi_file)) ) {
      //DIIS
      // ComplexTensor e1{O,diis_tis};
      // ComplexTensor e2{V,O,O,diis_tis};
      // ComplexTensor xx1{O,diis_tis};
      // ComplexTensor xx2{V,O,O,diis_tis};

      // sch
      //   .allocate(e1,e2,xx1,xx2)
      //   (Hx1()= 0) 
      //   (Hx2()= 0) 
      //   (d1() = 0)
      //   (d2() = 0).execute();

      ComplexTensor e1_a{o_alpha,diis_tis};
      ComplexTensor e2_aaa{v_alpha,o_alpha,o_alpha,diis_tis};
      ComplexTensor e2_bab{v_beta, o_alpha,o_beta, diis_tis};
      ComplexTensor xx1_a{o_alpha,diis_tis};
      ComplexTensor xx2_aaa{v_alpha,o_alpha,o_alpha,diis_tis};
      ComplexTensor xx2_bab{v_beta, o_alpha,o_beta, diis_tis};
  
      sch
        .allocate( e1_a, e2_aaa, e2_bab,
                  xx1_a,xx2_aaa,xx2_bab)
        (Hx1_a()   = 0)
        (Hx2_aaa() = 0) 
        (Hx2_bab() = 0)
        (d1_a()    = 0)
        (d2_aaa()  = 0) 
        (d2_bab()  = 0)
        (x1_a()    = 0)
        (x2_aaa()  = 0) 
        (x2_bab()  = 0)
        (Minv_a()  = 0)
        
        (x1()      = 0)
        //(x2()      = 0) 
        (Minv()    = 0)
        .execute();

      LabelLoopNest loop_nest{B1().labels()};
      sch(B1() = 0).execute();
      
       
      for(const IndexVector& bid : loop_nest) {
        const IndexVector blockid = internal::translate_blockid(bid, B1());
        
        const TAMM_SIZE size = B1.block_size(blockid);
        std::vector<TensorType> buf(size);
        B1.get(blockid, buf);
        auto block_dims   = B1.block_dims(blockid);
        auto block_offset = B1.block_offsets(blockid);
        auto dim          = block_dims[0];
        auto offset       = block_offset[0];
        if(pi >= offset && pi < offset+dim)
          buf[pi-offset] = 1.0;
        B1.put(blockid,buf);
      }

      sch
        (B1_a(h1_oa,u1) = B1(h1_oa,u1))
        // (B1_b(h1_ob,u1) = B1(h1_ob,u1))
        .execute();
      ec.pg().barrier();

      auto gf_t_guess1 = std::chrono::high_resolution_clock::now();

      gf_guess(ec, MO, nocc,gf_omega,gf_eta,pi,p_evl_sorted_occ,t2v2,x1,Minv);

      sch
      (x1_a(h1_oa,u1) = x1(h1_oa,u1))
      (Minv_a(h1_oa,h2_oa)  = Minv(h1_oa,h2_oa))
      .execute();

      auto gf_t_guess2 = std::chrono::high_resolution_clock::now();
      gf_t_guess = std::chrono::duration_cast<std::chrono::duration<double>>((gf_t_guess2 - gf_t_guess1)).count();

      for(size_t iter = 0; iter < gf_maxiter; iter += ndiis) {

        for(size_t micro = iter; micro < std::min(iter+ndiis,gf_maxiter); micro++){

          total_iter = micro;
          auto gf_t_ini = std::chrono::high_resolution_clock::now();

          gfccsd_x1(sch, MO, Hx1_a,
                    t1_a, t1_b, t2_aaaa, t2_bbbb, t2_abab, 
                    x1_a, x2_aaa, x2_bab, 
                    f1, //v2, 
                    unit_tis);

          auto gf_t_x1 = std::chrono::high_resolution_clock::now();
          gf_t_x1_tot += std::chrono::duration_cast<std::chrono::duration<double>>((gf_t_x1 - gf_t_ini)).count();

          gfccsd_x2(sch, MO, Hx2_aaa, Hx2_bab, 
                    t1_a, t1_b, t2_aaaa, t2_bbbb, t2_abab, 
                    x1_a, x2_aaa, x2_bab, 
                    f1, //v2, 
                    unit_tis);

          auto gf_t_x2 = std::chrono::high_resolution_clock::now();
          gf_t_x2_tot += std::chrono::duration_cast<std::chrono::duration<double>>((gf_t_x2 - gf_t_x1)).count();

          sch 
            (d1_a() = -1.0 * Hx1_a())
            (d1_a(h1_oa,u1) -= std::complex<double>(gf_omega,gf_eta) * x1_a(h1_oa,u1))
            (d1_a() += B1_a())
            (d2_aaa() = -1.0 * Hx2_aaa())
            (d2_aaa(p1_va,h1_oa,h2_oa,u1) -= std::complex<double>(gf_omega,gf_eta) * x2_aaa(p1_va,h1_oa,h2_oa,u1))
            (d2_bab() = -1.0 * Hx2_bab())
            (d2_bab(p1_vb,h1_oa,h2_ob,u1) -= std::complex<double>(gf_omega,gf_eta) * x2_bab(p1_vb,h1_oa,h2_ob,u1))
            .execute();
  
          auto gf_t_res_1 = std::chrono::high_resolution_clock::now();
          gf_t_res_tot_1 += std::chrono::duration_cast<std::chrono::duration<double>>((gf_t_res_1 - gf_t_x2)).count();

          auto d1_a_norm   = norm(d1_a);
          auto d2_aaa_norm = norm(d2_aaa);
          auto d2_bab_norm = norm(d2_bab);

          auto gf_t_res_2 = std::chrono::high_resolution_clock::now();
          gf_t_res_tot_2 += std::chrono::duration_cast<std::chrono::duration<double>>((gf_t_res_2 - gf_t_res_1)).count();
 
          double gf_residual 
                 = sqrt(d1_a_norm*d1_a_norm + 
                        d2_aaa_norm*d2_aaa_norm + 
                        d2_bab_norm*d2_bab_norm).real();

          if(debug && rank==root_ppi)
             cout << std::defaultfloat << "W,oi (" << gfo.str() << "," << pi << "): #iter " << total_iter << ", residual = "
               << gf_residual << std::endl;

          if(gf_residual < gf_threshold) {
            gf_conv = true;
            break;
          }

          auto gf_t_res = std::chrono::high_resolution_clock::now();
          gf_t_res_tot += std::chrono::duration_cast<std::chrono::duration<double>>((gf_t_res - gf_t_x2)).count();
        
          //JACOBI
          auto gindx=micro-iter;

          TiledIndexSpace hist_tis{diis_tis,range(gindx,gindx+1)};
          auto [dh1] = hist_tis.labels<1>("all");
          
          sch
            (dx1_a(h1_oa,u1) = Minv_a(h1_oa,h2_oa) * d1_a(h2_oa,u1))
            (dx2_aaa(p1_va,h1_oa,h2_oa,u1) = dtmp_aaa(p1_va,h1_oa,h2_oa) * d2_aaa(p1_va,h1_oa,h2_oa,u1))
            (dx2_bab(p1_vb,h1_oa,h2_ob,u1) = dtmp_bab(p1_vb,h1_oa,h2_ob) * d2_bab(p1_vb,h1_oa,h2_ob,u1))
            (x1_a(h1_oa,u1) += gf_damping_factor * dx1_a(h1_oa,u1))
            (x2_aaa(p1_va,h1_oa,h2_oa,u1) += gf_damping_factor * dx2_aaa(p1_va,h1_oa,h2_oa,u1))
            (x2_bab(p1_vb,h1_oa,h2_ob,u1) += gf_damping_factor * dx2_bab(p1_vb,h1_oa,h2_ob,u1))
            (e1_a(h1_oa,dh1) = d1_a(h1_oa,u1))
            (e2_aaa(p1_va,h1_oa,h2_oa,dh1) = d2_aaa(p1_va,h1_oa,h2_oa,u1)) 
            (e2_bab(p1_vb,h1_oa,h2_ob,dh1) = d2_bab(p1_vb,h1_oa,h2_ob,u1)) 
            (xx1_a(h1_oa,dh1) = x1_a(h1_oa,u1))
            (xx2_aaa(p1_va,h1_oa,h2_oa,dh1) = x2_aaa(p1_va,h1_oa,h2_oa,u1)) 
            (xx2_bab(p1_vb,h1_oa,h2_ob,dh1) = x2_bab(p1_vb,h1_oa,h2_ob,u1)) 
            .execute();

          auto gf_t_upd = std::chrono::high_resolution_clock::now();
          gf_t_upd_tot += std::chrono::duration_cast<std::chrono::duration<double>>((gf_t_upd - gf_t_res)).count();
        
        } //end micro

        if(gf_conv || iter + ndiis >= gf_maxiter) { break; }

        auto gf_t_tmp = std::chrono::high_resolution_clock::now();

        //DIIS
        gf_diis(ec,MO,ndiis,
            e1_a, e2_aaa, e2_bab,
            xx1_a,xx2_aaa,xx2_bab,
            x1_a, x2_aaa, x2_bab,
            diis_tis,unit_tis);

        auto gf_t_dis = std::chrono::high_resolution_clock::now();
        gf_t_dis_tot += std::chrono::duration_cast<std::chrono::duration<double>>((gf_t_dis - gf_t_tmp)).count();   
      } //end iter

      write_to_disk(x1_a,x1_a_wpi_file);
      write_to_disk(x2_aaa,x2_aaa_wpi_file);
      write_to_disk(x2_bab,x2_bab_wpi_file);   

      // sch.deallocate(e1,e2,xx1,xx2).execute();
      sch.deallocate( e1_a, e2_aaa, e2_bab,
                     xx1_a,xx2_aaa,xx2_bab)
                    .execute();

    }
    else {
      gf_conv = true;
    }

    if(!gf_conv && rank==root_ppi) cout << "ERROR: GF-CCSD does not converge for W,oi = " << gfo.str() << "," << pi << std::endl;
      
    auto gf_t2 = std::chrono::high_resolution_clock::now();
    double gftime =
      std::chrono::duration_cast<std::chrono::duration<double>>((gf_t2 - gf_t1)).count();
    if(rank == root_ppi) {
      std::stringstream gf_stats;
      gf_stats << std::defaultfloat << "GF-CCSD Time for W,oi (" << gfo.str() << "," << pi << ") = " 
               << gftime << " secs, #iter = " << total_iter << std::endl;
      
      if(debug){
        gf_stats << "|----------initial guess  : " << gf_t_guess   << std::endl;
        gf_stats << "|----------x1 contraction : " << gf_t_x1_tot  << std::endl;
        gf_stats << "|----------x2 contraction : " << gf_t_x2_tot  << std::endl;
        gf_stats << "|----------computing res. : " << gf_t_res_tot << std::endl;
        gf_stats << "           |----------misc. contr. : " << gf_t_res_tot_1 << std::endl;
        gf_stats << "           |----------compt. norm  : " << gf_t_res_tot_2 << std::endl;
        gf_stats << "|----------updating x1/x2 : " << gf_t_upd_tot << std::endl;
        gf_stats << "|----------diis update    : " << gf_t_dis_tot << std::endl;
      }
      std::cout << gf_stats.str();
    }

    sch.deallocate(Hx1_a, Hx2_aaa, Hx2_bab,
                  d1_a,  d2_aaa,  d2_bab,
                  dx1_a, dx2_aaa, dx2_bab,
                  Minv, x1, x1_a,x2_aaa,x2_bab,
                  B1, B1_a, B1_b,Minv_a,Minv_b).execute();

    #if GF_PGROUPS
    }
    #endif
  } //end pi

  #if GF_PGROUPS
    ec.flush_and_sync();
    MemoryManagerGA::destroy_coll(mgr);
  #endif
  gec.pg().barrier();

  auto cc_t2 = std::chrono::high_resolution_clock::now();
  double time =
    std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
  if(rank == 0) {
    std::cout << "Total GF-CCSD Time (W = " << gfo.str() << ") = " << time << " secs" << std::endl;
    std::cout << std::string(55, '-') << std::endl;
  }

  // sch.deallocate(B1,Hx1,Hx2,d1,d2,dx1,dx2,Minv,tdtmp).execute();
  gsch.deallocate(dtmp_aaa,dtmp_bab).execute();
  MPI_Comm_free(&gf_comm);
}

void gfccsd_main_driver(std::string filename) {

    // std::cout << "Input file provided = " << filename << std::endl;

    using T = double;

    ProcGroup pg{GA_MPI_Comm()};
    auto mgr = MemoryManagerGA::create_coll(pg);
    Distribution_NW distribution;
    RuntimeEngine re;
    ExecutionContext ec{pg, &distribution, mgr, &re};
    auto rank = ec.pg().rank();

    ProcGroup pg_l{MPI_COMM_SELF};
    auto mgr_l = MemoryManagerLocal::create_coll(pg_l);
    Distribution_NW distribution_l;
    RuntimeEngine re_l;
    ExecutionContext ec_l{pg_l, &distribution_l, mgr_l, &re_l};

    //TODO: read from input file, assume no freezing for now
    TAMM_SIZE freeze_core    = 0;
    TAMM_SIZE freeze_virtual = 0;

    auto [options_map, ov_alpha, nao, hf_energy, shells, shell_tile_map, C_AO, F_AO, AO_opt, AO_tis, scf_conv] 
                    = hartree_fock_driver<T>(ec,filename);

    int nsranks = nao/15;
    int ga_cnn = GA_Cluster_nnodes();
    if(nsranks>ga_cnn) nsranks=ga_cnn;
    nsranks = nsranks * GA_Cluster_nprocs(0);
    int subranks[nsranks];
    for (int i = 0; i < nsranks; i++) subranks[i] = i;
    auto world_comm = ec.pg().comm();
    MPI_Group world_group;
    MPI_Comm_group(world_comm,&world_group);
    MPI_Group subgroup;
    MPI_Group_incl(world_group,nsranks,subranks,&subgroup);
    MPI_Comm subcomm;
    MPI_Comm_create(world_comm,subgroup,&subcomm);
    
    ProcGroup *sub_pg=nullptr;
    MemoryManagerGA *sub_mgr=nullptr;
    Distribution_NW *sub_distribution=nullptr;
    RuntimeEngine *sub_re=nullptr;
    ExecutionContext *sub_ec=nullptr;

    if(subcomm != MPI_COMM_NULL){
        sub_pg = new ProcGroup{subcomm};
        sub_mgr = MemoryManagerGA::create_coll(*sub_pg);
        sub_distribution = new Distribution_NW();
        sub_re = new RuntimeEngine();
        sub_ec = new ExecutionContext(*sub_pg, sub_distribution, sub_mgr, sub_re);
    }

    Scheduler sub_sch{*sub_ec};

    CCSDOptions ccsd_options = options_map.ccsd_options;
    debug = ccsd_options.debug;
    if(rank == 0) ccsd_options.print();

    int maxiter    = ccsd_options.ccsd_maxiter;
    double thresh  = ccsd_options.threshold;
    double zshiftl = 0.0;
    ndiis   = 5;

    auto [MO,total_orbitals] = setupMOIS(ccsd_options.tilesize,
                    nao,ov_alpha,freeze_core,freeze_virtual);

    std::string out_fp = getfilename(filename)+"."+ccsd_options.basis;
    std::string files_dir = out_fp+"_files";
    std::string files_prefix = /*out_fp;*/ files_dir+"/"+out_fp;
    std::string f1file = files_prefix+".f1_mo";
    std::string t1file = files_prefix+".t1amp";
    std::string t2file = files_prefix+".t2amp";
    std::string v2file = files_prefix+".cholv2";
    std::string cholfile = files_prefix+".cholcount";
    std::string ccsdstatus = files_prefix+".ccsdstatus";

    bool ccsd_restart = ccsd_options.readt || 
        ( (fs::exists(t1file) && fs::exists(t2file)     
        && fs::exists(f1file) && fs::exists(v2file)) );
    
    //deallocates F_AO, C_AO
    auto [cholVpr,d_f1,chol_count, max_cvecs, CI] = cd_svd_ga_driver<T>
                    (options_map, ec, MO, AO_opt, ov_alpha, nao, freeze_core,
                     freeze_virtual, C_AO, F_AO, shells, shell_tile_map,
                     ccsd_restart, cholfile);

    TiledIndexSpace N = MO("all");

    auto [p_evl_sorted,d_t1,d_t2,d_r1,d_r2, d_r1s, d_r2s, d_t1s, d_t2s] 
            = setupTensors(ec,MO,d_f1,ndiis,ccsd_restart && fs::exists(ccsdstatus) && scf_conv);

    if(ccsd_restart) {
        read_from_disk(d_f1,f1file);
        read_from_disk(d_t1,t1file);
        read_from_disk(d_t2,t2file);
        read_from_disk(cholVpr,v2file);
        ec.pg().barrier();
        p_evl_sorted = tamm::diagonal(d_f1);
    }

    else if(ccsd_options.writet) {
        // fs::remove_all(files_dir); 
        if(!fs::exists(files_dir)) fs::create_directories(files_dir);

        write_to_disk(d_f1,f1file);
        write_to_disk(cholVpr,v2file);

        if(rank==0){
          std::ofstream out(cholfile, std::ios::out);
          if(!out) cerr << "Error opening file " << cholfile << endl;
          out << chol_count << std::endl;
          out.close();
        }        
    }

    if(rank==0 && debug){
      cout << "eigen values:" << endl << std::string(50,'-') << endl;
      for (auto x: p_evl_sorted) cout << x << endl;
      cout << std::string(50,'-') << endl;
    }
    
    auto cc_t1 = std::chrono::high_resolution_clock::now();

    ccsd_restart = ccsd_restart && fs::exists(ccsdstatus) && scf_conv;

    double residual=0, corr_energy=0;
    if(ccsd_restart){
      
      if(subcomm != MPI_COMM_NULL){
          std::tie(residual, corr_energy) = cd_ccsd_driver<T>(
                  *sub_ec, MO, CI, d_t1, d_t2, d_f1, 
                  d_r1,d_r2, d_r1s, d_r2s, d_t1s, d_t2s, 
                  p_evl_sorted, 
                  maxiter, thresh, zshiftl, ndiis, 
                  2 * ov_alpha, cholVpr, ccsd_options.writet, ccsd_restart, files_prefix);
      }
      ec.pg().barrier();
    }
    else{
      std::tie(residual, corr_energy) = cd_ccsd_driver<T>(
                ec, MO, CI, d_t1, d_t2, d_f1, 
                d_r1,d_r2, d_r1s, d_r2s, d_t1s, d_t2s, 
                p_evl_sorted, 
                maxiter, thresh, zshiftl, ndiis, 
                2 * ov_alpha, cholVpr, ccsd_options.writet, ccsd_restart, files_prefix);
    }

    ccsd_stats(ec, hf_energy,residual,corr_energy,thresh);

    if(ccsd_options.writet && !fs::exists(ccsdstatus)) {
        write_to_disk(d_t1,t1file);
        write_to_disk(d_t2,t2file);
        if(rank==0){
          std::ofstream out(ccsdstatus, std::ios::out);
          if(!out) cerr << "Error opening file " << ccsdstatus << endl;
          out << 1 << std::endl;
          out.close();
        }          
    }

    auto cc_t2 = std::chrono::high_resolution_clock::now();
    double ccsd_time = 
        std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
    if(rank == 0) std::cout << std::endl << "Time taken for Cholesky CCSD: " << ccsd_time << " secs" << std::endl;

    if(!ccsd_restart) {
        free_tensors(d_r1,d_r2);
        free_vec_tensors(d_r1s, d_r2s, d_t1s, d_t2s);
    }

    ec.flush_and_sync();

    // Tensor<T> d_v2 = setupV2<T>(ec,MO,CI,cholVpr,chol_count, total_orbitals, ov_alpha, 
    //                             nao - ov_alpha);
    // Tensor<T>::deallocate(cholVpr);


    #if DO_LAMBDA
      auto [l_r1,l_r2,d_y1,d_y2,l_r1s,l_r2s,d_y1s,d_y2s] = setupLambdaTensors<T>(ec,MO,ndiis);

      cc_t1 = std::chrono::high_resolution_clock::now();
      std::tie(residual,corr_energy) = lambda_ccsd_driver<T>(ec, MO, d_t1, d_t2, d_f1,
                            d_v2, l_r1,l_r2,d_y1, d_y2,l_r1s,l_r2s, d_y1s,d_y2s,
                            p_evl_sorted, maxiter, thresh, zshiftl, ndiis,
                            2 * ov_alpha);
      cc_t2 = std::chrono::high_resolution_clock::now();

      if(rank == 0) {
        std::cout << std::string(66, '-') << std::endl;
        if(residual < thresh) {
            std::cout << " Iterations converged" << std::endl;
        }
      }

      ccsd_time =
        std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
      if(rank == 0) std::cout << std::endl << "Time taken for Lambda CCSD: " << ccsd_time << " secs" << std::endl; 

      free_tensors(l_r1,l_r2);
      free_vec_tensors(l_r1s,l_r2s,d_y1s,d_y2s);

      ec.flush_and_sync();
    
      // lambda_y1 = d_y1;
      // lambda_y2 = d_y2;
    #endif

  //GFCCSD Routine
  cc_t1 = std::chrono::high_resolution_clock::now();

  const TAMM_SIZE nocc = 2 * ov_alpha;
  const TAMM_SIZE nvir = 2*nao - 2*ov_alpha;
   //TODO, need to get noa and nob from other places.
  const TAMM_SIZE noa  = nocc/2;
  // const TAMM_SIZE nob  = nocc/2;
  // gf_omega     = ccsd_options.gf_omega; //a.u (range min to max)
  ndiis                = ccsd_options.gf_ndiis;
  gf_eta               = ccsd_options.gf_eta;
  gf_maxiter           = ccsd_options.gf_maxiter;
  gf_threshold         = ccsd_options.gf_threshold;
  omega_min            = ccsd_options.gf_omega_min;
  omega_max            = ccsd_options.gf_omega_max;
  lomega_min           = ccsd_options.gf_omega_min_e;
  lomega_max           = ccsd_options.gf_omega_max_e;
  omega_delta          = ccsd_options.gf_omega_delta;
  omega_delta_e        = ccsd_options.gf_omega_delta_e;
  gf_nprocs_poi        = ccsd_options.gf_nprocs_poi;  
  gf_level_shift       = 0;
  gf_damping_factor    = ccsd_options.gf_damping_factor;
  gf_extrapolate_level = ccsd_options.gf_extrapolate_level;
  omega_npts   = (omega_max - omega_min) / omega_delta + 1;
  lomega_npts  = (lomega_max - lomega_min) / omega_delta_e + 1;

  const int gf_p_oi = ccsd_options.gf_p_oi_range;

  if(gf_p_oi == 1) p_oi = nocc;
  else p_oi = nocc+nvir;

  int level = 1;

  if(rank == 0) ccsd_options.print();
  
  using ComplexTensor = Tensor<std::complex<T>>;

  TiledIndexSpace o_alpha,v_alpha,o_beta,v_beta;

  const TiledIndexSpace &O = MO("occ");
  const TiledIndexSpace &V = MO("virt");
  
  const int otiles   = O.num_tiles();
  const int vtiles   = V.num_tiles();
  const int oabtiles = otiles/2;
  const int vabtiles = vtiles/2;

  o_alpha = {MO("occ"), range(oabtiles)};
  v_alpha = {MO("virt"),range(vabtiles)};
  o_beta  = {MO("occ"), range(oabtiles,otiles)};
  v_beta  = {MO("virt"),range(vabtiles,vtiles)};

  auto [p1,p2,p3,p4,p5,p6,p7,p8,p9] = MO.labels<9>("virt");
  auto [h1,h2,h3,h4,h5,h6,h7,h8,h9,h10] = MO.labels<10>("occ");
  
  auto [cind] = CI.labels<1>("all");

  auto [p1_va,p2_va] = v_alpha.labels<2>("all");
  auto [p1_vb,p2_vb] = v_beta.labels<2>("all");
  auto [h1_oa,h2_oa,h3_oa,h4_oa] = o_alpha.labels<4>("all");
  auto [h1_ob,h2_ob,h3_ob,h4_ob] = o_beta.labels<4>("all");

  Scheduler sch{ec};

  if(rank==0) cout << endl << "#occupied, #virtual = " << nocc << ", " << nvir << endl;
  std::vector<T> p_evl_sorted_occ(nocc);
  std::vector<T> p_evl_sorted_virt(nvir);
  std::copy(p_evl_sorted.begin(), p_evl_sorted.begin() + nocc,
            p_evl_sorted_occ.begin());
  std::copy(p_evl_sorted.begin() + nocc, p_evl_sorted.end(),
            p_evl_sorted_virt.begin());

  //START SF2

  std::vector<T> omega_space;
  for(int64_t ni=0;ni<omega_npts;ni++) {
    T omega_tmp =  omega_min + ni*omega_delta;
    omega_space.push_back(omega_tmp);
  }
  // cout << "Freq. space (before doing MOR): " << omega_npts<< "," << omega_delta <<"," << omega_space << endl;


  #ifdef SF2
    TiledIndexSpace sftis{IndexSpace{range(omega_npts)}};
    auto [sftil] = sftis.labels<1>("all");
    Tensor<T> sf2{O,O,sftis};
    Tensor<T> tmpsf2{O,O,V,V,sftis};
    Tensor<T> sf2I{V,V,O,sftis};
    Tensor<T>::allocate(&ec,sf2,sf2I,tmpsf2);

    sch(sf2I()=0).execute();
    auto sf2I_lambda = [&](const IndexVector& bid) {
      const IndexVector blockid = internal::translate_blockid(bid, sf2I());
      const TAMM_SIZE size = sf2I.block_size(blockid);
      std::vector<T> buf(size);
      auto block_dims   = sf2I.block_dims(blockid);
      auto block_offset = sf2I.block_offsets(blockid);
      size_t c = 0;
      for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0]; i++) {
          for(size_t j = block_offset[1]; j < block_offset[1] + block_dims[1];
              j++) {
              for(size_t k = block_offset[2]; k < block_offset[2] + block_dims[2];
                  k++) {
                for(size_t w = block_offset[3]; w < block_offset[3] + block_dims[3];
                  w++, c++) {
                buf[c] = 1.0 / (-p_evl_sorted_virt[i] - p_evl_sorted_virt[j]+p_evl_sorted_occ[k]+omega_space[w]);
                  }
              }
          }
      }
      sf2I.put(blockid,buf);
    };
    block_for(ec, sf2I(), sf2I_lambda);

    Scheduler{ec}
    (tmpsf2(h1,h2,p1,p2,sftil) = d_v2(h1,h2,p1,p2) * sf2I(p1,p2,h1,sftil))
    (sf2(h2,h3,sftil) = 0.5 * tmpsf2(h1,h2,p1,p2,sftil) * d_v2(h1,h3,p1,p2)).execute();

    //TODO: Use sf2 and deallocate later
    Tensor<T>::deallocate(sf2,sf2I,tmpsf2);
  #endif

  #define MOR 1
  #if MOR

    using Complex2DMatrix=Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    size_t nptsi = 0;
    
    std::vector<bool> omega_conv(omega_npts,false);
    std::vector<T> omega_A0(omega_npts,UINT_MAX);

    std::vector<T> omega_extra;
    std::vector<T> omega_extra_finished;

    // std::vector<ComplexTensor> x1_omega_a(omega_npts*p_oi/2);
    // std::vector<ComplexTensor> x2_omega_aaa(omega_npts*p_oi/2);
    // std::vector<ComplexTensor> x2_omega_bab(omega_npts*p_oi/2);

    //TODO, figure out the spin explicit parts of these intermediates
    std::string t2v2file        = files_prefix+".t2v2";
    std::string lt12afile       = files_prefix+".lt12_a";
    std::string v2ijabfile      = files_prefix+".v2ijab";
    // std::string ix1_1afile      = files_prefix+".ix1_1_a";
    std::string ix1_1_1afile    = files_prefix+".ix1_1_1_a";
    std::string ix1_1_1bfile    = files_prefix+".ix1_1_1_b";
    // std::string ix1_4aaaafile   = files_prefix+".ix1_4_aaaa";
    // std::string ix1_4ababfile   = files_prefix+".ix1_4_abab";
    std::string ix2_1aaaafile   = files_prefix+".ix2_1_aaaa";
    std::string ix2_1ababfile   = files_prefix+".ix2_1_abab";
    std::string ix2_2afile      = files_prefix+".ix2_2_a";
    std::string ix2_2bfile      = files_prefix+".ix2_2_b";
    std::string ix2_3afile      = files_prefix+".ix2_3_a";
    std::string ix2_3bfile      = files_prefix+".ix2_3_b";
    std::string ix2_4aaaafile   = files_prefix+".ix2_4_aaaa";
    std::string ix2_4ababfile   = files_prefix+".ix2_4_abab";
    std::string ix2_5aaaafile   = files_prefix+".ix2_5_aaaa";
    std::string ix2_5abbafile   = files_prefix+".ix2_5_abba";
    std::string ix2_5ababfile   = files_prefix+".ix2_5_abab";
    std::string ix2_5bbbbfile   = files_prefix+".ix2_5_bbbb";
    std::string ix2_5baabfile   = files_prefix+".ix2_5_baab";
    std::string ix2_6_2afile    = files_prefix+".ix2_6_2_a";
    std::string ix2_6_2bfile    = files_prefix+".ix2_6_2_b";
    std::string ix2_6_3aaaafile = files_prefix+".ix2_6_3_aaaa";
    std::string ix2_6_3abbafile = files_prefix+".ix2_6_3_abba";
    std::string ix2_6_3ababfile = files_prefix+".ix2_6_3_abab";
    std::string ix2_6_3baabfile = files_prefix+".ix2_6_3_baab";
    std::string ix2_6_3bbbbfile = files_prefix+".ix2_6_3_bbbb";
    
    t2v2 = Tensor<T>{{O,O},{1,1}};

    d_t1_a       = Tensor<T>{{v_alpha,o_alpha},{1,1}};
    d_t1_b       = Tensor<T>{{v_beta, o_beta}, {1,1}};
    d_t2_aaaa    = Tensor<T>{{v_alpha,v_alpha,o_alpha,o_alpha},{2,2}};
    d_t2_bbbb    = Tensor<T>{{v_beta, v_beta, o_beta, o_beta}, {2,2}};
    d_t2_abab    = Tensor<T>{{v_alpha,v_beta, o_alpha,o_beta}, {2,2}};
    v2_aaaa      = Tensor<T>{{o_alpha,o_alpha,v_alpha,v_alpha},{2,2}};
    v2_bbbb      = Tensor<T>{{o_beta, o_beta, v_beta, v_beta}, {2,2}};
    v2_abab      = Tensor<T>{{o_alpha,o_beta, v_alpha,v_beta}, {2,2}};
    lt12_a       = Tensor<T>{{o_alpha,o_alpha},{1,1}};
    // ix1_1_a      = Tensor<T>{{o_alpha,o_alpha},{1,1}};
    ix1_1_1_a    = Tensor<T>{{o_alpha,v_alpha},{1,1}};
    ix1_1_1_b    = Tensor<T>{{o_beta, v_beta}, {1,1}};
    // ix1_4_aaaa   = Tensor<T>{{o_alpha,o_alpha,o_alpha,v_alpha},{2,2}};
    // ix1_4_abab   = Tensor<T>{{o_alpha,o_beta, o_alpha,v_beta}, {2,2}};
    ix2_1_aaaa   = Tensor<T>{{o_alpha,v_alpha,o_alpha,o_alpha},{2,2}};
    ix2_1_abab   = Tensor<T>{{o_alpha,v_beta, o_alpha,o_beta}, {2,2}};
    ix2_2_a      = Tensor<T>{{o_alpha,o_alpha},{1,1}};
    ix2_2_b      = Tensor<T>{{o_beta, o_beta}, {1,1}};
    ix2_3_a      = Tensor<T>{{v_alpha,v_alpha},{1,1}};
    ix2_3_b      = Tensor<T>{{v_beta, v_beta}, {1,1}};
    ix2_4_aaaa   = Tensor<T>{{o_alpha,o_alpha,o_alpha,o_alpha},{2,2}};
    ix2_4_abab   = Tensor<T>{{o_alpha,o_beta, o_alpha,o_beta}, {2,2}};
    ix2_5_aaaa   = Tensor<T>{{o_alpha,v_alpha,o_alpha,v_alpha},{2,2}};
    ix2_5_baab   = Tensor<T>{{o_beta, v_alpha,o_alpha,v_beta}, {2,2}};
    ix2_5_bbbb   = Tensor<T>{{o_beta, v_beta, o_beta, v_beta}, {2,2}};
    ix2_5_abba   = Tensor<T>{{o_alpha,v_beta, o_beta, v_alpha},{2,2}};
    ix2_5_abab   = Tensor<T>{{o_alpha,v_beta, o_alpha,v_beta}, {2,2}};
    ix2_6_2_a    = Tensor<T>{{o_alpha,v_alpha},{1,1}};
    ix2_6_2_b    = Tensor<T>{{o_beta, v_beta},{1,1}};
    ix2_6_3_aaaa = Tensor<T>{{o_alpha,o_alpha,o_alpha,v_alpha},{2,2}};
    ix2_6_3_abba = Tensor<T>{{o_alpha,o_beta, o_beta, v_alpha},{2,2}};
    ix2_6_3_abab = Tensor<T>{{o_alpha,o_beta, o_alpha,v_beta}, {2,2}};
    ix2_6_3_baab = Tensor<T>{{o_beta, o_alpha,o_alpha,v_beta}, {2,2}};
    ix2_6_3_bbbb = Tensor<T>{{o_beta, o_beta, o_beta, v_beta}, {2,2}};
    
    Tensor<T> lt12         {{O, O},{1,1}};
    // Tensor<T> ix1_1        {{O, O},{1,1}};
    Tensor<T> ix1_1_1      {{O, V},{1,1}};
    // Tensor<T> ix1_4        {{O, O, O, V},{2,2}};
    Tensor<T> ix2_1_1      {{O, V, O, V},{2,2}};
    Tensor<T> ix2_1_3      {{O, O, O, V},{2,2}};
    Tensor<T> ix2_1_temp   {{O, V, O, O},{2,2}};
    Tensor<T> ix2_1        {{O, V, O, O},{2,2}};
    Tensor<T> ix2_2        {{O, O},{1,1}};
    Tensor<T> ix2_3        {{V, V},{1,1}};
    Tensor<T> ix2_4_1      {{O, O, O, V},{2,2}};
    Tensor<T> ix2_4_temp   {{O, O, O, O},{2,2}};
    Tensor<T> ix2_4        {{O, O, O, O},{2,2}};
    Tensor<T> ix2_5        {{O, V, O, V},{2,2}};
    Tensor<T> ix2_6_2      {{O, V},{1,1}};
    Tensor<T> ix2_6_3      {{O, O, O, V},{2,2}};

    Tensor<T> v2ijka  {{O,O,O,V},{2,2}};
    Tensor<T> v2ijkl  {{O,O,O,O},{2,2}};
    Tensor<T> v2iajb  {{O,V,O,V},{2,2}};
    Tensor<T> v2ijab  {{O,O,V,V},{2,2}};
    Tensor<T> v2iabc  {{O,V,V,V},{2,2}};
    
    auto in_t1 = std::chrono::high_resolution_clock::now();

    #if GF_IN_SG
    if(subcomm != MPI_COMM_NULL) {
      sub_sch.allocate
    #else 
      sch.allocate
    #endif
                (lt12,
                //  ix1_1,
                 ix1_1_1,
                //  ix1_4,
                 ix2_1_1,ix2_1_3,ix2_1_temp,ix2_1,
                 ix2_2,ix2_3,
                 ix2_4_1,ix2_4_temp,ix2_4,ix2_5,
                 ix2_6_2,ix2_6_3).execute();
    #if GF_IN_SG
    }
    #endif

    sch.allocate(d_t1_a, d_t1_b, 
                 d_t2_aaaa, d_t2_bbbb, d_t2_abab,
                 v2_aaaa, v2_bbbb, v2_abab,
                 lt12_a, v2ijab,
                //  ix1_1_a, 
                 ix1_1_1_a, ix1_1_1_b,
                //  ix1_4_aaaa, ix1_4_abab, 
                 ix2_1_aaaa, ix2_1_abab,
                 ix2_2_a, ix2_2_b, 
                 ix2_3_a, ix2_3_b, 
                 ix2_4_aaaa, ix2_4_abab, 
                 ix2_5_aaaa, ix2_5_abba, ix2_5_abab, 
                 ix2_5_bbbb, ix2_5_baab,
                 ix2_6_2_a, ix2_6_2_b, 
                 ix2_6_3_aaaa, ix2_6_3_abba, ix2_6_3_abab,
                 ix2_6_3_bbbb, ix2_6_3_baab,
                 t2v2)
       .execute();

    auto in_t2 = std::chrono::high_resolution_clock::now();
    auto in_alloc_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((in_t2 - in_t1)).count();

    if(!fs::exists(files_dir)) {
      fs::create_directories(files_dir);
    }

    if(fs::exists(t2v2file)       && fs::exists(lt12afile)       && 
       fs::exists(v2ijabfile)      &&  
       fs::exists(ix1_1_1afile)    && fs::exists(ix1_1_1bfile)    && 
      //  fs::exists(ix1_4aaaafile)   && fs::exists(ix1_4ababfile)   && 
       fs::exists(ix2_1aaaafile)   && fs::exists(ix2_1ababfile)   && 
       fs::exists(ix2_2afile)      && fs::exists(ix2_2bfile)      && 
       fs::exists(ix2_3afile)      && fs::exists(ix2_3bfile)      &&
       fs::exists(ix2_4aaaafile)   && fs::exists(ix2_4ababfile)   && 
       fs::exists(ix2_5aaaafile)   && fs::exists(ix2_5abbafile)   && fs::exists(ix2_5ababfile)   && 
       fs::exists(ix2_5bbbbfile)   && fs::exists(ix2_5baabfile)   &&  
       fs::exists(ix2_6_2afile)    && fs::exists(ix2_6_2bfile)    &&
       fs::exists(ix2_6_3aaaafile) && fs::exists(ix2_6_3abbafile) && fs::exists(ix2_6_3ababfile) &&
       fs::exists(ix2_6_3bbbbfile) && fs::exists(ix2_6_3baabfile) && 
       ccsd_options.gf_restart) {
      
      read_from_disk(t2v2,t2v2file);
      read_from_disk(lt12_a,lt12afile);
      read_from_disk(v2ijab,v2ijabfile);
      // read_from_disk(ix1_1_a,ix1_1afile);
      read_from_disk(ix1_1_1_a,ix1_1_1afile);
      read_from_disk(ix1_1_1_b,ix1_1_1bfile);
      // read_from_disk(ix1_4_aaaa,ix1_4aaaafile);
      // read_from_disk(ix1_4_abab,ix1_4ababfile);
      read_from_disk(ix2_1_aaaa,ix2_1aaaafile);
      read_from_disk(ix2_1_abab,ix2_1ababfile);
      read_from_disk(ix2_2_a,ix2_2afile);
      read_from_disk(ix2_2_b,ix2_2bfile);
      read_from_disk(ix2_3_a,ix2_3afile);
      read_from_disk(ix2_3_b,ix2_3bfile);
      read_from_disk(ix2_4_aaaa,ix2_4aaaafile);
      read_from_disk(ix2_4_abab,ix2_4ababfile);
      read_from_disk(ix2_5_aaaa,ix2_5aaaafile);
      read_from_disk(ix2_5_abba,ix2_5abbafile);
      read_from_disk(ix2_5_abab,ix2_5ababfile);
      read_from_disk(ix2_5_bbbb,ix2_5bbbbfile);
      read_from_disk(ix2_5_baab,ix2_5baabfile);
      read_from_disk(ix2_6_2_a,ix2_6_2afile);
      read_from_disk(ix2_6_2_b,ix2_6_2bfile);
      read_from_disk(ix2_6_3_aaaa,ix2_6_3aaaafile);
      read_from_disk(ix2_6_3_abba,ix2_6_3abbafile);
      read_from_disk(ix2_6_3_abab,ix2_6_3ababfile);
      read_from_disk(ix2_6_3_bbbb,ix2_6_3bbbbfile);
      read_from_disk(ix2_6_3_baab,ix2_6_3baabfile);
      
    }
    else {
      #if GF_IN_SG
      if(subcomm != MPI_COMM_NULL) {
        sub_sch
      #else 
        sch
      #endif
        .allocate(v2ijka,v2ijkl,v2iajb,v2iabc)
        
        ( v2ijka(h1,h2,h3,p1)  =  cholVpr(h1,h3,cind) * cholVpr(h2,p1,cind) )
        // ( v2ijka(h1,h2,h3,p1) +=  -1.0 * v2ijka(h2,h1,h3,p1) )
        ( v2ijka(h1,h2,h3,p1) +=  -1.0 * cholVpr(h2,h3,cind) * cholVpr(h1,p1,cind) )

        ( v2ijkl(h1,h2,h3,h4)  =  cholVpr(h1,h3,cind) * cholVpr(h2,h4,cind) )
        // ( v2ijkl(h1,h2,h3,h4) +=  -1.0 * v2ijkl(h1,h2,h4,h3) )
        ( v2ijkl(h1,h2,h3,h4) +=  -1.0 * cholVpr(h1,h4,cind) * cholVpr(h2,h3,cind) )

        ( v2iajb(h1,p1,h2,p2)  =  cholVpr(h1,h2,cind) * cholVpr(p1,p2,cind) )
        ( v2ijab(h1,h2,p1,p2)  =  cholVpr(h1,p1,cind) * cholVpr(h2,p2,cind) )
        // ( v2iajb(h1,p1,h2,p2) +=  -1.0 * v2ijab(h1,h2,p2,p1) )
        // ( v2ijab(h1,h2,p1,p2) +=  -1.0 * v2ijab(h1,h2,p2,p1) )
        ( v2iajb(h1,p1,h2,p2) +=  -1.0 * cholVpr(h1,p2,cind) * cholVpr(h2,p1,cind) )
        ( v2ijab(h1,h2,p1,p2) +=  -1.0 * cholVpr(h1,p2,cind) * cholVpr(h2,p1,cind) )

        ( v2iabc(h1,p1,p2,p3)  =  cholVpr(h1,p2,cind) * cholVpr(p1,p3,cind) )
        // ( v2iabc(h1,p1,p2,p3) +=  -1.0 * v2iabc(h1,p1,p3,p2) )
        ( v2iabc(h1,p1,p2,p3) +=  -1.0 * cholVpr(h1,p3,cind) * cholVpr(p1,p2,cind) )

        ( t2v2(h1,h2) = 0.5 * d_t2(p1,p2,h1,h3) * v2ijab(h3,h2,p1,p2)                                )

        ( lt12(h1_oa,h3_oa)  = 0.5 * d_t2(p1_va,p2_va,h1_oa,h2_oa) * d_t2(p1_va,p2_va,h3_oa,h2_oa) )
        ( lt12(h1_oa,h3_oa) +=       d_t2(p1_va,p2_vb,h1_oa,h2_ob) * d_t2(p1_va,p2_vb,h3_oa,h2_ob) )
        ( lt12(h1_oa,h2_oa) +=       d_t1(p1_va,h1_oa) * d_t1(p1_va,h2_oa)                         )
        ( lt12(h1_ob,h3_ob)  = 0.5 * d_t2(p1_vb,p2_vb,h1_ob,h2_ob) * d_t2(p1_vb,p2_vb,h3_ob,h2_ob) )
        ( lt12(h1_ob,h3_ob) +=       d_t2(p2_va,p1_vb,h2_oa,h1_ob) * d_t2(p2_va,p1_vb,h2_oa,h3_ob) )
        ( lt12(h1_ob,h2_ob) +=       d_t1(p1_vb,h1_ob) * d_t1(p1_vb,h2_ob)                         )

        (   ix1_1_1(h6,p7)    =        d_f1(h6,p7)                            )
        (   ix1_1_1(h6,p7)   +=        d_t1(p4,h5)       * v2ijab(h5,h6,p4,p7)  )
        
        ( ix2_1(h9,p3,h1,h2)       =         v2ijka(h1,h2,h9,p3)                          )

        (   ix2_1_1(h9,p3,h1,p5)   =         v2iajb(h9,p3,h1,p5)                          )
        ( ix2_5(h7,p3,h1,p8)       =         d_t1(p5,h1) * v2iabc(h7,p3,p5,p8)     ) //O2V3
        (   ix2_1_1(h9,p3,h1,p5)  += -0.5  * ix2_5(h9,p3,h1,p5)            ) //O2V3
        ( ix2_5(h7,p3,h1,p8)      +=         v2iajb(h7,p3,h1,p8)                          )
        

        ( ix2_1_temp(h9,p3,h1,h2)  =         d_t1(p5,h1) * ix2_1_1(h9,p3,h2,p5)         ) //O3V2
        ( ix2_1(h9,p3,h1,h2)      += -1    * ix2_1_temp(h9,p3,h1,h2)                    )
        ( ix2_1(h9,p3,h2,h1)      +=         ix2_1_temp(h9,p3,h1,h2)                    ) 
        ( ix2_1(h9,p3,h1,h2)      += -1    * d_t2(p3,p8,h1,h2)  * ix1_1_1(h9,p8)        ) //O3V2
        (   ix2_1_3(h6,h9,h1,p5)   =         v2ijka(h6,h9,h1,p5)                          )
        (   ix2_1_3(h6,h9,h1,p5)  += -1    * d_t1(p7,h1)        * v2ijab(h6,h9,p5,p7)     ) //O3V2
        ( ix2_1_temp(h9,p3,h1,h2)  =         d_t2(p3,p5,h1,h6)  * ix2_1_3(h6,h9,h2,p5)  ) //O4V2
        ( ix2_1(h9,p3,h1,h2)      +=         ix2_1_temp(h9,p3,h1,h2)                    )
        ( ix2_1(h9,p3,h2,h1)      += -1    * ix2_1_temp(h9,p3,h1,h2)                    ) 
        ( ix2_1(h9,p3,h1,h2)      +=  0.5  * d_t2(p5,p6,h1,h2)  * v2iabc(h9,p3,p5,p6)     ) //O3V3

        ( ix2_2(h8,h1)             =         d_f1(h8,h1)                                )
        ( ix2_2(h8,h1)            +=         d_t1(p9,h1)        * ix1_1_1(h8,p9)        )
        ( ix2_2(h8,h1)            += -1    * d_t1(p5,h6)        * v2ijka(h6,h8,h1,p5)     )
        ( ix2_2(h8,h1)            += -0.5  * d_t2(p5,p6,h1,h7)  * v2ijab(h7,h8,p5,p6)     ) //O3V2

        ( ix2_3(p3,p8)             =         d_f1(p3,p8)                                )
        ( ix2_3(p3,p8)            +=         d_t1(p5,h6)        * v2iabc(h6,p3,p5,p8)     ) 
        ( ix2_3(p3,p8)            +=  0.5  * d_t2(p3,p5,h6,h7)  * v2ijab(h6,h7,p5,p8)     ) //O2V3

        ( ix2_4(h9,h10,h1,h2)      =         v2ijkl(h9,h10,h1,h2)                         )
        (   ix2_4_1(h9,h10,h1,p5)  =         v2ijka(h9,h10,h1,p5)                         )
        (   ix2_4_1(h9,h10,h1,p5) += -0.5  * d_t1(p6,h1)        * v2ijab(h9,h10,p5,p6)    ) //O3V2
        ( ix2_4_temp(h9,h10,h1,h2) =         d_t1(p5,h1)        * ix2_4_1(h9,h10,h2,p5) ) //O4V
        ( ix2_4(h9,h10,h1,h2)     += -1    * ix2_4_temp(h9,h10,h1,h2)                   )
        ( ix2_4(h9,h10,h2,h1)     +=         ix2_4_temp(h9,h10,h1,h2)                   ) 
        ( ix2_4(h9,h10,h1,h2)     +=  0.5  * d_t2(p5,p6,h1,h2)  * v2ijab(h9,h10,p5,p6)    ) //O4V2

        ( ix2_6_2(h10,p5)          =         d_f1(h10,p5)                               )
        ( ix2_6_2(h10,p5)         += 1.0   * d_t1(p6,h7)        * v2ijab(h7,h10,p5,p6)    )

        ( ix2_6_3(h8,h10,h1,p9)    =         v2ijka(h8,h10,h1,p9)                         )
        ( ix2_6_3(h8,h10,h1,p9)   +=         d_t1(p5,h1)        * v2ijab(h8,h10,p5,p9)    ) //O3V2
      //   .execute();

      // sch
        (lt12_a(h1_oa,h2_oa) = lt12(h1_oa,h2_oa))
        
        // (ix1_1_a(h1_oa,h2_oa) = ix1_1(h1_oa,h2_oa))
        (ix1_1_1_a(h1_oa,p1_va) = ix1_1_1(h1_oa,p1_va))
        (ix1_1_1_b(h1_ob,p1_vb) = ix1_1_1(h1_ob,p1_vb))
        // (ix1_4_aaaa(h1_oa,h2_oa,h3_oa,p1_va) = ix1_4(h1_oa,h2_oa,h3_oa,p1_va))
        // (ix1_4_abab(h1_oa,h2_ob,h3_oa,p1_vb) = ix1_4(h1_oa,h2_ob,h3_oa,p1_vb))

        (ix2_1_aaaa(h1_oa,p1_va,h2_oa,h3_oa) = ix2_1(h1_oa,p1_va,h2_oa,h3_oa))
        (ix2_1_abab(h1_oa,p1_vb,h2_oa,h3_ob) = ix2_1(h1_oa,p1_vb,h2_oa,h3_ob))
        (ix2_2_a(h1_oa,h2_oa) = ix2_2(h1_oa,h2_oa))
        (ix2_2_b(h1_ob,h2_ob) = ix2_2(h1_ob,h2_ob))
        (ix2_3_a(p1_va,p2_va) = ix2_3(p1_va,p2_va))
        (ix2_3_b(p1_vb,p2_vb) = ix2_3(p1_vb,p2_vb))
        (ix2_4_aaaa(h1_oa,h2_oa,h3_oa,h4_oa) = ix2_4(h1_oa,h2_oa,h3_oa,h4_oa))
        (ix2_4_abab(h1_oa,h2_ob,h3_oa,h4_ob) = ix2_4(h1_oa,h2_ob,h3_oa,h4_ob))

        (ix2_5_aaaa(h1_oa,p1_va,h2_oa,p2_va) = ix2_5(h1_oa,p1_va,h2_oa,p2_va))
        (ix2_5_abba(h1_oa,p1_vb,h2_ob,p2_va) = ix2_5(h1_oa,p1_vb,h2_ob,p2_va))
        (ix2_5_abab(h1_oa,p1_vb,h2_oa,p2_vb) = ix2_5(h1_oa,p1_vb,h2_oa,p2_vb))
        (ix2_5_bbbb(h1_ob,p1_vb,h2_ob,p2_vb) = ix2_5(h1_ob,p1_vb,h2_ob,p2_vb))
        (ix2_5_baab(h1_ob,p1_va,h2_oa,p2_vb) = ix2_5(h1_ob,p1_va,h2_oa,p2_vb))
        (ix2_6_2_a(h1_oa,p1_va) = ix2_6_2(h1_oa,p1_va))
        (ix2_6_2_b(h1_ob,p1_vb) = ix2_6_2(h1_ob,p1_vb))

        (ix2_6_3_aaaa(h1_oa,h2_oa,h3_oa,p1_va) = ix2_6_3(h1_oa,h2_oa,h3_oa,p1_va))
        (ix2_6_3_abba(h1_oa,h2_ob,h3_ob,p1_va) = ix2_6_3(h1_oa,h2_ob,h3_ob,p1_va))
        (ix2_6_3_abab(h1_oa,h2_ob,h3_oa,p1_vb) = ix2_6_3(h1_oa,h2_ob,h3_oa,p1_vb))
        (ix2_6_3_bbbb(h1_ob,h2_ob,h3_ob,p1_vb) = ix2_6_3(h1_ob,h2_ob,h3_ob,p1_vb))
        (ix2_6_3_baab(h1_ob,h2_oa,h3_oa,p1_vb) = ix2_6_3(h1_ob,h2_oa,h3_oa,p1_vb))
        .deallocate(v2ijka,v2ijkl,v2iajb,v2iabc)
        .execute();
      #if GF_IN_SG
      }
      ec.pg().barrier();
      #endif

      write_to_disk(t2v2,t2v2file);
      write_to_disk(lt12_a,lt12afile);
      write_to_disk(v2ijab,v2ijabfile);
      // write_to_disk(ix1_1_a,ix1_1afile);
      write_to_disk(ix1_1_1_a,ix1_1_1afile);
      write_to_disk(ix1_1_1_b,ix1_1_1bfile);
      // write_to_disk(ix1_4_aaaa,ix1_4aaaafile);
      // write_to_disk(ix1_4_abab,ix1_4ababfile);
      write_to_disk(ix2_1_aaaa,ix2_1aaaafile);
      write_to_disk(ix2_1_abab,ix2_1ababfile);
      write_to_disk(ix2_2_a,ix2_2afile);
      write_to_disk(ix2_2_b,ix2_2bfile);
      write_to_disk(ix2_3_a,ix2_3afile);
      write_to_disk(ix2_3_b,ix2_3bfile);
      write_to_disk(ix2_4_aaaa,ix2_4aaaafile);
      write_to_disk(ix2_4_abab,ix2_4ababfile);
      write_to_disk(ix2_5_aaaa,ix2_5aaaafile);
      write_to_disk(ix2_5_abba,ix2_5abbafile);
      write_to_disk(ix2_5_abab,ix2_5ababfile);
      write_to_disk(ix2_5_bbbb,ix2_5bbbbfile);
      write_to_disk(ix2_5_baab,ix2_5baabfile);
      write_to_disk(ix2_6_2_a,ix2_6_2afile);
      write_to_disk(ix2_6_2_b,ix2_6_2bfile);
      write_to_disk(ix2_6_3_aaaa,ix2_6_3aaaafile);
      write_to_disk(ix2_6_3_abba,ix2_6_3abbafile);
      write_to_disk(ix2_6_3_abab,ix2_6_3ababfile);
      write_to_disk(ix2_6_3_bbbb,ix2_6_3bbbbfile);
      write_to_disk(ix2_6_3_baab,ix2_6_3baabfile);
    }

    if(subcomm != MPI_COMM_NULL) {
      sub_sch
          (d_t1_a(p1_va,h3_oa) = d_t1(p1_va,h3_oa))
          (d_t1_b(p1_vb,h3_ob) = d_t1(p1_vb,h3_ob))
          (d_t2_aaaa(p1_va,p2_va,h3_oa,h4_oa) = d_t2(p1_va,p2_va,h3_oa,h4_oa))
          (d_t2_abab(p1_va,p2_vb,h3_oa,h4_ob) = d_t2(p1_va,p2_vb,h3_oa,h4_ob))
          (d_t2_bbbb(p1_vb,p2_vb,h3_ob,h4_ob) = d_t2(p1_vb,p2_vb,h3_ob,h4_ob))
          (v2_aaaa(h1_oa,h2_oa,p1_va,p2_va) = v2ijab(h1_oa,h2_oa,p1_va,p2_va))
          (v2_abab(h1_oa,h2_ob,p1_va,p2_vb) = v2ijab(h1_oa,h2_ob,p1_va,p2_vb))
          (v2_bbbb(h1_ob,h2_ob,p1_vb,p2_vb) = v2ijab(h1_ob,h2_ob,p1_vb,p2_vb)).execute();
    }
    ec.pg().barrier();

    auto in_t3 = std::chrono::high_resolution_clock::now();
    auto in_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((in_t3 - in_t2)).count();

    #if GF_IN_SG
    if(subcomm != MPI_COMM_NULL) {
      sub_sch
    #else 
      sch
    #endif
          .deallocate(lt12,
                //  ix1_1,
                 ix1_1_1,
                //  ix1_4,
                 ix2_1_1,ix2_1_3,ix2_1_temp,ix2_1,
                 ix2_2,ix2_3,
                 ix2_4_1,ix2_4_temp,ix2_4,ix2_5,
                 ix2_6_2,ix2_6_3).execute();
    #if GF_IN_SG
    }
    #endif

    sch.deallocate(cholVpr,v2ijab,d_t1,d_t2).execute();

    auto in_t4 = std::chrono::high_resolution_clock::now();
    auto in_dealloc_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((in_t4 - in_t3)).count();
    if(rank == 0) {
      std::cout << std::endl << "Time for computing intermediates: " << in_time << " secs" << std::endl; 
      std::cout << "  -- Time for allocating intermediates: " << in_alloc_time << " secs" << std::endl; 
      std::cout << "  -- Time for deallocating intermediates: " << in_dealloc_time << " secs" << std::endl; 
    }

    size_t prev_qr_rank = 0;

    while (true) {

      const std::string levelstr = std::to_string(level);
      std::string q1afile    = files_prefix+".q1_a.l"+levelstr;
      std::string q2aaafile  = files_prefix+".q2_aaa.l"+levelstr;
      std::string q2babfile  = files_prefix+".q2_bab.l"+levelstr;
      std::string hx1afile   = files_prefix+".hx1_a.l"+levelstr;
      std::string hx2aaafile = files_prefix+".hx2_aaa.l"+levelstr;
      std::string hx2babfile = files_prefix+".hx2_bab.l"+levelstr;
      std::string hsubfile   = files_prefix+".hsub.l"+levelstr;
      std::string bsubafile  = files_prefix+".bsub_a.l"+levelstr;
      std::string cpafile    = files_prefix+".cp_a.l"+levelstr;
      
      bool gf_restart = fs::exists(q1afile)    && 
                        fs::exists(q2aaafile)  && fs::exists(q2babfile)  &&
                        fs::exists(hx1afile)   && 
                        fs::exists(hx2aaafile) && fs::exists(hx2babfile) &&
                        fs::exists(hsubfile)   && fs::exists(bsubafile)  && 
                        fs::exists(cpafile)    && ccsd_options.gf_restart;


      if(level==1) {
        omega_extra.push_back(omega_min);
        // omega_extra.push_back((omega_max+omega_min)/2);
        omega_extra.push_back(omega_max);
        // omega_extra_finished = omega_extra;
      }

      for(auto x: omega_extra) 
        omega_extra_finished.push_back(x);

      auto qr_rank = omega_extra_finished.size() * p_oi/2;
      if (rank == 0) cout << "qr_rank in level " << level << " = " << qr_rank << endl;

      TiledIndexSpace otis;
      if(ndiis > qr_rank){
        diis_tis = {IndexSpace{range(0,ndiis)}};
        otis = {diis_tis, range(0,qr_rank)};
      }
      else{
        otis = {IndexSpace{range(qr_rank)}};
        diis_tis = {otis,range(0,ndiis)};
      }

      TiledIndexSpace unit_tis{diis_tis,range(0,1)};
      auto [u1] = unit_tis.labels<1>("all");
      
      // ComplexTensor x1_a{o_alpha,unit_tis};
      // ComplexTensor x2_aaa{v_alpha,o_alpha,o_alpha,unit_tis};
      // ComplexTensor x2_bab{v_beta, o_alpha,o_beta, unit_tis};
      // ComplexTensor::allocate(&ec,x1_a,x2_aaa,x2_bab);

      for(auto x: omega_extra) {
        // omega_extra_finished.push_back(x);
        ndiis=ccsd_options.gf_ndiis;
        gf_omega = x;
  #endif

      if(!gf_restart){
          gfccsd_driver<T>(ec, *sub_ec, subcomm, MO, 
                           d_t1_a, d_t1_b, d_t2_aaaa, d_t2_bbbb, d_t2_abab, 
                           d_f1, //d_v2, 
                           p_evl_sorted_occ,p_evl_sorted_virt, total_orbitals, nocc, nvir,
                           nptsi,unit_tis,files_prefix,levelstr);
      }

  #if MOR


      }

    // auto cc_read_x0 = std::chrono::high_resolution_clock::now();
    // for(auto x: omega_extra) {
    //   gf_omega = x;
    //   std::stringstream gfo;
    //   gfo << std::defaultfloat << gf_omega;

    //   for (size_t pi=0; pi < p_oi/2; pi++){
    //     std::string x1_a_wpi_file = files_prefix+".x1_a.W"+gfo.str()+".oi"+std::to_string(pi);
    //     std::string x2_aaa_wpi_file = files_prefix+".x2_aaa.W"+gfo.str()+".oi"+std::to_string(pi);
    //     std::string x2_bab_wpi_file = files_prefix+".x2_bab.W"+gfo.str()+".oi"+std::to_string(pi);

    //     if(fs::exists(x1_a_wpi_file) && fs::exists(x2_aaa_wpi_file) && fs::exists(x2_bab_wpi_file)){
    //       read_from_disk(x1_a,x1_a_wpi_file);
    //       read_from_disk(x2_aaa,x2_aaa_wpi_file);
    //       read_from_disk(x2_bab,x2_bab_wpi_file);  

    //       x1_omega_a[nptsi]   = ComplexTensor{o_alpha,unit_tis};
    //       x2_omega_aaa[nptsi] = ComplexTensor{v_alpha,o_alpha,o_alpha,unit_tis};
    //       x2_omega_bab[nptsi] = ComplexTensor{v_beta, o_alpha,o_beta, unit_tis};
    //       ComplexTensor::allocate(&ec,x1_omega_a[nptsi],
    //                               x2_omega_aaa[nptsi],x2_omega_bab[nptsi]);
    //       sch
    //         (x1_omega_a[nptsi]()   = x1_a()) 
    //         (x2_omega_aaa[nptsi]() = x2_aaa())
    //         (x2_omega_bab[nptsi]() = x2_bab())
    //         .execute();

    //       nptsi++;

    //     }
    //     else nwx_terminate("ERROR: One or both of " + x1_a_wpi_file + " and " + x2_aaa_wpi_file + " do not exist!");
    //   }
    // }

    // sch.deallocate(x1_a, x2_aaa, x2_bab).execute();

    // auto cc_read_x1 = std::chrono::high_resolution_clock::now();
    // double time_read_x  = std::chrono::duration_cast<std::chrono::duration<double>>((cc_read_x1 - cc_read_x0)).count();
    // if(rank == 0) cout << endl << "Time to read in pre-computed x1/x2: " << time_read_x << " secs" << endl;

    ComplexTensor q1_tamm_a{o_alpha,otis};
    ComplexTensor q2_tamm_aaa{v_alpha,o_alpha,o_alpha,otis};
    ComplexTensor q2_tamm_bab{v_beta, o_alpha,o_beta, otis};
    ComplexTensor Hx1_tamm_a{o_alpha,otis};   
    ComplexTensor Hx2_tamm_aaa{v_alpha,o_alpha,o_alpha,otis};
    ComplexTensor Hx2_tamm_bab{v_beta, o_alpha,o_beta, otis};
      
    if(!gf_restart) {
      auto cc_t1 = std::chrono::high_resolution_clock::now();

      sch.allocate(q1_tamm_a, q2_tamm_aaa, q2_tamm_bab,
          Hx1_tamm_a,Hx2_tamm_aaa,Hx2_tamm_bab)
          .execute();

      const std::string plevelstr = std::to_string(level-1);
      std::string pq1afile    = files_prefix+".q1_a.l"+plevelstr;
      std::string pq2aaafile  = files_prefix+".q2_aaa.l"+plevelstr;
      std::string pq2babfile  = files_prefix+".q2_bab.l"+plevelstr;

      decltype(qr_rank) ivec_start = 0;
      bool prev_q12 = fs::exists(pq1afile) && fs::exists(pq2aaafile) && fs::exists(pq2babfile);

      if(prev_q12) {
        TiledIndexSpace otis_prev{otis,range(0,prev_qr_rank)};
        auto [op1] = otis_prev.labels<1>("all");
        ComplexTensor q1_prev_a  {o_alpha,otis_prev};
        ComplexTensor q2_prev_aaa{v_alpha,o_alpha,o_alpha,otis_prev};
        ComplexTensor q2_prev_bab{v_beta, o_alpha,o_beta, otis_prev};
        sch.allocate(q1_prev_a,q2_prev_aaa,q2_prev_bab
                     ).execute();

        read_from_disk(q1_prev_a  ,  pq1afile);
        read_from_disk(q2_prev_aaa,pq2aaafile);
        read_from_disk(q2_prev_bab,pq2babfile);

        ivec_start = prev_qr_rank;

        if(subcomm != MPI_COMM_NULL){
          sub_sch(q1_tamm_a(h1_oa,op1) = q1_prev_a(h1_oa,op1))
            (q2_tamm_aaa(p1_va,h1_oa,h2_oa,op1) = q2_prev_aaa(p1_va,h1_oa,h2_oa,op1))
            (q2_tamm_bab(p1_vb,h1_oa,h2_ob,op1) = q2_prev_bab(p1_vb,h1_oa,h2_ob,op1)).execute();
        }

        sch.deallocate(q1_prev_a,q2_prev_aaa,q2_prev_bab).execute();

      }     

      auto cc_t2 = std::chrono::high_resolution_clock::now();
      double time  = std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
      if(rank == 0) cout << endl << "Time to read in pre-computed Q1/Q2: " << time << " secs" << endl;

      ComplexTensor q1_tmp_a{o_alpha,unit_tis};
      ComplexTensor q2_tmp_aaa{v_alpha,o_alpha,o_alpha,unit_tis};
      ComplexTensor q2_tmp_bab{v_beta, o_alpha,o_beta, unit_tis};
        
      //TODO: optimize Q1/Q2 computation
      //Gram-Schmidt orthogonalization
      double time_gs_orth = 0.0;
      double time_gs_norm = 0.0;
      double total_time_gs  = 0.0;

      bool gs_restart = fs::exists(q1afile) && fs::exists(q2aaafile) && fs::exists(q2babfile);

      if(!gs_restart){
       sch.allocate(q1_tmp_a, q2_tmp_aaa, q2_tmp_bab).execute();

       for(decltype(qr_rank) ivec=ivec_start;ivec<qr_rank;ivec++) {

        auto cc_t0 = std::chrono::high_resolution_clock::now();

        auto W_read = omega_extra_finished[ivec/(p_oi/2)];
        auto pi_read = ivec%(p_oi/2);
        std::stringstream gfo;
        gfo << std::defaultfloat << W_read;
        
        std::string x1_a_wpi_file = files_prefix+".x1_a.W"+gfo.str()+".oi"+std::to_string(pi_read);
        std::string x2_aaa_wpi_file = files_prefix+".x2_aaa.W"+gfo.str()+".oi"+std::to_string(pi_read);
        std::string x2_bab_wpi_file = files_prefix+".x2_bab.W"+gfo.str()+".oi"+std::to_string(pi_read);

        if(fs::exists(x1_a_wpi_file) && fs::exists(x2_aaa_wpi_file) && fs::exists(x2_bab_wpi_file)){
          read_from_disk(q1_tmp_a,x1_a_wpi_file);
          read_from_disk(q2_tmp_aaa,x2_aaa_wpi_file);
          read_from_disk(q2_tmp_bab,x2_bab_wpi_file);  

        }
        else nwx_terminate("ERROR: One or both of " + x1_a_wpi_file + " and " + x2_aaa_wpi_file + " do not exist!");
        
        // sch
        //     (q1_tmp_a()   = x1_omega_a.at(ivec)())
        //     (q2_tmp_aaa() = x2_omega_aaa.at(ivec)())
        //     (q2_tmp_bab() = x2_omega_bab.at(ivec)())
        //     .execute();
            
        //TODO: schedule all iterations before executing
        // for(decltype(ivec) jvec = 0; jvec<ivec; jvec++){
          if(ivec>0){
            TiledIndexSpace tsc{otis, range(0,ivec)};
            auto [sc] = tsc.labels<1>("all");

            ComplexTensor oscalar{tsc,unit_tis};
            ComplexTensor x1c_a{o_alpha,tsc};
            ComplexTensor x2c_aaa{v_alpha,o_alpha,o_alpha,tsc};
            ComplexTensor x2c_bab{v_beta, o_alpha,o_beta, tsc};

            //sch.allocate(x1c_a,x2c_aaa,x2c_bab).execute();

            #if GF_GS_SG
            if(subcomm != MPI_COMM_NULL){
             sub_sch.allocate
            #else
             sch.allocate
            #endif
             (x1c_a,x2c_aaa,x2c_bab)
             (x1c_a(h1_oa,sc) = q1_tamm_a(h1_oa,sc))
             (x2c_aaa(p1_va,h1_oa,h2_oa,sc) = q2_tamm_aaa(p1_va,h1_oa,h2_oa,sc))
             (x2c_bab(p1_vb,h1_oa,h2_ob,sc) = q2_tamm_bab(p1_vb,h1_oa,h2_ob,sc))
             .execute();

             tamm::conj_ip(x1c_a);
             tamm::conj_ip(x2c_aaa);
             tamm::conj_ip(x2c_bab);

             #if GF_GS_SG
              sub_sch.allocate
             #else
              sch.allocate
             #endif
             (oscalar)
             (oscalar(sc,u1)  = -1.0 * q1_tmp_a(h1_oa,u1) * x1c_a(h1_oa,sc))
             (oscalar(sc,u1) += -1.0 * q2_tmp_aaa(p1_va,h1_oa,h2_oa,u1) * x2c_aaa(p1_va,h1_oa,h2_oa,sc))
             (oscalar(sc,u1) += -2.0 * q2_tmp_bab(p1_vb,h1_oa,h2_ob,u1) * x2c_bab(p1_vb,h1_oa,h2_ob,sc))

             (q1_tmp_a(h1_oa,u1) += oscalar(sc,u1) * q1_tamm_a(h1_oa,sc))
             (q2_tmp_aaa(p1_va,h1_oa,h2_oa,u1) += oscalar(sc,u1) * q2_tamm_aaa(p1_va,h1_oa,h2_oa,sc))
             (q2_tmp_bab(p1_vb,h1_oa,h2_ob,u1) += oscalar(sc,u1) * q2_tamm_bab(p1_vb,h1_oa,h2_ob,sc))
             .deallocate(x1c_a,x2c_aaa,x2c_bab,oscalar).execute();
            #if GF_GS_SG
            }
            ec.pg().barrier();
            #endif
          }

        auto cc_t1 = std::chrono::high_resolution_clock::now();
        time_gs_orth += std::chrono::duration_cast<std::chrono::duration<double>>((cc_t1 - cc_t0)).count();
      
        auto q1norm_a   = norm(q1_tmp_a); 
        auto q2norm_aaa = norm(q2_tmp_aaa);
        auto q2norm_bab = norm(q2_tmp_bab);

        // Normalization factor
        T newsc = 1.0/std::real(sqrt(q1norm_a*q1norm_a + 
                      q2norm_aaa*q2norm_aaa + 2.0*q2norm_bab*q2norm_bab));

        std::complex<T> cnewsc = static_cast<std::complex<T>>(newsc);
        // scale(q1_tmp_a,  static_cast<std::complex<T>>(newsc));
        // scale(q2_tmp_aaa,static_cast<std::complex<T>>(newsc));
        // scale(q2_tmp_bab,static_cast<std::complex<T>>(newsc));

        TiledIndexSpace tsc{otis, range(ivec,ivec+1)};
        auto [sc] = tsc.labels<1>("all");

        if(subcomm != MPI_COMM_NULL){
           sub_sch
           (q1_tamm_a(h1_oa,sc) = cnewsc * q1_tmp_a(h1_oa,u1))
           (q2_tamm_aaa(p2_va,h1_oa,h2_oa,sc) = cnewsc * q2_tmp_aaa(p2_va,h1_oa,h2_oa,u1))
           (q2_tamm_bab(p2_vb,h1_oa,h2_ob,sc) = cnewsc * q2_tmp_bab(p2_vb,h1_oa,h2_ob,u1))
          .execute();
        }
        ec.pg().barrier();
        
        auto cc_gs = std::chrono::high_resolution_clock::now();
        time_gs_norm  += std::chrono::duration_cast<std::chrono::duration<double>>((cc_gs - cc_t1)).count();
        total_time_gs   += std::chrono::duration_cast<std::chrono::duration<double>>((cc_gs - cc_t0)).count();
    
       } //end of Gram-Schmidt loop

       sch.deallocate(q1_tmp_a,q2_tmp_aaa,q2_tmp_bab).execute();

       write_to_disk(q1_tamm_a,   q1afile);
       write_to_disk(q2_tamm_aaa, q2aaafile);
       write_to_disk(q2_tamm_bab, q2babfile);
      } //end gs-restart

      else { //restart GS
        read_from_disk(q1_tamm_a,   q1afile);
        read_from_disk(q2_tamm_aaa, q2aaafile);
        read_from_disk(q2_tamm_bab, q2babfile);
      }
      
      if(rank == 0) {
        cout << endl << "Time for orthogonalization: " << time_gs_orth << " secs" << endl;
        cout << endl << "Time for normalizing and copying back: " << time_gs_norm << " secs" << endl;
        cout << endl << "Total time for Gram-Schmidt: " << total_time_gs << " secs" << endl;
      }
      auto cc_gs_x = std::chrono::high_resolution_clock::now();

      bool gs_x12_restart = fs::exists(hx1afile) && fs::exists(hx2aaafile) && fs::exists(hx2babfile);

      if(!gs_x12_restart){
        #if GF_IN_SG
         if(subcomm != MPI_COMM_NULL){           
          gfccsd_x1(sub_sch,
        #else 
          gfccsd_x1(sch,
        #endif
                    MO, Hx1_tamm_a, 
                    d_t1_a, d_t1_b, d_t2_aaaa, d_t2_bbbb, d_t2_abab,
                    q1_tamm_a, q2_tamm_aaa, q2_tamm_bab, 
                    d_f1, //d_v2, 
                    otis);

        #if GF_IN_SG
          gfccsd_x2(sub_sch,
        #else 
          gfccsd_x2(sch,
        #endif
                    MO, Hx2_tamm_aaa, Hx2_tamm_bab, 
                    d_t1_a, d_t1_b, d_t2_aaaa, d_t2_bbbb, d_t2_abab,
                    q1_tamm_a, q2_tamm_aaa, q2_tamm_bab, 
                    d_f1, //d_v2, 
                    otis);

        #if GF_IN_SG
          sub_sch.execute();    
         }
         ec.pg().barrier();
        #else 
          sch.execute();    
        #endif
        write_to_disk(Hx1_tamm_a,  hx1afile);
        write_to_disk(Hx2_tamm_aaa,hx2aaafile);
        write_to_disk(Hx2_tamm_bab,hx2babfile);
      }
      else {
        read_from_disk(Hx1_tamm_a,  hx1afile);
        read_from_disk(Hx2_tamm_aaa,hx2aaafile);
        read_from_disk(Hx2_tamm_bab,hx2babfile);
      }
      auto cc_q12 = std::chrono::high_resolution_clock::now();
      double time_q12  = std::chrono::duration_cast<std::chrono::duration<double>>((cc_q12 - cc_gs_x)).count();
      if(rank == 0) cout << endl << "Time to contract Q1/Q2: " << time_q12 << " secs" << endl;

    } //if !gf_restart

      prev_qr_rank = qr_rank;
      
      auto cc_t1 = std::chrono::high_resolution_clock::now();

      auto [otil,otil1,otil2] = otis.labels<3>("all");
      ComplexTensor hsub_tamm{otis,otis};  
      ComplexTensor bsub_tamm_a{otis,o_alpha};  
      ComplexTensor Cp_a{o_alpha,otis};
      ComplexTensor::allocate(&ec,hsub_tamm,bsub_tamm_a,Cp_a);

      if(!gf_restart){

        ComplexTensor p1_k_a{v_alpha,otis};       
        ComplexTensor q1_conj_a   = tamm::conj(q1_tamm_a  );
        ComplexTensor q2_conj_aaa = tamm::conj(q2_tamm_aaa);
        ComplexTensor q2_conj_bab = tamm::conj(q2_tamm_bab);
        
        sch
           (bsub_tamm_a(otil1,h1_oa)  =       q1_conj_a(h1_oa,otil1))
           (hsub_tamm(otil1,otil2)    =       q1_conj_a(h1_oa,otil1) * Hx1_tamm_a(h1_oa,otil2))
           (hsub_tamm(otil1,otil2)   +=       q2_conj_aaa(p1_va,h1_oa,h2_oa,otil1) * Hx2_tamm_aaa(p1_va,h1_oa,h2_oa,otil2))
           (hsub_tamm(otil1,otil2)   += 2.0 * q2_conj_bab(p1_vb,h1_oa,h2_ob,otil1) * Hx2_tamm_bab(p1_vb,h1_oa,h2_ob,otil2))
           .deallocate(q1_conj_a,q2_conj_aaa,q2_conj_bab)
          //  .execute();        

           .allocate(p1_k_a)
           ( Cp_a(h1_oa,otil)    =        q1_tamm_a(h1_oa,otil)                                     )
           ( Cp_a(h2_oa,otil)   += -1.0 * lt12_a(h1_oa,h2_oa) * q1_tamm_a(h1_oa,otil)               )
           ( Cp_a(h2_oa,otil)   +=        d_t1_a(p1_va,h1_oa) * q2_tamm_aaa(p1_va,h2_oa,h1_oa,otil) )
           ( Cp_a(h2_oa,otil)   +=        d_t1_b(p1_vb,h1_ob) * q2_tamm_bab(p1_vb,h2_oa,h1_ob,otil) )
           ( p1_k_a(p1_va,otil)  =        d_t2_aaaa(p1_va,p2_va,h1_oa,h2_oa) * q2_tamm_aaa(p2_va,h1_oa,h2_oa,otil) )
           ( p1_k_a(p1_va,otil) +=  2.0 * d_t2_abab(p1_va,p2_vb,h1_oa,h2_ob) * q2_tamm_bab(p2_vb,h1_oa,h2_ob,otil) )
           ( Cp_a(h1_oa,otil)   += -0.5 * p1_k_a(p1_va,otil) * d_t1_a(p1_va,h1_oa) )
           .deallocate(p1_k_a)
           .execute();

      } //if !gf_restart

      auto cc_t2 = std::chrono::high_resolution_clock::now();
      auto time  = std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();

      if(rank == 0) cout << endl << "Time to compute Cp: " << time << " secs" << endl;

      //Write all tensors
      if(!gf_restart) {
        write_to_disk(hsub_tamm,   hsubfile);
        write_to_disk(bsub_tamm_a, bsubafile);
        write_to_disk(Cp_a,        cpafile);
        
        sch.deallocate(q1_tamm_a, q2_tamm_aaa, q2_tamm_bab,
                       Hx1_tamm_a,Hx2_tamm_aaa,Hx2_tamm_bab
                      ).execute();

      }
      else {
        read_from_disk(hsub_tamm,   hsubfile);
        read_from_disk(bsub_tamm_a, bsubafile);
        read_from_disk(Cp_a,        cpafile);
      }      

      Complex2DMatrix hsub(qr_rank,qr_rank);
      Complex2DMatrix bsub_a(qr_rank,noa);

      tamm_to_eigen_tensor(hsub_tamm,hsub);
      tamm_to_eigen_tensor(bsub_tamm_a,bsub_a);
      Complex2DMatrix hident = Complex2DMatrix::Identity(hsub.rows(),hsub.cols());

      ComplexTensor xsub_local_a{otis,o_alpha};
      ComplexTensor o_local_a{o_alpha};
      ComplexTensor Cp_local_a{o_alpha,otis};
      
      ComplexTensor::allocate(&ec_l,xsub_local_a,o_local_a,Cp_local_a);

      Scheduler sch_l{ec_l};
      sch_l
         (Cp_local_a(h1_oa,otil) = Cp_a(h1_oa,otil))
         .execute();

      if(rank==0) 
        cout << endl << "spectral function (omega_npts = " << omega_npts << "):" <<  endl;
      
      cc_t1 = std::chrono::high_resolution_clock::now();

      // Compute spectral function for designated omega regime
      for(int64_t ni=0;ni<omega_npts;ni++) {
          std::complex<T> omega_tmp =  std::complex<T>(omega_min + ni*omega_delta, -0.01);

          Complex2DMatrix xsub_a = (hsub + omega_tmp * hident).lu().solve(bsub_a);
          eigen_to_tamm_tensor(xsub_local_a,xsub_a);

          sch_l
             (o_local_a(h1_oa) = Cp_local_a(h1_oa,otil)*xsub_local_a(otil,h1_oa))
             .execute();
          
          auto oscalar  = std::imag(tamm::sum(o_local_a));
          
          if(level == 1) {
            omega_A0[ni] = oscalar;
          }
          else {
            if (level > 1) {
              T oerr = oscalar - omega_A0[ni];
              omega_A0[ni] = oscalar;
              if(std::abs(oerr) < gf_threshold) omega_conv[ni] = true; 
            }
          }
          if(rank==0){
            std::ostringstream spf;
            spf << "W = " << std::setprecision(12) << std::real(omega_tmp) << ", omega_A0 =  " << omega_A0[ni] << endl;
            cout << spf.str();
          }
      }

      cc_t2 = std::chrono::high_resolution_clock::now();
      time  = std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
      if(rank == 0) {
        cout << endl << "omegas processed in level " << level << " = " << omega_extra << endl;
        cout << "Time to compute spectral function (omega_npts = " << omega_npts << "): " 
                  << time << " secs" << endl;
      }

      auto extrap_file = files_prefix+".extrapolate.txt";
      std::ostringstream spfe;
      spfe << "";

      // extrapolate or proceed to next level
      if(std::all_of(omega_conv.begin(),omega_conv.end(), [](bool x){return x;}) || gf_extrapolate_level == level) {
        if(rank==0) cout << endl << "--------------------extrapolate & converge-----------------------" << endl;
        auto cc_t1 = std::chrono::high_resolution_clock::now();

        AtomicCounter* ac = new AtomicCounterGA(ec.pg(), 1);
        ac->allocate(0);
        int64_t taskcount = 0;
        int64_t next = ac->fetch_add(0, 1);

        for(int64_t ni=0;ni<lomega_npts;ni++) {
          if (next == taskcount) {
            std::complex<T> omega_tmp =  std::complex<T>(lomega_min + ni*omega_delta_e, -0.01);

            Complex2DMatrix xsub_a = (hsub + omega_tmp * hident).lu().solve(bsub_a);
            eigen_to_tamm_tensor(xsub_local_a,xsub_a);

            sch_l
             (o_local_a(h1_oa) = Cp_local_a(h1_oa,otil)*xsub_local_a(otil,h1_oa))
              .execute();
          
            auto oscalar  = std::imag(tamm::sum(o_local_a));
            
            //if(rank==0)            

          Eigen::Tensor<std::complex<T>, 1, Eigen::RowMajor> olocala_eig(noa);
          tamm_to_eigen_tensor(o_local_a,olocala_eig);
          for(TAMM_SIZE nj = 0; nj<noa; nj++){
            auto gpp = olocala_eig(nj).imag();
            spfe << "orb_index = " << nj << ", gpp = " << gpp << endl;
          }

            spfe << "W = " << std::setprecision(12) << std::real(omega_tmp) << ", lomega_A0 =  " << oscalar << endl;
            // cout << spfe.str();

            next = ac->fetch_add(0, 1); 
          }
          taskcount++;
        }
        
        ec.pg().barrier();
        ac->deallocate();
        delete ac;

        write_string_to_disk(ec,spfe.str(),extrap_file);

        sch.deallocate(xsub_local_a,o_local_a,Cp_local_a,
                       hsub_tamm,bsub_tamm_a,Cp_a).execute();  

        auto cc_t2 = std::chrono::high_resolution_clock::now();
        double time =
          std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
        if(rank == 0) std::cout << endl << 
        "Time taken for extrapolation (lomega_npts = " << lomega_npts << "): " << time << " secs" << endl;
          
        break;
      }
      else {
        if(level==1){
          auto o1 = (omega_extra[0] + omega_extra[1] ) / 2;
          omega_extra.clear();          
          o1 = find_closest(o1,omega_space);         
          omega_extra.push_back(o1);
        }
        else{
          std::sort(omega_extra_finished.begin(),omega_extra_finished.end());
          omega_extra.clear();
          std::vector<T> wtemp;
          for (size_t i=1;i<omega_extra_finished.size();i++){
            bool oe_add = false;
            auto w1 = omega_extra_finished[i-1];
            auto w2 = omega_extra_finished[i];
            size_t num_w = (w2-w1)/omega_delta + 1;
            for(size_t j=0;j<num_w;j++){
              T otmp = w1 + j*omega_delta;
              size_t ind = (otmp - omega_min)/omega_delta;
              if (!omega_conv[ind]) { oe_add = true; break; }
            }

            if(oe_add){
              T Win = (w1+w2)/2;
              
              Win = find_closest(Win,omega_space);
              if (std::find(omega_extra_finished.begin(),omega_extra_finished.end(),Win) != omega_extra_finished.end()){
                if(rank==0){
                  cout << "WIN = " << Win << endl;
                  cout << "Omega-combined = ";
                  cout << omega_extra_finished << endl;
                }
                //TODO: deallocate all tensors
                nwx_terminate("GFCCSD-MOR: Need higher resolution or different frequency region!");
              }
              omega_extra.push_back(Win);

            } //end oe add
          } //end oe finished
        }
        level++;
      
      }

    sch.deallocate(xsub_local_a,o_local_a,Cp_local_a,
                   hsub_tamm,bsub_tamm_a,Cp_a).execute();  

    } //end while
  #endif

    free_tensors(lt12_a,t2v2,
                 d_t1_a,d_t1_b,d_t2_aaaa,d_t2_bbbb,d_t2_abab,
                 v2_aaaa,v2_bbbb,v2_abab,
                 d_f1,//d_v2,
                //  ix1_1_a, 
                 ix1_1_1_a, ix1_1_1_b, 
                //  ix1_4_aaaa, ix1_4_abab,
                 ix2_1_aaaa, ix2_1_abab, 
                 ix2_2_a, ix2_2_b, 
                 ix2_3_a, ix2_3_b, 
                 ix2_4_aaaa, ix2_4_abab, 
                 ix2_5_aaaa, ix2_5_abba, ix2_5_abab,
                 ix2_5_bbbb, ix2_5_baab,
                 ix2_6_2_a, ix2_6_2_b,
                 ix2_6_3_aaaa, ix2_6_3_abba, ix2_6_3_abab,
                 ix2_6_3_bbbb, ix2_6_3_baab
                ); 
    // std::for_each(x1_omega_a.begin(), x1_omega_a.begin()+nptsi, [](auto& t) { t.deallocate(); });
    // std::for_each(x2_omega_aaa.begin(), x2_omega_aaa.begin()+nptsi, [](auto& t) { t.deallocate(); })  ;
    // std::for_each(x2_omega_bab.begin(), x2_omega_bab.begin()+nptsi, [](auto& t) { t.deallocate(); })  ;

  cc_t2 = std::chrono::high_resolution_clock::now();

  ccsd_time =
    std::chrono::duration_cast<std::chrono::duration<T>>((cc_t2 - cc_t1)).count();
  if(rank==0) std::cout << std::endl << "Time taken for GF-CCSD: " << ccsd_time << " secs" << std::endl;

  // --------- END GF CCSD -----------
  // GA_Summarize(0);
  ec.flush_and_sync();
  MemoryManagerGA::destroy_coll(mgr);
  ec_l.flush_and_sync();
  MemoryManagerLocal::destroy_coll(mgr_l);  
  if(subcomm != MPI_COMM_NULL){
    (*sub_ec).flush_and_sync();
    MemoryManagerGA::destroy_coll(sub_mgr);
    MPI_Comm_free(&subcomm);
  }
  // delete ec;
}

#endif 
