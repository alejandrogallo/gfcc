#ifndef TAMM_METHODS_GFCCSD_IP_HPP_
#define TAMM_METHODS_GFCCSD_IP_HPP_

#include <algorithm>
#include <complex>
using namespace tamm;

template<typename T>
void gfccsd_x1_a(/* ExecutionContext& ec, */
               Scheduler& sch, const TiledIndexSpace& MO,
               Tensor<std::complex<T>>& i0_a,
               const Tensor<T>& t1_a,    const Tensor<T>& t1_b, 
               const Tensor<T>& t2_aaaa, const Tensor<T>& t2_bbbb, const Tensor<T>& t2_abab,
               const Tensor<std::complex<T>>& x1_a,
               const Tensor<std::complex<T>>& x2_aaa, const Tensor<std::complex<T>>& x2_bab,
               const Tensor<T>& f1, const Tensor<T>& ix2_2_a, 
               const Tensor<T>& ix1_1_1_a, const Tensor<T>& ix1_1_1_b,
               const Tensor<T>& ix2_6_3_aaaa, const Tensor<T>& ix2_6_3_abab,
               const TiledIndexSpace& gf_tis, bool has_tis, bool debug=false) {

    TiledIndexSpace o_alpha,v_alpha,o_beta,v_beta;

    const TiledIndexSpace &O = MO("occ");
    const TiledIndexSpace &V = MO("virt");

    const int otiles = O.num_tiles();
    const int vtiles = V.num_tiles();
    const int oatiles = MO("occ_alpha").num_tiles();
    const int obtiles = MO("occ_beta").num_tiles();
    const int vatiles = MO("virt_alpha").num_tiles();
    const int vbtiles = MO("virt_beta").num_tiles();

    o_alpha = {MO("occ"), range(oatiles)};
    v_alpha = {MO("virt"), range(vatiles)};
    o_beta = {MO("occ"), range(obtiles,otiles)};
    v_beta = {MO("virt"), range(vbtiles,vtiles)};

    auto [p7_va] = v_alpha.labels<1>("all");
    auto [p7_vb] = v_beta.labels<1>("all");
    auto [h1_oa,h6_oa,h8_oa] = o_alpha.labels<3>("all");
    auto [h1_ob,h6_ob,h8_ob] = o_beta.labels<3>("all");
    auto [u1] = gf_tis.labels<1>("all");

    if(has_tis) {
      sch
      ( i0_a(h1_oa,u1)  =  0 )
      ( i0_a(h1_oa,u1) += -1   * x1_a(h6_oa,u1) * ix2_2_a(h6_oa,h1_oa) )
      ( i0_a(h1_oa,u1) +=        x2_aaa(p7_va,h1_oa,h6_oa,u1) * ix1_1_1_a(h6_oa,p7_va) )
      ( i0_a(h1_oa,u1) +=        x2_bab(p7_vb,h1_oa,h6_ob,u1) * ix1_1_1_b(h6_ob,p7_vb) )
      ( i0_a(h1_oa,u1) +=  0.5 * x2_aaa(p7_va,h6_oa,h8_oa,u1) * ix2_6_3_aaaa(h6_oa,h8_oa,h1_oa,p7_va) )
      ( i0_a(h1_oa,u1) +=        x2_bab(p7_vb,h6_oa,h8_ob,u1) * ix2_6_3_abab(h6_oa,h8_ob,h1_oa,p7_vb) );         
    }
    else {
      sch
      ( i0_a(h1_oa)  =  0 )
      ( i0_a(h1_oa) += -1   * x1_a(h6_oa) * ix2_2_a(h6_oa,h1_oa) )
      ( i0_a(h1_oa) +=        x2_aaa(p7_va,h1_oa,h6_oa) * ix1_1_1_a(h6_oa,p7_va) )
      ( i0_a(h1_oa) +=        x2_bab(p7_vb,h1_oa,h6_ob) * ix1_1_1_b(h6_ob,p7_vb) )
      ( i0_a(h1_oa) +=  0.5 * x2_aaa(p7_va,h6_oa,h8_oa) * ix2_6_3_aaaa(h6_oa,h8_oa,h1_oa,p7_va) )
      ( i0_a(h1_oa) +=        x2_bab(p7_vb,h6_oa,h8_ob) * ix2_6_3_abab(h6_oa,h8_ob,h1_oa,p7_vb) );      
    }
    if(debug) sch.execute();

}

template<typename T>
void gfccsd_x2_a(/* ExecutionContext& ec, */
               Scheduler& sch, const TiledIndexSpace& MO,
               Tensor<std::complex<T>>& i0_aaa, Tensor<std::complex<T>>& i0_bab,
               const Tensor<T>& t1_a,    const Tensor<T>& t1_b,
               const Tensor<T>& t2_aaaa, const Tensor<T>& t2_bbbb, const Tensor<T>& t2_abab,
               const Tensor<std::complex<T>>& x1_a,
               const Tensor<std::complex<T>>& x2_aaa, const Tensor<std::complex<T>>& x2_bab,
               const Tensor<T>& f1, const Tensor<T>& ix2_1_aaaa, const Tensor<T>& ix2_1_abab,
               const Tensor<T>& ix2_2_a, const Tensor<T>& ix2_2_b,
               const Tensor<T>& ix2_3_a, const Tensor<T>& ix2_3_b, 
               const Tensor<T>& ix2_4_aaaa, const Tensor<T>& ix2_4_abab,
               const Tensor<T>& ix2_5_aaaa, const Tensor<T>& ix2_5_abba, const Tensor<T>& ix2_5_abab, 
               const Tensor<T>& ix2_5_bbbb, const Tensor<T>& ix2_5_baab,
               const Tensor<T>& ix2_6_2_a, const Tensor<T>& ix2_6_2_b, 
               const Tensor<T>& ix2_6_3_aaaa, const Tensor<T>& ix2_6_3_abba, const Tensor<T>& ix2_6_3_abab,
               const Tensor<T>& ix2_6_3_bbbb, const Tensor<T>& ix2_6_3_baab,
               const Tensor<T>& v2ijab_aaaa, const Tensor<T>& v2ijab_abab, const Tensor<T>& v2ijab_bbbb,
               const TiledIndexSpace& gf_tis, bool has_tis, bool debug=false) {

    using ComplexTensor = Tensor<std::complex<T>>;
    TiledIndexSpace o_alpha,v_alpha,o_beta,v_beta;

    const TiledIndexSpace &O = MO("occ");
    const TiledIndexSpace &V = MO("virt");
    auto [u1] = gf_tis.labels<1>("all");

    const int otiles = O.num_tiles();
    const int vtiles = V.num_tiles();
    const int oatiles = MO("occ_alpha").num_tiles();
    const int obtiles = MO("occ_beta").num_tiles();
    const int vatiles = MO("virt_alpha").num_tiles();
    const int vbtiles = MO("virt_beta").num_tiles();

    o_alpha = {MO("occ"), range(oatiles)};
    v_alpha = {MO("virt"), range(vatiles)};
    o_beta = {MO("occ"), range(obtiles,otiles)};
    v_beta = {MO("virt"), range(vbtiles,vtiles)};

    auto [p3_va,p4_va,p5_va,p8_va,p9_va] = v_alpha.labels<5>("all");
    auto [p3_vb,p4_vb,p5_vb,p8_vb,p9_vb] = v_beta.labels<5>("all");
    auto [h1_oa,h2_oa,h6_oa,h7_oa,h8_oa,h9_oa,h10_oa] = o_alpha.labels<7>("all");
    auto [h1_ob,h2_ob,h6_ob,h7_ob,h8_ob,h9_ob,h10_ob] = o_beta.labels<7>("all");

    ComplexTensor i_6_aaa     ;                                      
    ComplexTensor i_6_bab     ;                                     
    ComplexTensor i_10_a      ;                      
    ComplexTensor i_11_aaa    ;                                      
    ComplexTensor i_11_bab    ;                                     
    ComplexTensor i_11_bba    ;                                      
    ComplexTensor i0_temp_aaa ;                                      
    ComplexTensor i0_temp_bab ;                                     
    ComplexTensor i0_temp_bba ;                                      
    ComplexTensor i_6_temp_aaa;                                      
    ComplexTensor i_6_temp_bab;                                     
    ComplexTensor i_6_temp_bba;        

  if(has_tis){
     i_6_aaa      = {o_alpha,o_alpha,o_alpha,gf_tis};
     i_6_bab      = {o_beta, o_alpha,o_beta, gf_tis};
     i_10_a       = {v_alpha,gf_tis};
     i_11_aaa     = {o_alpha,o_alpha,v_alpha,gf_tis};
     i_11_bab     = {o_beta, o_alpha,v_beta, gf_tis};
     i_11_bba     = {o_beta, o_beta, v_alpha,gf_tis};
     i0_temp_aaa  = {v_alpha,o_alpha,o_alpha,gf_tis};
     i0_temp_bab  = {v_beta, o_alpha,o_beta, gf_tis};
     i0_temp_bba  = {v_beta, o_beta, o_alpha,gf_tis};
     i_6_temp_aaa = {o_alpha,o_alpha,o_alpha,gf_tis};
     i_6_temp_bab = {o_beta, o_alpha,o_beta, gf_tis};
     i_6_temp_bba = {o_beta, o_beta, o_alpha,gf_tis};
  }
  else {
    i_6_aaa      = {o_alpha,o_alpha,o_alpha};   
    i_6_bab      = {o_beta, o_alpha,o_beta};   
    i_10_a       = {v_alpha};                    
    i_11_aaa     = {o_alpha,o_alpha,v_alpha};
    i_11_bab     = {o_beta, o_alpha,v_beta};
    i_11_bba     = {o_beta, o_beta, v_alpha};
    i0_temp_aaa  = {v_alpha,o_alpha,o_alpha};
    i0_temp_bab  = {v_beta, o_alpha,o_beta};
    i0_temp_bba  = {v_beta, o_beta, o_alpha};
    i_6_temp_aaa = {o_alpha,o_alpha,o_alpha};
    i_6_temp_bab = {o_beta, o_alpha,o_beta};
    i_6_temp_bba = {o_beta, o_beta, o_alpha};
  }
    sch
    .allocate(i_6_aaa,i_6_bab,
              i_10_a,i_11_aaa,i_11_bab,i_11_bba,
              i0_temp_aaa,i0_temp_bab,i0_temp_bba,
              i_6_temp_aaa,i_6_temp_bab,i_6_temp_bba);
    if(has_tis) {
      sch( i0_aaa(p4_va,h1_oa,h2_oa,u1)    =  0 )
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
  
      (   i_11_aaa(h6_oa,h1_oa,p5_va,u1)   =  x2_aaa(p8_va,h1_oa,h7_oa,u1) * v2ijab_aaaa(h6_oa,h7_oa,p5_va,p8_va) )
      (   i_11_aaa(h6_oa,h1_oa,p5_va,u1)  +=  x2_bab(p8_vb,h1_oa,h7_ob,u1) * v2ijab_abab(h6_oa,h7_ob,p5_va,p8_vb) )
      (   i_11_bab(h6_ob,h1_oa,p5_vb,u1)   =  x2_bab(p8_vb,h1_oa,h7_ob,u1) * v2ijab_bbbb(h6_ob,h7_ob,p5_vb,p8_vb) )
      (   i_11_bab(h6_ob,h1_oa,p5_vb,u1)  +=  x2_aaa(p8_va,h1_oa,h7_oa,u1) * v2ijab_abab(h7_oa,h6_ob,p8_va,p5_vb) )
      (   i_11_bba(h6_ob,h1_ob,p5_va,u1)   =  x2_bab(p8_vb,h7_oa,h1_ob,u1) * v2ijab_abab(h7_oa,h6_ob,p5_va,p8_vb) )
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
  
      (   i_10_a(p5_va,u1)  =  0.5 * x2_aaa(p8_va,h6_oa,h7_oa,u1) * v2ijab_aaaa(h6_oa,h7_oa,p5_va,p8_va) )
      (   i_10_a(p5_va,u1) +=        x2_bab(p8_vb,h6_oa,h7_ob,u1) * v2ijab_abab(h6_oa,h7_ob,p5_va,p8_vb) )
      ( i0_aaa(p3_va,h1_oa,h2_oa,u1) +=      t2_aaaa(p3_va,p5_va,h1_oa,h2_oa) * i_10_a(p5_va,u1) )
      ( i0_bab(p3_vb,h1_oa,h2_ob,u1) += -1 * t2_abab(p5_va,p3_vb,h1_oa,h2_ob) * i_10_a(p5_va,u1) );
    }
    else {
      sch( i0_aaa(p4_va,h1_oa,h2_oa)    =  0 )
      ( i0_bab(p4_vb,h1_oa,h2_ob)    =  0 )
  
      ( i0_aaa(p4_va,h1_oa,h2_oa)      +=  x1_a(h9_oa) * ix2_1_aaaa(h9_oa,p4_va,h1_oa,h2_oa) ) 
      ( i0_bab(p4_vb,h1_oa,h2_ob)      +=  x1_a(h9_oa) * ix2_1_abab(h9_oa,p4_vb,h1_oa,h2_ob) )
      
      ( i0_temp_aaa(p3_va,h1_oa,h2_oa)  =       x2_aaa(p3_va,h1_oa,h8_oa) * ix2_2_a(h8_oa,h2_oa) )
      ( i0_temp_bab(p3_vb,h1_oa,h2_ob)  =       x2_bab(p3_vb,h1_oa,h8_ob) * ix2_2_b(h8_ob,h2_ob) )
      ( i0_temp_bba(p3_vb,h1_ob,h2_oa)  =  -1 * x2_bab(p3_vb,h8_oa,h1_ob) * ix2_2_a(h8_oa,h2_oa) )
  
      ( i0_temp_aaa(p4_va,h1_oa,h2_oa) +=       x2_aaa(p8_va,h1_oa,h7_oa) * ix2_5_aaaa(h7_oa,p4_va,h2_oa,p8_va) ) //O3V2
      ( i0_temp_aaa(p4_va,h1_oa,h2_oa) +=       x2_bab(p8_vb,h1_oa,h7_ob) * ix2_5_baab(h7_ob,p4_va,h2_oa,p8_vb) )
      ( i0_temp_bab(p4_vb,h1_oa,h2_ob) +=       x2_bab(p8_vb,h1_oa,h7_ob) * ix2_5_bbbb(h7_ob,p4_vb,h2_ob,p8_vb) )
      ( i0_temp_bab(p4_vb,h1_oa,h2_ob) +=       x2_aaa(p8_va,h1_oa,h7_oa) * ix2_5_abba(h7_oa,p4_vb,h2_ob,p8_va) )
      ( i0_temp_bba(p4_vb,h1_ob,h2_oa) +=  -1 * x2_bab(p8_vb,h7_oa,h1_ob) * ix2_5_abab(h7_oa,p4_vb,h2_oa,p8_vb) )
  
      (   i_11_aaa(h6_oa,h1_oa,p5_va)   =  x2_aaa(p8_va,h1_oa,h7_oa) * v2ijab_aaaa(h6_oa,h7_oa,p5_va,p8_va) )
      (   i_11_aaa(h6_oa,h1_oa,p5_va)  +=  x2_bab(p8_vb,h1_oa,h7_ob) * v2ijab_abab(h6_oa,h7_ob,p5_va,p8_vb) )
      (   i_11_bab(h6_ob,h1_oa,p5_vb)   =  x2_bab(p8_vb,h1_oa,h7_ob) * v2ijab_bbbb(h6_ob,h7_ob,p5_vb,p8_vb) )
      (   i_11_bab(h6_ob,h1_oa,p5_vb)  +=  x2_aaa(p8_va,h1_oa,h7_oa) * v2ijab_abab(h7_oa,h6_ob,p8_va,p5_vb) )
      (   i_11_bba(h6_ob,h1_ob,p5_va)   =  x2_bab(p8_vb,h7_oa,h1_ob) * v2ijab_abab(h7_oa,h6_ob,p5_va,p8_vb) )
      ( i0_temp_aaa(p3_va,h2_oa,h1_oa) += -1 * t2_aaaa(p3_va,p5_va,h1_oa,h6_oa) * i_11_aaa(h6_oa,h2_oa,p5_va) )
      ( i0_temp_aaa(p3_va,h2_oa,h1_oa) += -1 * t2_abab(p3_va,p5_vb,h1_oa,h6_ob) * i_11_bab(h6_ob,h2_oa,p5_vb) )
      ( i0_temp_bab(p3_vb,h2_oa,h1_ob) += -1 * t2_bbbb(p3_vb,p5_vb,h1_ob,h6_ob) * i_11_bab(h6_ob,h2_oa,p5_vb) )
      ( i0_temp_bab(p3_vb,h2_oa,h1_ob) += -1 * t2_abab(p5_va,p3_vb,h6_oa,h1_ob) * i_11_aaa(h6_oa,h2_oa,p5_va) )
      ( i0_temp_bba(p3_vb,h2_ob,h1_oa) +=      t2_abab(p5_va,p3_vb,h1_oa,h6_ob) * i_11_bba(h6_ob,h2_ob,p5_va) )
      
      ( i0_aaa(p3_va,h1_oa,h2_oa) += -1 * i0_temp_aaa(p3_va,h1_oa,h2_oa) )
      ( i0_aaa(p3_va,h2_oa,h1_oa) +=      i0_temp_aaa(p3_va,h1_oa,h2_oa) )
      ( i0_bab(p3_vb,h1_oa,h2_ob) += -1 * i0_temp_bab(p3_vb,h1_oa,h2_ob) )
      ( i0_bab(p3_vb,h2_oa,h1_ob) +=      i0_temp_bba(p3_vb,h1_ob,h2_oa) )
  
      ( i0_aaa(p4_va,h1_oa,h2_oa) +=  x2_aaa(p8_va,h1_oa,h2_oa) * ix2_3_a(p4_va,p8_va) )
      ( i0_bab(p4_vb,h1_oa,h2_ob) +=  x2_bab(p8_vb,h1_oa,h2_ob) * ix2_3_b(p4_vb,p8_vb) )
  
      ( i0_aaa(p3_va,h1_oa,h2_oa) +=  0.5 * x2_aaa(p3_va,h9_oa,h10_oa) * ix2_4_aaaa(h9_oa,h10_oa,h1_oa,h2_oa) ) //O4V
      ( i0_bab(p3_vb,h1_oa,h2_ob) +=        x2_bab(p3_vb,h9_oa,h10_ob) * ix2_4_abab(h9_oa,h10_ob,h1_oa,h2_ob) )
  
      (   i_6_aaa(h10_oa,h1_oa,h2_oa)  = -1 * x1_a(h8_oa) * ix2_4_aaaa(h8_oa,h10_oa,h1_oa,h2_oa) )
      (   i_6_bab(h10_ob,h1_oa,h2_ob)  = -1 * x1_a(h8_oa) * ix2_4_abab(h8_oa,h10_ob,h1_oa,h2_ob) )
      
      (   i_6_aaa(h10_oa,h1_oa,h2_oa) +=  x2_aaa(p5_va,h1_oa,h2_oa) * ix2_6_2_a(h10_oa,p5_va) )
      (   i_6_bab(h10_ob,h1_oa,h2_ob) +=  x2_bab(p5_vb,h1_oa,h2_ob) * ix2_6_2_b(h10_ob,p5_vb) )
      
      (   i_6_temp_aaa(h10_oa,h1_oa,h2_oa)  =  x2_aaa(p9_va,h2_oa,h8_oa) * ix2_6_3_aaaa(h8_oa,h10_oa,h1_oa,p9_va) )
      (   i_6_temp_aaa(h10_oa,h1_oa,h2_oa) +=  x2_bab(p9_vb,h2_oa,h8_ob) * ix2_6_3_baab(h8_ob,h10_oa,h1_oa,p9_vb) ) 
  
      (   i_6_temp_bab(h10_ob,h1_oa,h2_ob)  =  -1 * x2_bab(p9_vb,h8_oa,h2_ob) * ix2_6_3_abab(h8_oa,h10_ob,h1_oa,p9_vb) )
      (   i_6_temp_bba(h10_ob,h1_ob,h2_oa)  =       x2_bab(p9_vb,h2_oa,h8_ob) * ix2_6_3_bbbb(h8_ob,h10_ob,h1_ob,p9_vb) )
      (   i_6_temp_bba(h10_ob,h1_ob,h2_oa) +=       x2_aaa(p9_va,h2_oa,h8_oa) * ix2_6_3_abba(h8_oa,h10_ob,h1_ob,p9_va) )
  
      (   i_6_aaa(h10_oa,h1_oa,h2_oa) += -1 * i_6_temp_aaa(h10_oa,h1_oa,h2_oa) )
      (   i_6_aaa(h10_oa,h2_oa,h1_oa) +=      i_6_temp_aaa(h10_oa,h1_oa,h2_oa) )
      (   i_6_bab(h10_ob,h1_oa,h2_ob) += -1 * i_6_temp_bab(h10_ob,h1_oa,h2_ob) )
      (   i_6_bab(h10_ob,h2_oa,h1_ob) +=      i_6_temp_bba(h10_ob,h1_ob,h2_oa) )
      
      ( i0_aaa(p3_va,h1_oa,h2_oa)  +=  t1_a(p3_va,h10_oa) * i_6_aaa(h10_oa,h1_oa,h2_oa) )
      ( i0_bab(p3_vb,h1_oa,h2_ob)  +=  t1_b(p3_vb,h10_ob) * i_6_bab(h10_ob,h1_oa,h2_ob) )

      (   i_10_a(p5_va)  =  0.5 * x2_aaa(p8_va,h6_oa,h7_oa) * v2ijab_aaaa(h6_oa,h7_oa,p5_va,p8_va) )
      (   i_10_a(p5_va) +=        x2_bab(p8_vb,h6_oa,h7_ob) * v2ijab_abab(h6_oa,h7_ob,p5_va,p8_vb) )
      ( i0_aaa(p3_va,h1_oa,h2_oa) +=      t2_aaaa(p3_va,p5_va,h1_oa,h2_oa) * i_10_a(p5_va) )
      ( i0_bab(p3_vb,h1_oa,h2_ob) += -1 * t2_abab(p5_va,p3_vb,h1_oa,h2_ob) * i_10_a(p5_va) );
    }
    sch.deallocate(i_6_aaa,i_6_bab,
                i_10_a,i_11_aaa,i_11_bab,i_11_bba,
                i0_temp_aaa,i0_temp_bab,i0_temp_bba,
                i_6_temp_aaa,i_6_temp_bab,i_6_temp_bba);
    if(debug) sch.execute();

}

#endif //TAMM_METHODS_GFCCSD_IP_HPP_
