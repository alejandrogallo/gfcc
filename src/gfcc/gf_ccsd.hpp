#ifndef GFCCSD_HPP_
#define GFCCSD_HPP_

#include "contrib/cd_ccsd_common.hpp"
#include "gf_guess.hpp"
#include "gfccsd_ip.hpp"
#include <algorithm>
#undef I

using namespace tamm;

#include <filesystem>
namespace fs = std::filesystem;

//TODO input file
size_t  ndiis;
size_t  ngmres;
size_t  gf_maxiter;

int gf_nprocs_poi;
double  gf_omega;
size_t  p_oi; //number of occupied/all MOs
double  gf_eta;
double  gf_threshold;
double  omega_min_ip;
double  omega_max_ip;
double  lomega_min_ip;
double  lomega_max_ip;
double  omega_delta;
int64_t omega_npts_ip;
int64_t lomega_npts_ip;
double  omega_delta_e;
double  gf_level_shift;
double  gf_damping_factor;
int gf_extrapolate_level;
int gf_analyze_level;
int gf_analyze_num_omega;
std::vector<double> gf_analyze_omega;

#define GF_PGROUPS 1
#define GF_IN_SG 0
#define GF_GS_SG 0

// Tensor<double> lambda_y1, lambda_y2,
// Tensor<double> d_t1_a, d_t1_b,
//                d_t2_aaaa, d_t2_bbbb, d_t2_abab,
//                v2ijab_aaaa, v2ijab_abab, v2ijab_bbbb,
//                cholOO_a, cholOO_b, cholOV_a, cholOV_b, cholVV_a, cholVV_b;

Tensor<double> 
               t2v2_o,
               lt12_o_a, lt12_o_b,
               ix1_1_1_a, ix1_1_1_b, 
               ix2_1_aaaa, ix2_1_abab, ix2_1_bbbb, ix2_1_baba,
               ix2_2_a, ix2_2_b, 
               ix2_3_a, ix2_3_b, 
               ix2_4_aaaa, ix2_4_abab, ix2_4_bbbb, 
               ix2_5_aaaa, ix2_5_abba, ix2_5_abab, 
               ix2_5_bbbb, ix2_5_baab, ix2_5_baba,
               ix2_6_2_a, ix2_6_2_b,
               ix2_6_3_aaaa, ix2_6_3_abba, ix2_6_3_abab, 
               ix2_6_3_bbbb, ix2_6_3_baab, ix2_6_3_baba,
               t2v2_v,
               lt12_v_a, lt12_v_b,
               iy1_1_a, iy1_1_b, 
               iy1_2_1_a,iy1_2_1_b,
               iy1_a, iy1_b,
               iy2_a, iy2_b,
               iy3_1_aaaa, iy3_1_abba, iy3_1_baba,
               iy3_1_bbbb, iy3_1_baab, iy3_1_abab,
               iy3_1_2_a,iy3_1_2_b,
               iy3_aaaa, iy3_baab, iy3_abba,
               iy3_bbbb, iy3_baba, iy3_abab,
               iy4_1_aaaa, iy4_1_baab, iy4_1_baba,
               iy4_1_bbbb, iy4_1_abba, iy4_1_abab,
               iy4_2_aaaa, iy4_2_abba, iy4_2_bbbb, iy4_2_baab,
               iy5_aaaa, iy5_baab, iy5_baba,
               iy5_abba, iy5_bbbb, iy5_abab,
               iy6_a, iy6_b
               ;

TiledIndexSpace diis_tis;
// std::ofstream ofs_profile;

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

template<typename ... Ts>
std::string gfacc_str(Ts&&...args)
{
    std::string res;
    (res.append(args), ...);
    res.append("\n");
    return res;
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
void gfccsd_driver_ip_a(ExecutionContext& gec, ExecutionContext& sub_ec, MPI_Comm &subcomm,
                   const TiledIndexSpace& MO, Tensor<T>& t1_a,   Tensor<T>& t1_b, 
                   Tensor<T>& t2_aaaa, Tensor<T>& t2_bbbb, Tensor<T>& t2_abab,
                   Tensor<T>& f1, Tensor<T>& t2v2_o,
                   Tensor<T>& lt12_o_a, Tensor<T>& lt12_o_b,
                   Tensor<T>& ix1_1_1_a, Tensor<T>& ix1_1_1_b,
                   Tensor<T>& ix2_1_aaaa, Tensor<T>& ix2_1_abab, Tensor<T>& ix2_1_bbbb, Tensor<T>& ix2_1_baba,
                   Tensor<T>& ix2_2_a, Tensor<T>& ix2_2_b, 
                   Tensor<T>& ix2_3_a, Tensor<T>& ix2_3_b, 
                   Tensor<T>& ix2_4_aaaa, Tensor<T>& ix2_4_abab, Tensor<T>& ix2_4_bbbb, 
                   Tensor<T>& ix2_5_aaaa, Tensor<T>& ix2_5_abba, Tensor<T>& ix2_5_abab, 
                   Tensor<T>& ix2_5_bbbb, Tensor<T>& ix2_5_baab, Tensor<T>& ix2_5_baba,
                   Tensor<T>& ix2_6_2_a, Tensor<T>& ix2_6_2_b, 
                   Tensor<T>& ix2_6_3_aaaa, Tensor<T>& ix2_6_3_abba, Tensor<T>& ix2_6_3_abab,
                   Tensor<T>& ix2_6_3_bbbb, Tensor<T>& ix2_6_3_baab, Tensor<T>& ix2_6_3_baba,
                   Tensor<T>& v2ijab_aaaa, Tensor<T>& v2ijab_abab, Tensor<T>& v2ijab_bbbb,
                   std::vector<T>& p_evl_sorted_occ, std::vector<T>& p_evl_sorted_virt,
                   long int total_orbitals, const TAMM_SIZE nocc,const TAMM_SIZE nvir,
                   size_t& nptsi, const TiledIndexSpace& unit_tis,string files_prefix,
                   string levelstr, int noa) {


  using ComplexTensor = Tensor<std::complex<T>>;
  using VComplexTensor = std::vector<Tensor<std::complex<T>>>;
  using CMatrix = Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  const TiledIndexSpace& O = MO("occ");
  const TiledIndexSpace& V = MO("virt");
  const TiledIndexSpace& N = MO("all");
  // auto [u1] = unit_tis.labels<1>("all");

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

  auto [p1_va] = v_alpha.labels<1>("all");
  auto [p1_vb] = v_beta.labels<1>("all");
  auto [h1_oa,h2_oa] = o_alpha.labels<2>("all");
  auto [h1_ob,h2_ob] = o_beta.labels<2>("all");

  std::cout.precision(15);

  Scheduler gsch{gec};
  auto rank = gec.pg().rank();

  std::stringstream gfo;
  gfo << std::defaultfloat << gf_omega;

  // PRINT THE HEADER FOR GF-CCSD ITERATIONS
  if(rank == 0) {
    std::stringstream gfp;
    gfp << std::endl << std::string(55, '-') << std::endl << "GF-CCSD (w = " << gfo.str() << ") " << std::endl;
    std::cout << gfp.str() << std::flush;
  }

  ComplexTensor dtmp_aaa{v_alpha,o_alpha,o_alpha};
  ComplexTensor dtmp_bab{v_beta, o_alpha,o_beta};  
  ComplexTensor::allocate(&gec,dtmp_aaa,dtmp_bab);
  
  // double au2ev = 27.2113961;

  std::string dtmp_aaa_file = files_prefix+".W"+gfo.str()+".r_dtmp_aaa.l"+levelstr;
  std::string dtmp_bab_file = files_prefix+".W"+gfo.str()+".r_dtmp_bab.l"+levelstr;

  if (fs::exists(dtmp_aaa_file) && fs::exists(dtmp_bab_file)) {
    read_from_disk(dtmp_aaa,dtmp_aaa_file);
    read_from_disk(dtmp_bab,dtmp_bab_file);
  }
  else {
    ComplexTensor DEArr_IP{V,O,O};
    
    double denominator = 0.0;
    const double lshift1 = 0; 
    const double lshift2 = 0.50000000;
    auto DEArr_lambda = [&](const IndexVector& bid) {
      const IndexVector blockid = internal::translate_blockid(bid, DEArr_IP());
      const TAMM_SIZE size = DEArr_IP.block_size(blockid);
      std::vector<std::complex<T>> buf(size);
      auto block_dims   = DEArr_IP.block_dims(blockid);
      auto block_offset = DEArr_IP.block_offsets(blockid);
      size_t c = 0;
      for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0]; i++) {
      for(size_t j = block_offset[1]; j < block_offset[1] + block_dims[1]; j++) {
      for(size_t k = block_offset[2]; k < block_offset[2] + block_dims[2]; k++, c++) {
        denominator = gf_omega + p_evl_sorted_virt[i] - p_evl_sorted_occ[j] 
                               - p_evl_sorted_occ[k];
        if (denominator < lshift1 && denominator > -1.0*lshift2){
            denominator += -1.0*lshift2;
        }
        else if (denominator > lshift1 && denominator < lshift2) {
            denominator += lshift2;
        }
        buf[c] = 1.0/std::complex<T>(denominator, -1.0*gf_eta);
      }
      }
      }
      DEArr_IP.put(blockid,buf);
    };

    gsch.allocate(DEArr_IP).execute();
    if(subcomm != MPI_COMM_NULL){
        Scheduler sub_sch{sub_ec};
        sub_sch(DEArr_IP() = 0).execute();        
        block_for(sub_ec, DEArr_IP(), DEArr_lambda);
        sub_sch      
          (dtmp_aaa() = 0)
          (dtmp_bab() = 0)
          (dtmp_aaa(p1_va,h1_oa,h2_oa) = DEArr_IP(p1_va,h1_oa,h2_oa))
          (dtmp_bab(p1_vb,h1_oa,h2_ob) = DEArr_IP(p1_vb,h1_oa,h2_ob))
          .execute();
    }
    gec.pg().barrier();
    gsch.deallocate(DEArr_IP).execute();
    write_to_disk(dtmp_aaa,dtmp_aaa_file);
    write_to_disk(dtmp_bab,dtmp_bab_file);
  }

  //------------------------
  auto nranks = GA_Nnodes();
  auto world_comm = gec.pg().comm();
  auto world_rank = gec.pg().rank().value();

  MPI_Group world_group;
  int world_size;
  MPI_Comm_group(world_comm,&world_group);
  MPI_Comm_size(world_comm,&world_size);
  MPI_Comm gf_comm;

  const size_t num_oi = noa;
  size_t num_pi_processed = 0;
  std::vector<size_t> pi_tbp;
  //Check pi's already processed
  for (size_t pi=0; pi < num_oi; pi++) {
    std::string x1_a_conv_wpi_file   = files_prefix+".x1_a.w"  +gfo.str()+".oi"+std::to_string(pi);
    std::string x2_aaa_conv_wpi_file = files_prefix+".x2_aaa.w"+gfo.str()+".oi"+std::to_string(pi);
    std::string x2_bab_conv_wpi_file = files_prefix+".x2_bab.w"+gfo.str()+".oi"+std::to_string(pi);

    if(fs::exists(x1_a_conv_wpi_file) && fs::exists(x2_aaa_conv_wpi_file) && fs::exists(x2_bab_conv_wpi_file)) 
      num_pi_processed++;
    else pi_tbp.push_back(pi);
  }

  size_t num_pi_remain = num_oi-num_pi_processed;
  if(num_pi_remain == 0) {
    gsch.deallocate(dtmp_aaa,dtmp_bab).execute();
    return;
  }
  EXPECTS(num_pi_remain == pi_tbp.size());
  //if(num_pi_remain == 0) num_pi_remain = 1;
  int subranks = std::floor(nranks/num_pi_remain);
  const bool no_pg=(subranks == 0 || subranks == 1);
  if(no_pg) subranks=nranks;
  if(gf_nprocs_poi > 0) subranks = gf_nprocs_poi;

  //Figure out how many orbitals in pi_tbp can be processed with subranks
  size_t num_oi_can_bp = std::ceil(nranks / (1.0*subranks));
  if(num_pi_remain < num_oi_can_bp) {
    num_oi_can_bp = num_pi_remain;
    subranks = std::floor(nranks/num_pi_remain);
    if(no_pg) subranks=nranks;
  }

  if(rank==0) {
    cout << "Total number of process groups = " << num_oi_can_bp << endl;
    cout << "Total, remaining orbitals, batch size = " << num_oi << ", " << num_pi_remain << ", " << num_oi_can_bp << endl;
    cout << "No of processes used to compute each orbital = " << subranks << endl;
    //ofs_profile << "No of processes used to compute each orbital = " << subranks << endl;
  }

  int color = 0;
  if(subranks > 1) color = world_rank/subranks;
  
  MPI_Comm_split(world_comm, color, world_rank, &gf_comm);

  ///////////////////////////
  //                       //
  //  MAIN ITERATION LOOP  //
  //        (alpha)        //
  ///////////////////////////
  auto cc_t1 = std::chrono::high_resolution_clock::now();

  #if GF_PGROUPS
    ProcGroup pg = ProcGroup::create_coll(gf_comm);
    ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};
    Scheduler sch{ec};
  #else
    Scheduler& sch = gsch;
    ExecutionContext& ec = gec;    
  #endif

  AtomicCounter* ac = new AtomicCounterGA(gec.pg(), 1);
  ac->allocate(0);
  int64_t taskcount = 0;
  int64_t next = -1; 
  int total_pi_pg = 0;

  int root_ppi = -1;
  MPI_Comm_rank( ec.pg().comm(), &root_ppi );
  int pg_id = rank.value()/subranks;
  if(root_ppi == 0) next = ac->fetch_add(0, 1);
  MPI_Bcast(&next        ,1,mpi_type<int64_t>(),0,ec.pg().comm());

  for (size_t piv=0; piv < pi_tbp.size(); piv++) {   
    
    if (next == taskcount) {
      total_pi_pg++;
      size_t pi = pi_tbp[piv];
      if(root_ppi==0 && debug) cout << "Process group " << pg_id << " is executing orbital " << pi << endl;

    auto gf_t1 = std::chrono::high_resolution_clock::now();

    bool gf_conv = false;
    ComplexTensor Minv{{O,O},{1,1}};
    ComplexTensor x1{O};
    Tensor<T> B1{O};
    
    ComplexTensor Hx1_a{o_alpha};
    ComplexTensor Hx2_aaa{v_alpha,o_alpha,o_alpha};
    ComplexTensor Hx2_bab{v_beta, o_alpha,o_beta};
    ComplexTensor x1_a{o_alpha};
    ComplexTensor x2_aaa{v_alpha,o_alpha,o_alpha};
    ComplexTensor x2_bab{v_beta, o_alpha,o_beta};
    Tensor<T> B1_a{o_alpha};

    // if(rank==0) cout << "allocate B" << endl;
    sch.allocate(B1,B1_a).execute();
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
      (B1_a(h1_oa) = B1(h1_oa))
      .deallocate(B1)
      .execute();
    ec.pg().barrier();

    sch.allocate(Hx1_a, Hx2_aaa, Hx2_bab,
                  x1_a,  x2_aaa,  x2_bab).execute();

    double gf_t_guess     = 0.0;
    double gf_t_x1_tot    = 0.0;
    double gf_t_x2_tot    = 0.0;
    double gf_t_res_tot   = 0.0;
    double gf_t_res_tot_1 = 0.0;
    double gf_t_res_tot_2 = 0.0;
    double gf_t_upd_tot   = 0.0;
    double gf_t_dis_tot   = 0.0;
    size_t gf_iter        = 1;

    std::string x1_a_inter_wpi_file = files_prefix+".x1_a.inter.w"+gfo.str()+".oi"+std::to_string(pi);
    std::string x2_aaa_inter_wpi_file = files_prefix+".x2_aaa.inter.w"+gfo.str()+".oi"+std::to_string(pi);
    std::string x2_bab_inter_wpi_file = files_prefix+".x2_bab.inter.w"+gfo.str()+".oi"+std::to_string(pi);

    if(fs::exists(x1_a_inter_wpi_file) && fs::exists(x2_aaa_inter_wpi_file) && fs::exists(x2_bab_inter_wpi_file)) {
      read_from_disk(x1_a,   x1_a_inter_wpi_file);
      read_from_disk(x2_aaa, x2_aaa_inter_wpi_file);
      read_from_disk(x2_bab, x2_bab_inter_wpi_file);
    }
    else {
      sch
        .allocate(x1,Minv)
        (x1()      = 0)
        (Minv()    = 0)        
        .execute();
  
      gf_guess_ip(ec,MO,nocc,gf_omega,gf_eta,pi,p_evl_sorted_occ,t2v2_o,x1,Minv,true);
    
      sch
        (x1_a(h1_oa) = x1(h1_oa))
        .deallocate(x1,Minv)
        .execute();    
    }
    
    // GMRES
    int64_t gmres_hist = ngmres;
      
    ComplexTensor tmp{};
    sch.allocate(tmp).execute();

    do {

      auto gf_gmres_0 = std::chrono::high_resolution_clock::now();

      ComplexTensor r1_a{o_alpha};
      ComplexTensor r2_aaa{v_alpha,o_alpha,o_alpha};
      ComplexTensor r2_bab{v_beta, o_alpha,o_beta};
      VComplexTensor Q1_a;
      VComplexTensor Q2_aaa;
      VComplexTensor Q2_bab;

      gfccsd_x1_a(sch, MO, Hx1_a, 
                  t1_a, t1_b, t2_aaaa, t2_bbbb, t2_abab, 
                  x1_a, x2_aaa, x2_bab, 
                  f1, ix2_2_a, ix1_1_1_a, ix1_1_1_b,
                  ix2_6_3_aaaa, ix2_6_3_abab,
                  unit_tis,false);
          
      gfccsd_x2_a(sch, MO, Hx2_aaa, Hx2_bab, 
                  t1_a, t1_b, t2_aaaa, t2_bbbb, t2_abab, 
                  x1_a, x2_aaa, x2_bab, 
                  f1, ix2_1_aaaa, ix2_1_abab,
                  ix2_2_a, ix2_2_b,
                  ix2_3_a, ix2_3_b, 
                  ix2_4_aaaa, ix2_4_abab,
                  ix2_5_aaaa, ix2_5_abba, ix2_5_abab, 
                  ix2_5_bbbb, ix2_5_baab,
                  ix2_6_2_a, ix2_6_2_b, 
                  ix2_6_3_aaaa, ix2_6_3_abba, ix2_6_3_abab,
                  ix2_6_3_bbbb, ix2_6_3_baab,
                  v2ijab_aaaa, v2ijab_abab, v2ijab_bbbb,
                  unit_tis,false);
        
        #ifdef USE_TALSH
          sch.execute(ExecutionHW::GPU);
        #else
          sch.execute();
        #endif

      sch 
        .allocate(r1_a,  r2_aaa,  r2_bab)
        (r1_a()   = -1.0 * Hx1_a())
        (r2_aaa() = -1.0 * Hx2_aaa())
        (r2_bab() = -1.0 * Hx2_bab())
        (r1_a(h1_oa)               -= std::complex<double>(gf_omega,-1.0*gf_eta) * x1_a(h1_oa))
        (r2_aaa(p1_va,h1_oa,h2_oa) -= std::complex<double>(gf_omega,-1.0*gf_eta) * x2_aaa(p1_va,h1_oa,h2_oa))
        (r2_bab(p1_vb,h1_oa,h2_ob) -= std::complex<double>(gf_omega,-1.0*gf_eta) * x2_bab(p1_vb,h1_oa,h2_ob))
        (r1_a() += B1_a())
        .execute();

      auto r1_a_norm   = norm(r1_a);
      auto r2_aaa_norm = norm(r2_aaa);
      auto r2_bab_norm = norm(r2_bab);

      auto gf_residual 
             = sqrt(r1_a_norm*r1_a_norm + 0.5*r2_aaa_norm*r2_aaa_norm + r2_bab_norm*r2_bab_norm);
        
      auto gf_gmres = std::chrono::high_resolution_clock::now();
      double gftime =
        std::chrono::duration_cast<std::chrono::duration<double>>((gf_gmres - gf_gmres_0)).count();
      if(root_ppi==0 && debug) {
        cout << "----------------" << endl;
        cout << "  #iter " << gf_iter << ", T(x_update contraction): " << gftime << endl;
        cout << std::defaultfloat << "  w,oi (" << gfo.str() << "," << pi << "), residual = " << gf_residual << std::endl;
      }

      if(std::abs(gf_residual) < gf_threshold || gf_iter > gf_maxiter) {
        sch.deallocate(r1_a, r2_aaa, r2_bab).execute();
        free_vec_tensors(Q1_a, Q2_aaa, Q2_bab);
        Q1_a.clear();
        Q2_aaa.clear();
        Q2_bab.clear();
        if(std::abs(gf_residual) < gf_threshold) gf_conv = true;
        break;
      }
      
      tamm::scale_ip(r1_a,1.0/gf_residual);
      tamm::scale_ip(r2_aaa,1.0/gf_residual);
      tamm::scale_ip(r2_bab,1.0/gf_residual);
      Q1_a.push_back(r1_a);
      Q2_aaa.push_back(r2_aaa);
      Q2_bab.push_back(r2_bab);
      CMatrix cn = CMatrix::Zero(gmres_hist, 1);
      CMatrix sn = CMatrix::Zero(gmres_hist, 1);
      CMatrix H  = CMatrix::Zero(gmres_hist+1, gmres_hist);
      CMatrix b  = CMatrix::Zero(gmres_hist+1, 1);
      b(0, 0)    = gf_residual;

      // if(root_ppi==0) cout << "gf_iter: " << gf_iter << endl;

      // GMRES inner loop
      for(auto k=0; k<gmres_hist; k++) {

        ComplexTensor q1_a{o_alpha};
        ComplexTensor q2_aaa{v_alpha,o_alpha,o_alpha};
        ComplexTensor q2_bab{v_beta, o_alpha,o_beta};

        auto gf_gmres_1 = std::chrono::high_resolution_clock::now();

        gfccsd_x1_a(sch, MO, Hx1_a, 
                    t1_a, t1_b, t2_aaaa, t2_bbbb, t2_abab, 
                    Q1_a[k], Q2_aaa[k], Q2_bab[k], 
                    f1, ix2_2_a, ix1_1_1_a, ix1_1_1_b,
                    ix2_6_3_aaaa, ix2_6_3_abab,
                    unit_tis,false);
          
        gfccsd_x2_a(sch, MO, Hx2_aaa, Hx2_bab, 
                    t1_a, t1_b, t2_aaaa, t2_bbbb, t2_abab, 
                    Q1_a[k], Q2_aaa[k], Q2_bab[k],
                    f1, ix2_1_aaaa, ix2_1_abab,
                    ix2_2_a, ix2_2_b,
                    ix2_3_a, ix2_3_b, 
                    ix2_4_aaaa, ix2_4_abab,
                    ix2_5_aaaa, ix2_5_abba, ix2_5_abab, 
                    ix2_5_bbbb, ix2_5_baab,
                    ix2_6_2_a, ix2_6_2_b, 
                    ix2_6_3_aaaa, ix2_6_3_abba, ix2_6_3_abab,
                    ix2_6_3_bbbb, ix2_6_3_baab,
                    v2ijab_aaaa, v2ijab_abab, v2ijab_bbbb,
                    unit_tis,false);

        #ifdef USE_TALSH
          sch.execute(ExecutionHW::GPU);
        #else
          sch.execute();
        #endif
        
        sch 
          .allocate(q1_a,q2_aaa,q2_bab)
          (q1_a()    = 1.0 * Hx1_a())
          (q2_aaa()  = 1.0 * Hx2_aaa())
          (q2_bab()  = 1.0 * Hx2_bab())
          (q1_a()   += std::complex<double>(gf_omega,-1.0*gf_eta) * Q1_a[k]())
          (q2_aaa() += std::complex<double>(gf_omega,-1.0*gf_eta) * Q2_aaa[k]())
          (q2_bab() += std::complex<double>(gf_omega,-1.0*gf_eta) * Q2_bab[k]())
          .execute();

        auto gf_gmres_2 = std::chrono::high_resolution_clock::now();
        double gftime =
          std::chrono::duration_cast<std::chrono::duration<double>>((gf_gmres_2 - gf_gmres_1)).count();
        if(root_ppi==0 && debug) cout << "    k: " << k << ", T(gfcc contraction): " << gftime << endl;

        // Arnoldi iteration or G-S orthogonalization
        for(auto j=0; j<=k; j++) {
          auto conj_a   = tamm::conj(Q1_a[j]);
          auto conj_aaa = tamm::conj(Q2_aaa[j]);
          auto conj_bab = tamm::conj(Q2_bab[j]);
          sch
            (tmp()  = 1.0 * conj_a(h1_oa) * q1_a(h1_oa))
            (tmp() += 0.5 * conj_aaa(p1_va,h1_oa,h2_oa) * q2_aaa(p1_va,h1_oa,h2_oa))
            (tmp() += 1.0 * conj_bab(p1_vb,h1_oa,h2_ob) * q2_bab(p1_vb,h1_oa,h2_ob))
            (q1_a()   -= tmp() * Q1_a[j]())
            (q2_aaa() -= tmp() * Q2_aaa[j]())
            (q2_bab() -= tmp() * Q2_bab[j]())
            .deallocate(conj_a,conj_aaa,conj_bab) 
            .execute();
            
          H(j,k) = get_scalar(tmp);
        } // j loop

        r1_a_norm   = norm(q1_a);
        r2_aaa_norm = norm(q2_aaa);
        r2_bab_norm = norm(q2_bab);

        H(k+1,k) = sqrt(r1_a_norm*r1_a_norm + 0.5*r2_aaa_norm*r2_aaa_norm + r2_bab_norm*r2_bab_norm);

        if(std::abs(H(k+1,k))<1e-8) {
          gmres_hist = k+1;
          sch.deallocate(q1_a,q2_aaa,q2_bab).execute();
          break;
        }
        std::complex<double> scaling = 1.0/H(k+1,k);

        auto gf_gmres_3 = std::chrono::high_resolution_clock::now();
        gftime =
          std::chrono::duration_cast<std::chrono::duration<double>>((gf_gmres_3 - gf_gmres_2)).count();
        if(root_ppi==0 && debug) cout << "    k: " << k << ", T(Arnoldi): " << gftime << endl;
          
        CMatrix Hsub = H.block(0,0,k+2,k+1);
        CMatrix bsub = b.block(0,0,k+2,1);
 
        //apply givens rotation
        for(auto i=0; i<k; i++){
          auto temp = cn(i,0) * H(i,k) + sn(i,0) * H(i+1,k);
          H(i+1,k) = -sn(i,0) * H(i,k) + cn(i,0) * H(i+1,k);
          H(i,k) = temp;
        }
  
        if(std::abs(H(k,k))<1e-16){
          cn(k,0) = std::complex<double>(0,0);
          sn(k,0) = std::complex<double>(1,0);
        }
        else{
          auto t = sqrt(H(k,k)*H(k,k)+H(k+1,k)*H(k+1,k));
          cn(k,0) = abs(H(k,k))/t;
          sn(k,0) = cn(k,0) * H(k+1,k)/H(k,k);
        }

        H(k,k)   = cn(k,0) * H(k,k) + sn(k,0) * H(k+1,k);
        H(k+1,k) = std::complex<double>(0,0);

        b(k+1,0) = -sn(k,0) * b(k,0);
        b(k,0)   =  cn(k,0) * b(k,0);

        auto gf_gmres_4 = std::chrono::high_resolution_clock::now();
        gftime =
          std::chrono::duration_cast<std::chrono::duration<double>>((gf_gmres_4 - gf_gmres_3)).count();
        if(root_ppi==0 && debug) {
          cout << "    k: " << k << ", T(Givens rotation): " << gftime << ", error: " << std::abs(b(k+1,0)) << endl;
          cout << "    ----------" << endl;
        }// if(root_ppi==0&&debug) cout<< "k: " << k << ", error: " << std::abs(b(k+1,0)) << endl;

        //normalization
        if(std::abs(b(k+1,0))>1e-2) {
          tamm::scale_ip(q1_a,scaling);
          tamm::scale_ip(q2_aaa,scaling);
          tamm::scale_ip(q2_bab,scaling);
          Q1_a.push_back(q1_a);
          Q2_aaa.push_back(q2_aaa);
          Q2_bab.push_back(q2_bab);
        }
        else {
          gmres_hist = k+1;
          sch.deallocate(q1_a,q2_aaa,q2_bab).execute();
          break;
        }
      } // k loop

      auto gf_gmres_5 = std::chrono::high_resolution_clock::now();
      gftime =
        std::chrono::duration_cast<std::chrono::duration<double>>((gf_gmres_5 - gf_gmres)).count();
      if(root_ppi==0 && debug) cout << "  #iter " << gf_iter << ", T(micro_tot): " << gftime << endl;

      //solve a least square problem in the subspace
      CMatrix Hsub = H.block(0,0,gmres_hist,gmres_hist);
      CMatrix bsub = b.block(0,0,gmres_hist,1);
      CMatrix y = Hsub.householderQr().solve(bsub);

      for(auto i = 0; i < gmres_hist; i++) { 
        sch
          (x1_a(h1_oa)               += y(i,0) * Q1_a[i](h1_oa))
          (x2_aaa(p1_va,h1_oa,h2_oa) += y(i,0) * Q2_aaa[i](p1_va,h1_oa,h2_oa))
          (x2_bab(p1_vb,h1_oa,h2_ob) += y(i,0) * Q2_bab[i](p1_vb,h1_oa,h2_ob));
      }
      sch.execute();

      write_to_disk(x1_a,  x1_a_inter_wpi_file);
      write_to_disk(x2_aaa,x2_aaa_inter_wpi_file);
      write_to_disk(x2_bab,x2_bab_inter_wpi_file);       

      free_vec_tensors(Q1_a,Q2_aaa,Q2_bab);
      Q1_a.clear();
      Q2_aaa.clear();
      Q2_bab.clear();

      auto gf_gmres_6 = std::chrono::high_resolution_clock::now();
      gftime =
        std::chrono::duration_cast<std::chrono::duration<double>>((gf_gmres_6 - gf_gmres_5)).count();
      if(root_ppi==0 && debug) cout << "  #iter " << gf_iter << ", T(least_square+X_updat+misc.): " << gftime << endl;

      gf_iter++;
                
    }while(true);

    //deallocate memory
    sch.deallocate(tmp).execute();

    if(gf_conv) {
      std::string x1_a_conv_wpi_file   = files_prefix+".x1_a.w"  +gfo.str()+".oi"+std::to_string(pi);
      std::string x2_aaa_conv_wpi_file = files_prefix+".x2_aaa.w"+gfo.str()+".oi"+std::to_string(pi);
      std::string x2_bab_conv_wpi_file = files_prefix+".x2_bab.w"+gfo.str()+".oi"+std::to_string(pi);
      write_to_disk(x1_a,  x1_a_conv_wpi_file);
      write_to_disk(x2_aaa,x2_aaa_conv_wpi_file);
      write_to_disk(x2_bab,x2_bab_conv_wpi_file);
      fs::remove(x1_a_inter_wpi_file);
      fs::remove(x2_aaa_inter_wpi_file);
      fs::remove(x2_bab_inter_wpi_file);
    }
    
    if(!gf_conv && root_ppi==0) {
      std::string error_string = gfo.str()+","+std::to_string(pi)+".";
      nwx_terminate("ERROR: GF-CCSD does not converge for w,oi = "+error_string);
    }      

    auto gf_t2 = std::chrono::high_resolution_clock::now();
    double gftime =
      std::chrono::duration_cast<std::chrono::duration<double>>((gf_t2 - gf_t1)).count();
    if(root_ppi == 0) {
      std::string gf_stats;
      gf_stats = gfacc_str("R-GF-CCSD Time for w,oi (", gfo.str(), ",", std::to_string(pi), ") = ", 
                 std::to_string(gftime), " secs, #iter = ", std::to_string(gf_iter),
                 ", using PG ", std::to_string(pg_id));
      
      if(debug){
        gf_stats += gfacc_str("|----------initial guess  : ",std::to_string(gf_t_guess  ));
        gf_stats += gfacc_str("|----------x1 contraction : ",std::to_string(gf_t_x1_tot ));
        gf_stats += gfacc_str("|----------x2 contraction : ",std::to_string(gf_t_x2_tot ));
        gf_stats += gfacc_str("|----------computing res. : ",std::to_string(gf_t_res_tot));
        gf_stats += gfacc_str("           |----------misc. contr. : ",std::to_string(gf_t_res_tot_1));
        gf_stats += gfacc_str("           |----------compt. norm  : ",std::to_string(gf_t_res_tot_2));
        gf_stats += gfacc_str("|----------updating x1/x2 : ",std::to_string(gf_t_upd_tot));
        gf_stats += gfacc_str("|----------diis update    : ",std::to_string(gf_t_dis_tot));
      }
      std::cout << std::defaultfloat << gf_stats << std::flush;
    }      

    sch.deallocate(Hx1_a, Hx2_aaa, Hx2_bab,
                    x1_a,  x2_aaa,  x2_bab, B1_a).execute();

    if(root_ppi == 0) next = ac->fetch_add(0, 1);
    MPI_Bcast(&next,1,mpi_type<int64_t>(),0,ec.pg().comm());

   }
    
   if(root_ppi == 0) taskcount++;
   MPI_Bcast(&taskcount        ,1,mpi_type<int64_t>(),0,ec.pg().comm());
  
  } //end all remaining pi 

  auto cc_t2 = std::chrono::high_resolution_clock::now();
  double time =
    std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();

  if(root_ppi==0) {
    cout << "Total orbitals executed by process group " << pg_id << " = " << total_pi_pg << endl;
    cout << "  --> Total R-GF-CCSD Time = " << time << " secs" << endl;
  }

  #if GF_PGROUPS
    ec.flush_and_sync();
    //MemoryManagerGA::destroy_coll(mgr);    
    pg.destroy_coll();
  #endif
  ac->deallocate();
  delete ac;
  gec.pg().barrier();

  cc_t2 = std::chrono::high_resolution_clock::now();
  time =
    std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
  if(rank == 0) {
    std::cout << "Total R-GF-CCSD Time (w = " << gfo.str() << ") = " << time << " secs" << std::endl;
    std::cout << std::string(55, '-') << std::endl;
  }

  gsch.deallocate(dtmp_aaa,dtmp_bab).execute();
  MPI_Comm_free(&gf_comm);
}

////////////////////_Main-///////////////////////////

void gfccsd_main_driver(std::string filename) {

    // std::cout << "Input file provided = " << filename << std::endl;

    using T = double;

    ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
    ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};
    auto rank = ec.pg().rank();

    ProcGroup pg_l = ProcGroup::create_coll(MPI_COMM_SELF);
    ExecutionContext ec_l{pg_l, DistributionKind::nw, MemoryManagerKind::local};

    auto restart_time_start = std::chrono::high_resolution_clock::now();

    auto [sys_data, hf_energy, shells, shell_tile_map, C_AO, F_AO, C_beta_AO, F_beta_AO, AO_opt, AO_tis,scf_conv]  
                    = hartree_fock_driver<T>(ec,filename);

    int nsranks = sys_data.nbf/15;
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
    
    ProcGroup sub_pg;
    ExecutionContext *sub_ec=nullptr;

    if(subcomm != MPI_COMM_NULL){
        sub_pg = ProcGroup::create_coll(subcomm);
        sub_ec = new ExecutionContext(sub_pg, DistributionKind::nw, MemoryManagerKind::ga);
    }

    Scheduler sub_sch{*sub_ec};

    CCSDOptions ccsd_options = sys_data.options_map.ccsd_options;
    debug = ccsd_options.debug;
    if(rank == 0) ccsd_options.print();

    if(rank==0) cout << endl << "#occupied, #virtual = " << sys_data.nocc << ", " << sys_data.nvir << endl;
    
    auto [MO,total_orbitals] = setupMOIS(sys_data);

    std::string out_fp = sys_data.output_file_prefix+"."+ccsd_options.basis;
    std::string files_dir = out_fp+"_files";
    std::string files_prefix = /*out_fp;*/ files_dir+"/"+out_fp;
    std::string f1file = files_prefix+".f1_mo";
    std::string t1file = files_prefix+".t1amp";
    std::string t2file = files_prefix+".t2amp";
    std::string v2file = files_prefix+".cholv2";
    std::string cholfile = files_prefix+".cholcount";
    std::string ccsdstatus = files_prefix+".ccsdstatus";
    // std::string outputfile = out_fp+".gfcc.profile";
    // ofs_profile.open(outputfile, std::ios::out);    

    bool ccsd_restart = ccsd_options.readt || 
        ( (fs::exists(t1file) && fs::exists(t2file)     
        && fs::exists(f1file) && fs::exists(v2file)) );

    //deallocates F_AO, C_AO
    auto [cholVpr,d_f1,chol_count, max_cvecs, CI] = cd_svd_ga_driver<T>
                        (sys_data, ec, MO, AO_opt, C_AO, F_AO, C_beta_AO, F_beta_AO, shells, shell_tile_map,
                                ccsd_restart, cholfile);

    TiledIndexSpace N = MO("all");

    auto [p_evl_sorted,d_t1,d_t2,d_r1,d_r2, d_r1s, d_r2s, d_t1s, d_t2s] 
            = setupTensors(ec,MO,d_f1,ccsd_options.ndiis,ccsd_restart && fs::exists(ccsdstatus) && scf_conv);

    if(ccsd_restart) {
        read_from_disk(d_f1,f1file);
        if(fs::exists(t1file) && fs::exists(t2file)) {
          read_from_disk(d_t1,t1file);
          read_from_disk(d_t2,t2file);
        }
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
      for (size_t i=0;i<p_evl_sorted.size();i++) cout << i+1 << "   " << p_evl_sorted[i] << endl;
      cout << std::string(50,'-') << endl;
    }

    ec.pg().barrier();

    auto cc_t1 = std::chrono::high_resolution_clock::now();

    #ifdef USE_TALSH
    const bool has_gpu = ec.has_gpu();
    TALSH talsh_instance;
    if(has_gpu) talsh_instance.initialize(ec.gpu_devid(),rank.value());
    #endif

    ccsd_restart = ccsd_restart && fs::exists(ccsdstatus) && scf_conv;

    double residual=0, corr_energy=0;
    if(ccsd_restart){
      
      if(subcomm != MPI_COMM_NULL){
          std::tie(residual, corr_energy) = cd_ccsd_driver<T>(
            sys_data, *sub_ec, MO, CI, d_t1, d_t2, d_f1, 
            d_r1,d_r2, d_r1s, d_r2s, d_t1s, d_t2s, 
            p_evl_sorted, 
            cholVpr, ccsd_restart, files_prefix);
      }
      ec.pg().barrier();
    }
    else{
      std::tie(residual, corr_energy) = cd_ccsd_driver<T>(
            sys_data, ec, MO, CI, d_t1, d_t2, d_f1, 
            d_r1,d_r2, d_r1s, d_r2s, d_t1s, d_t2s, 
            p_evl_sorted, 
            cholVpr, ccsd_restart, files_prefix);
    }

    ccsd_stats(ec, hf_energy,residual,corr_energy,ccsd_options.threshold);

    if(ccsd_options.writet && !fs::exists(ccsdstatus)) {
        // write_to_disk(d_t1,t1file);
        // write_to_disk(d_t2,t2file);
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

  //////////////////////////
  //                      //
  // Start GFCCSD Routine //
  //                      //
  //////////////////////////
  cc_t1 = std::chrono::high_resolution_clock::now();

  const TAMM_SIZE nocc = sys_data.nocc;
  const TAMM_SIZE nvir = sys_data.nvir;
  const TAMM_SIZE noa  = sys_data.n_occ_alpha;
  // const TAMM_SIZE nob  = sys_data.n_occ_beta;
  // const TAMM_SIZE nva  = sys_data.n_vir_alpha;
  // const TAMM_SIZE nvb  = sys_data.n_vir_beta;
  
  ndiis                = ccsd_options.gf_ndiis;
  ngmres               = ccsd_options.gf_ngmres;
  gf_eta               = ccsd_options.gf_eta;
  gf_maxiter           = ccsd_options.gf_maxiter;
  gf_threshold         = ccsd_options.gf_threshold;
  omega_min_ip         = ccsd_options.gf_omega_min_ip;
  omega_max_ip         = ccsd_options.gf_omega_max_ip;
  lomega_min_ip        = ccsd_options.gf_omega_min_ip_e;
  lomega_max_ip        = ccsd_options.gf_omega_max_ip_e;
  omega_delta          = ccsd_options.gf_omega_delta;
  omega_delta_e        = ccsd_options.gf_omega_delta_e;
  gf_nprocs_poi        = ccsd_options.gf_nprocs_poi;  
  gf_level_shift       = 0;
  gf_damping_factor    = ccsd_options.gf_damping_factor;
  gf_extrapolate_level = ccsd_options.gf_extrapolate_level;
  gf_analyze_level = ccsd_options.gf_analyze_level;
  gf_analyze_num_omega = ccsd_options.gf_analyze_num_omega;
  omega_npts_ip        = (omega_max_ip - omega_min_ip) / omega_delta + 1;
  lomega_npts_ip       = (lomega_max_ip - lomega_min_ip) / omega_delta_e + 1;

  const int gf_p_oi = ccsd_options.gf_p_oi_range;

  if(gf_p_oi == 1) p_oi = nocc;
  else p_oi = nocc+nvir;

  int level = 1;
  size_t prev_qr_rank = 0;

  if(rank == 0) ccsd_options.print();
  
  using ComplexTensor = Tensor<std::complex<T>>;

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

  auto [p1,p2,p3,p4,p5,p6,p7,p8,p9,p10] = MO.labels<10>("virt");
  auto [h1,h2,h3,h4,h5,h6,h7,h8,h9,h10] = MO.labels<10>("occ");

  auto [cind] = CI.labels<1>("all");
  auto [h1_oa,h2_oa,h3_oa,h4_oa,h5_oa,h6_oa,h7_oa,h8_oa,h9_oa,h10_oa] = o_alpha.labels<10>("all");
  auto [h1_ob,h2_ob,h3_ob,h4_ob,h5_ob,h6_ob,h7_ob,h8_ob,h9_ob,h10_ob] = o_beta.labels<10>("all");
  auto [p1_va,p2_va,p3_va,p4_va,p5_va,p6_va,p7_va,p8_va,p9_va,p10_va] = v_alpha.labels<10>("all");
  auto [p1_vb,p2_vb,p3_vb,p4_vb,p5_vb,p6_vb,p7_vb,p8_vb,p9_vb,p10_vb] = v_beta.labels<10>("all");
    
  Scheduler sch{ec};

  if(rank==0) cout << endl << "#occupied, #virtual = " << nocc << ", " << nvir << endl;
  std::vector<T> p_evl_sorted_occ(nocc);
  std::vector<T> p_evl_sorted_virt(nvir);
  std::copy(p_evl_sorted.begin(), p_evl_sorted.begin() + nocc,
            p_evl_sorted_occ.begin());
  std::copy(p_evl_sorted.begin() + nocc, p_evl_sorted.end(),
            p_evl_sorted_virt.begin());

  //START SF2

  std::vector<T> omega_space_ip;

  if(ccsd_options.gf_ip) {
    for(int64_t ni=0;ni<omega_npts_ip;ni++) {
      T omega_tmp =  omega_min_ip + ni*omega_delta;
      omega_space_ip.push_back(omega_tmp);
    }
    if (rank==0) cout << "Freq. space (before doing MOR): " << omega_space_ip << endl;
  }
  
    auto restart_time_end = std::chrono::high_resolution_clock::now();
    double total_restart_time = 
        std::chrono::duration_cast<std::chrono::duration<double>>((restart_time_end - restart_time_start)).count();
    if(rank == 0) std::cout << std::endl << "GFCC: Time taken pre-restart: " << total_restart_time << " secs" << std::endl;

    using Complex2DMatrix=Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    auto inter_read_start = std::chrono::high_resolution_clock::now();

    size_t nptsi = 0;
    
    std::vector<bool> omega_ip_conv_a(omega_npts_ip,false);
    std::vector<bool> omega_ip_conv_b(omega_npts_ip,false);
    std::vector<T> omega_ip_A0(omega_npts_ip,UINT_MAX);

    std::vector<T> omega_extra;
    std::vector<T> omega_extra_finished;

    ///////////////////////////
    //                       //
    // Compute intermediates //
    //                       //
    ///////////////////////////
    
    if(!fs::exists(files_dir)) {
      fs::create_directories(files_dir);
    }

    Tensor<T> d_t1_a       {v_alpha,o_alpha};
    Tensor<T> d_t1_b       {v_beta, o_beta};
    Tensor<T> d_t2_aaaa    {v_alpha,v_alpha,o_alpha,o_alpha};
    Tensor<T> d_t2_bbbb    {v_beta, v_beta, o_beta, o_beta};
    Tensor<T> d_t2_abab    {v_alpha,v_beta, o_alpha,o_beta};
    Tensor<T> cholOO_a     {o_alpha,o_alpha,CI};
    Tensor<T> cholOO_b     {o_beta, o_beta, CI};
    Tensor<T> cholOV_a     {o_alpha,v_alpha,CI};
    Tensor<T> cholOV_b     {o_beta, v_beta, CI};
    Tensor<T> cholVV_a     {v_alpha,v_alpha,CI};
    Tensor<T> cholVV_b     {v_beta, v_beta, CI};
    Tensor<T> v2ijab_aaaa  {o_alpha,o_alpha,v_alpha,v_alpha};
    Tensor<T> v2ijab_bbbb  {o_beta, o_beta, v_beta, v_beta};
    Tensor<T> v2ijab_abab  {o_alpha,o_beta, v_alpha,v_beta};

    Tensor<T> v2ijab       {{O,O,V,V},{2,2}};
    Tensor<T> v2ijka       {{O,O,O,V},{2,2}};
    Tensor<T> v2ijkl       {{O,O,O,O},{2,2}};
    Tensor<T> v2iajb       {{O,V,O,V},{2,2}};    
    Tensor<T> v2iabc       {{O,V,V,V},{2,2}};

    std::string d_t1_a_file       = files_prefix+".d_t1_a";
    std::string d_t1_b_file       = files_prefix+".d_t1_b";
    std::string d_t2_aaaa_file    = files_prefix+".d_t2_aaaa";
    std::string d_t2_bbbb_file    = files_prefix+".d_t2_bbbb";
    std::string d_t2_abab_file    = files_prefix+".d_t2_abab";
    std::string cholOO_a_file     = files_prefix+".cholOO_a";
    std::string cholOO_b_file     = files_prefix+".cholOO_b";
    std::string cholOV_a_file     = files_prefix+".cholOV_a";
    std::string cholOV_b_file     = files_prefix+".cholOV_b";
    std::string cholVV_a_file     = files_prefix+".cholVV_a";
    std::string cholVV_b_file     = files_prefix+".cholVV_b";
    std::string v2ijab_aaaa_file  = files_prefix+".v2ijab_aaaa";
    std::string v2ijab_bbbb_file  = files_prefix+".v2ijab_bbbb";
    std::string v2ijab_abab_file  = files_prefix+".v2ijab_abab";  

    sch.allocate(d_t1_a, d_t1_b, 
                 d_t2_aaaa, d_t2_bbbb, d_t2_abab,
                 cholOO_a, cholOO_b,
                 cholOV_a, cholOV_b,
                 cholVV_a, cholVV_b,
               v2ijab_aaaa, v2ijab_bbbb, v2ijab_abab,
               v2ijab, v2ijka, v2iajb).execute();

     sch( v2ijka(h1,h2,h3,p1)      =   1.0 * cholVpr(h1,h3,cind) * cholVpr(h2,p1,cind) )
        ( v2ijka(h1,h2,h3,p1)     +=  -1.0 * cholVpr(h2,h3,cind) * cholVpr(h1,p1,cind) )  
        ( v2iajb(h1,p1,h2,p2)      =   1.0 * cholVpr(h1,h2,cind) * cholVpr(p1,p2,cind) )
        ( v2iajb(h1,p1,h2,p2)     +=  -1.0 * cholVpr(h1,p2,cind) * cholVpr(h2,p1,cind) )
        ( v2ijab(h1,h2,p1,p2)      =   1.0 * cholVpr(h1,p1,cind) * cholVpr(h2,p2,cind) )
        ( v2ijab(h1,h2,p1,p2)     +=  -1.0 * cholVpr(h1,p2,cind) * cholVpr(h2,p1,cind) );

        #ifdef USE_TALSH
          sch.execute(ExecutionHW::GPU);
        #else
          sch.execute();
        #endif        

    if(fs::exists(d_t1_a_file)      && fs::exists(d_t1_b_file)      &&
       fs::exists(d_t2_aaaa_file)   && fs::exists(d_t2_bbbb_file)   && fs::exists(d_t2_abab_file)   &&
       fs::exists(cholOO_a_file)    && fs::exists(cholOO_b_file)    && 
       fs::exists(cholOV_a_file)    && fs::exists(cholOV_b_file)    && 
       fs::exists(cholVV_a_file)    && fs::exists(cholVV_b_file)    && 
       fs::exists(v2ijab_aaaa_file) && fs::exists(v2ijab_bbbb_file) && fs::exists(v2ijab_abab_file) &&
       ccsd_options.gf_restart) {
       read_from_disk(d_t1_a,d_t1_a_file);
       read_from_disk(d_t1_b,d_t1_b_file);
       read_from_disk(d_t2_aaaa,d_t2_aaaa_file);
       read_from_disk(d_t2_bbbb,d_t2_bbbb_file);
       read_from_disk(d_t2_abab,d_t2_abab_file);
       read_from_disk(cholOO_a,cholOO_a_file);
       read_from_disk(cholOO_b,cholOO_b_file);
       read_from_disk(cholOV_a,cholOV_a_file);
       read_from_disk(cholOV_b,cholOV_b_file);
       read_from_disk(cholVV_a,cholVV_a_file);
       read_from_disk(cholVV_b,cholVV_b_file);
       read_from_disk(v2ijab_aaaa,v2ijab_aaaa_file);
       read_from_disk(v2ijab_bbbb,v2ijab_bbbb_file);
       read_from_disk(v2ijab_abab,v2ijab_abab_file);
    }
    else {
      #if GF_IN_SG
      if(subcomm != MPI_COMM_NULL) {
        sub_sch
      #else 
        sch
      #endif
          // ( d_t2() = 0 ) // CCS
          ( d_t1_a(p1_va,h3_oa)  =  1.0 * d_t1(p1_va,h3_oa)                            )
          ( d_t1_b(p1_vb,h3_ob)  =  1.0 * d_t1(p1_vb,h3_ob)                            )
          ( d_t2_aaaa(p1_va,p2_va,h3_oa,h4_oa)  =  1.0 * d_t2(p1_va,p2_va,h3_oa,h4_oa) )
          ( d_t2_abab(p1_va,p2_vb,h3_oa,h4_ob)  =  1.0 * d_t2(p1_va,p2_vb,h3_oa,h4_ob) )
          ( d_t2_bbbb(p1_vb,p2_vb,h3_ob,h4_ob)  =  1.0 * d_t2(p1_vb,p2_vb,h3_ob,h4_ob) )
          ( cholOO_a(h1_oa,h2_oa,cind)  =  1.0 * cholVpr(h1_oa,h2_oa,cind)             )
          ( cholOO_b(h1_ob,h2_ob,cind)  =  1.0 * cholVpr(h1_ob,h2_ob,cind)             )
          ( cholOV_a(h1_oa,p1_va,cind)  =  1.0 * cholVpr(h1_oa,p1_va,cind)             )
          ( cholOV_b(h1_ob,p1_vb,cind)  =  1.0 * cholVpr(h1_ob,p1_vb,cind)             )
          ( cholVV_a(p1_va,p2_va,cind)  =  1.0 * cholVpr(p1_va,p2_va,cind)             )
          ( cholVV_b(p1_vb,p2_vb,cind)  =  1.0 * cholVpr(p1_vb,p2_vb,cind)             )
          ( v2ijab_aaaa(h1_oa,h2_oa,p1_va,p2_va)  =  1.0 * v2ijab(h1_oa,h2_oa,p1_va,p2_va) )
          ( v2ijab_abab(h1_oa,h2_ob,p1_va,p2_vb)  =  1.0 * v2ijab(h1_oa,h2_ob,p1_va,p2_vb) )
          ( v2ijab_bbbb(h1_ob,h2_ob,p1_vb,p2_vb)  =  1.0 * v2ijab(h1_ob,h2_ob,p1_vb,p2_vb) )
          .execute();
      #if GF_IN_SG
      }
      #endif
      write_to_disk(d_t1_a,d_t1_a_file);
      write_to_disk(d_t1_b,d_t1_b_file);
      write_to_disk(d_t2_aaaa,d_t2_aaaa_file);
      write_to_disk(d_t2_bbbb,d_t2_bbbb_file);
      write_to_disk(d_t2_abab,d_t2_abab_file);
      write_to_disk(cholOO_a,cholOO_a_file);
      write_to_disk(cholOO_b,cholOO_b_file);
      write_to_disk(cholOV_a,cholOV_a_file);
      write_to_disk(cholOV_b,cholOV_b_file);
      write_to_disk(cholVV_a,cholVV_a_file);
      write_to_disk(cholVV_b,cholVV_b_file);
      write_to_disk(v2ijab_aaaa,v2ijab_aaaa_file);
      write_to_disk(v2ijab_bbbb,v2ijab_bbbb_file);
      write_to_disk(v2ijab_abab,v2ijab_abab_file);
    }

    ec.pg().barrier();
     
    if(ccsd_options.gf_ip) {
      std::string t2v2_o_file       = files_prefix+".t2v2_o";      
      std::string lt12_o_a_file     = files_prefix+".lt12_o_a";
      std::string lt12_o_b_file     = files_prefix+".lt12_o_b";
      std::string ix1_1_1_a_file    = files_prefix+".ix1_1_1_a";   //
      std::string ix1_1_1_b_file    = files_prefix+".ix1_1_1_b";   //
      std::string ix2_1_aaaa_file   = files_prefix+".ix2_1_aaaa";  //
      std::string ix2_1_abab_file   = files_prefix+".ix2_1_abab";  //
      std::string ix2_1_bbbb_file   = files_prefix+".ix2_1_bbbb";  //
      std::string ix2_1_baba_file   = files_prefix+".ix2_1_baba";  //
      std::string ix2_2_a_file      = files_prefix+".ix2_2_a";     //
      std::string ix2_2_b_file      = files_prefix+".ix2_2_b";     //
      std::string ix2_3_a_file      = files_prefix+".ix2_3_a";     //
      std::string ix2_3_b_file      = files_prefix+".ix2_3_b";     //
      std::string ix2_4_aaaa_file   = files_prefix+".ix2_4_aaaa";  //
      std::string ix2_4_abab_file   = files_prefix+".ix2_4_abab";  //
      std::string ix2_4_bbbb_file   = files_prefix+".ix2_4_bbbb";  //
      std::string ix2_5_aaaa_file   = files_prefix+".ix2_5_aaaa";  //
      std::string ix2_5_abba_file   = files_prefix+".ix2_5_abba";  //
      std::string ix2_5_abab_file   = files_prefix+".ix2_5_abab";  //
      std::string ix2_5_bbbb_file   = files_prefix+".ix2_5_bbbb";  //
      std::string ix2_5_baab_file   = files_prefix+".ix2_5_baab";  //
      std::string ix2_5_baba_file   = files_prefix+".ix2_5_baba";  //
      std::string ix2_6_2_a_file    = files_prefix+".ix2_6_2_a";   //
      std::string ix2_6_2_b_file    = files_prefix+".ix2_6_2_b";   //
      std::string ix2_6_3_aaaa_file = files_prefix+".ix2_6_3_aaaa";//
      std::string ix2_6_3_abba_file = files_prefix+".ix2_6_3_abba";//
      std::string ix2_6_3_abab_file = files_prefix+".ix2_6_3_abab";//
      std::string ix2_6_3_baab_file = files_prefix+".ix2_6_3_baab";//
      std::string ix2_6_3_bbbb_file = files_prefix+".ix2_6_3_bbbb";//
      std::string ix2_6_3_baba_file = files_prefix+".ix2_6_3_baba";//

      t2v2_o       = Tensor<T>{{O,O},{1,1}};
      lt12_o_a     = Tensor<T>{o_alpha,o_alpha};
      lt12_o_b     = Tensor<T>{o_beta, o_beta};
      ix1_1_1_a    = Tensor<T>{o_alpha,v_alpha};
      ix1_1_1_b    = Tensor<T>{o_beta, v_beta};
      ix2_1_aaaa   = Tensor<T>{o_alpha,v_alpha,o_alpha,o_alpha};
      ix2_1_bbbb   = Tensor<T>{o_beta, v_beta, o_beta, o_beta};
      ix2_1_abab   = Tensor<T>{o_alpha,v_beta, o_alpha,o_beta};
      ix2_1_baba   = Tensor<T>{o_beta, v_alpha,o_beta, o_alpha};
      ix2_2_a      = Tensor<T>{o_alpha,o_alpha};
      ix2_2_b      = Tensor<T>{o_beta, o_beta};
      ix2_3_a      = Tensor<T>{v_alpha,v_alpha};
      ix2_3_b      = Tensor<T>{v_beta, v_beta};
      ix2_4_aaaa   = Tensor<T>{o_alpha,o_alpha,o_alpha,o_alpha};
      ix2_4_abab   = Tensor<T>{o_alpha,o_beta, o_alpha,o_beta};
      ix2_4_bbbb   = Tensor<T>{o_beta, o_beta, o_beta, o_beta};
      ix2_5_aaaa   = Tensor<T>{o_alpha,v_alpha,o_alpha,v_alpha};
      ix2_5_abba   = Tensor<T>{o_alpha,v_beta, o_beta, v_alpha};
      ix2_5_abab   = Tensor<T>{o_alpha,v_beta, o_alpha,v_beta};
      ix2_5_bbbb   = Tensor<T>{o_beta, v_beta, o_beta, v_beta};
      ix2_5_baab   = Tensor<T>{o_beta, v_alpha,o_alpha,v_beta};
      ix2_5_baba   = Tensor<T>{o_beta, v_alpha,o_beta, v_alpha};
      ix2_6_2_a    = Tensor<T>{o_alpha,v_alpha};
      ix2_6_2_b    = Tensor<T>{o_beta, v_beta};
      ix2_6_3_aaaa = Tensor<T>{o_alpha,o_alpha,o_alpha,v_alpha};
      ix2_6_3_abba = Tensor<T>{o_alpha,o_beta, o_beta, v_alpha};
      ix2_6_3_abab = Tensor<T>{o_alpha,o_beta, o_alpha,v_beta};
      ix2_6_3_baab = Tensor<T>{o_beta, o_alpha,o_alpha,v_beta};
      ix2_6_3_bbbb = Tensor<T>{o_beta, o_beta, o_beta, v_beta};
      ix2_6_3_baba = Tensor<T>{o_beta, o_alpha,o_beta, v_alpha};

      Tensor<T> lt12_o       {{O, O},{1,1}};
      Tensor<T> ix1_1_1      {{O, V},{1,1}};
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

      sch.allocate(t2v2_o,
                   lt12_o_a, lt12_o_b,
                   ix1_1_1_a, ix1_1_1_b,
                   ix2_1_aaaa, ix2_1_abab, ix2_1_bbbb, ix2_1_baba,
                   ix2_2_a, ix2_2_b, 
                   ix2_3_a, ix2_3_b, 
                   ix2_4_aaaa, ix2_4_abab, ix2_4_bbbb, 
                   ix2_5_aaaa, ix2_5_abba, ix2_5_abab, 
                   ix2_5_bbbb, ix2_5_baab, ix2_5_baba,
                   ix2_6_2_a, ix2_6_2_b, 
                   ix2_6_3_aaaa, ix2_6_3_abba, ix2_6_3_abab,
                   ix2_6_3_bbbb, ix2_6_3_baab, ix2_6_3_baba).execute();

      if(fs::exists(t2v2_o_file)       &&
         fs::exists(lt12_o_a_file)     && fs::exists(lt12_o_b_file)     &&
         fs::exists(ix1_1_1_a_file)    && fs::exists(ix1_1_1_b_file)    && 
         fs::exists(ix2_1_aaaa_file)   && fs::exists(ix2_1_bbbb_file)   && 
         fs::exists(ix2_1_abab_file)   && fs::exists(ix2_1_baba_file)   &&
         fs::exists(ix2_2_a_file)      && fs::exists(ix2_2_b_file)      && 
         fs::exists(ix2_3_a_file)      && fs::exists(ix2_3_b_file)      &&
         fs::exists(ix2_4_aaaa_file)   && fs::exists(ix2_4_abab_file)   && fs::exists(ix2_4_bbbb_file)   && 
         fs::exists(ix2_5_aaaa_file)   && fs::exists(ix2_5_abba_file)   && fs::exists(ix2_5_abab_file)   && 
         fs::exists(ix2_5_bbbb_file)   && fs::exists(ix2_5_baab_file)   && fs::exists(ix2_5_baba_file)   && 
         fs::exists(ix2_6_2_a_file)    && fs::exists(ix2_6_2_b_file)    &&
         fs::exists(ix2_6_3_aaaa_file) && fs::exists(ix2_6_3_abba_file) && fs::exists(ix2_6_3_abab_file) &&
         fs::exists(ix2_6_3_bbbb_file) && fs::exists(ix2_6_3_baab_file) && fs::exists(ix2_6_3_baba_file) &&
         ccsd_options.gf_restart) {
        read_from_disk(t2v2_o,t2v2_o_file);
        read_from_disk(lt12_o_a,lt12_o_a_file);
        read_from_disk(lt12_o_b,lt12_o_b_file);
        read_from_disk(ix1_1_1_a,ix1_1_1_a_file);
        read_from_disk(ix1_1_1_b,ix1_1_1_b_file);
        read_from_disk(ix2_1_aaaa,ix2_1_aaaa_file);
        read_from_disk(ix2_1_bbbb,ix2_1_bbbb_file);
        read_from_disk(ix2_1_abab,ix2_1_abab_file);
        read_from_disk(ix2_1_baba,ix2_1_baba_file);
        read_from_disk(ix2_2_a,ix2_2_a_file);
        read_from_disk(ix2_2_b,ix2_2_b_file);
        read_from_disk(ix2_3_a,ix2_3_a_file);
        read_from_disk(ix2_3_b,ix2_3_b_file);
        read_from_disk(ix2_4_aaaa,ix2_4_aaaa_file);
        read_from_disk(ix2_4_abab,ix2_4_abab_file);
        read_from_disk(ix2_4_bbbb,ix2_4_bbbb_file);
        read_from_disk(ix2_5_aaaa,ix2_5_aaaa_file);
        read_from_disk(ix2_5_abba,ix2_5_abba_file);
        read_from_disk(ix2_5_abab,ix2_5_abab_file);
        read_from_disk(ix2_5_bbbb,ix2_5_bbbb_file);
        read_from_disk(ix2_5_baab,ix2_5_baab_file);
        read_from_disk(ix2_5_baba,ix2_5_baba_file);
        read_from_disk(ix2_6_2_a,ix2_6_2_a_file);
        read_from_disk(ix2_6_2_b,ix2_6_2_b_file);
        read_from_disk(ix2_6_3_aaaa,ix2_6_3_aaaa_file);
        read_from_disk(ix2_6_3_abba,ix2_6_3_abba_file);
        read_from_disk(ix2_6_3_abab,ix2_6_3_abab_file);
        read_from_disk(ix2_6_3_bbbb,ix2_6_3_bbbb_file);
        read_from_disk(ix2_6_3_baab,ix2_6_3_baab_file);
        read_from_disk(ix2_6_3_baba,ix2_6_3_baba_file);
      }
      else {
        #if GF_IN_SG
        if(subcomm != MPI_COMM_NULL) {
          sub_sch
        #else 
          sch
        #endif
            .allocate(lt12_o,
                      ix1_1_1,
                      ix2_1_1,ix2_1_3,ix2_1_temp,ix2_1,
                      ix2_2,
                      ix2_3,
                      ix2_4_1,ix2_4_temp,ix2_4,
                      ix2_5,
                      ix2_6_2,ix2_6_3,
                      v2ijkl,v2iabc)
            ( v2ijkl(h1,h2,h3,h4)      =   1.0 * cholVpr(h1,h3,cind) * cholVpr(h2,h4,cind) )
            ( v2ijkl(h1,h2,h3,h4)     +=  -1.0 * cholVpr(h1,h4,cind) * cholVpr(h2,h3,cind) )
            ( v2iabc(h1,p1,p2,p3)      =   1.0 * cholVpr(h1,p2,cind) * cholVpr(p1,p3,cind) )
            ( v2iabc(h1,p1,p2,p3)     +=  -1.0 * cholVpr(h1,p3,cind) * cholVpr(p1,p2,cind) )
            ( t2v2_o(h1,h2)            =   0.5 * d_t2(p1,p2,h1,h3) * v2ijab(h3,h2,p1,p2)   )
            ( lt12_o(h1_oa,h3_oa)      =   0.5 * d_t2(p1_va,p2_va,h1_oa,h2_oa) * d_t2(p1_va,p2_va,h3_oa,h2_oa) )
            ( lt12_o(h1_oa,h3_oa)     +=   1.0 * d_t2(p1_va,p2_vb,h1_oa,h2_ob) * d_t2(p1_va,p2_vb,h3_oa,h2_ob) )
            ( lt12_o(h1_oa,h2_oa)     +=   1.0 * d_t1(p1_va,h1_oa) * d_t1(p1_va,h2_oa)                         )
            ( lt12_o(h1_ob,h3_ob)      =   0.5 * d_t2(p1_vb,p2_vb,h1_ob,h2_ob) * d_t2(p1_vb,p2_vb,h3_ob,h2_ob) )
            ( lt12_o(h1_ob,h3_ob)     +=   1.0 * d_t2(p2_va,p1_vb,h2_oa,h1_ob) * d_t2(p2_va,p1_vb,h2_oa,h3_ob) )
            ( lt12_o(h1_ob,h2_ob)     +=   1.0 * d_t1(p1_vb,h1_ob) * d_t1(p1_vb,h2_ob)                         )
            (   ix1_1_1(h6,p7)         =   1.0 * d_f1(h6,p7)                               )
            (   ix1_1_1(h6,p7)        +=   1.0 * d_t1(p4,h5) * v2ijab(h5,h6,p4,p7)         )  
            ( ix2_1(h9,p3,h1,h2)       =   1.0 * v2ijka(h1,h2,h9,p3)                       )  
            (   ix2_1_1(h9,p3,h1,p5)   =   1.0 * v2iajb(h9,p3,h1,p5)                       )
            ( ix2_5(h7,p3,h1,p8)       =   1.0 * d_t1(p5,h1) * v2iabc(h7,p3,p5,p8)         ) //O2V3
            (   ix2_1_1(h9,p3,h1,p5)  +=  -0.5 * ix2_5(h9,p3,h1,p5)                        ) //O2V3
            ( ix2_5(h7,p3,h1,p8)      +=   1.0 * v2iajb(h7,p3,h1,p8)                       )  
            ( ix2_1_temp(h9,p3,h1,h2)  =   1.0 * d_t1(p5,h1) * ix2_1_1(h9,p3,h2,p5)        ) //O3V2
            ( ix2_1(h9,p3,h1,h2)      +=  -1.0 * ix2_1_temp(h9,p3,h1,h2)                   )
            ( ix2_1(h9,p3,h2,h1)      +=   1.0 * ix2_1_temp(h9,p3,h1,h2)                   ) 
            ( ix2_1(h9,p3,h1,h2)      +=  -1.0 * d_t2(p3,p8,h1,h2) * ix1_1_1(h9,p8)        ) //O3V2
            (   ix2_1_3(h6,h9,h1,p5)   =   1.0 * v2ijka(h6,h9,h1,p5)                       )
            (   ix2_1_3(h6,h9,h1,p5)  +=  -1.0 * d_t1(p7,h1) * v2ijab(h6,h9,p5,p7)         ) //O3V2
            ( ix2_1_temp(h9,p3,h1,h2)  =   1.0 * d_t2(p3,p5,h1,h6) * ix2_1_3(h6,h9,h2,p5)  ) //O4V2
            ( ix2_1(h9,p3,h1,h2)      +=   1.0 * ix2_1_temp(h9,p3,h1,h2)                   )
            ( ix2_1(h9,p3,h2,h1)      +=  -1.0 * ix2_1_temp(h9,p3,h1,h2)                   ) 
            ( ix2_1(h9,p3,h1,h2)      +=   0.5 * d_t2(p5,p6,h1,h2) * v2iabc(h9,p3,p5,p6)   ) //O3V3  
            ( ix2_2(h8,h1)             =   1.0 * d_f1(h8,h1)                               )
            ( ix2_2(h8,h1)            +=   1.0 * d_t1(p9,h1) * ix1_1_1(h8,p9)              )
            ( ix2_2(h8,h1)            +=  -1.0 * d_t1(p5,h6) * v2ijka(h6,h8,h1,p5)         )
            ( ix2_2(h8,h1)            +=  -0.5 * d_t2(p5,p6,h1,h7) * v2ijab(h7,h8,p5,p6)   ) //O3V2  
            ( ix2_3(p3,p8)             =   1.0 * d_f1(p3,p8)                               )
            ( ix2_3(p3,p8)            +=   1.0 * d_t1(p5,h6) * v2iabc(h6,p3,p5,p8)         ) 
            ( ix2_3(p3,p8)            +=   0.5 * d_t2(p3,p5,h6,h7) * v2ijab(h6,h7,p5,p8)   ) //O2V3 
            ( ix2_4(h9,h10,h1,h2)      =   1.0 * v2ijkl(h9,h10,h1,h2)                      )
            (   ix2_4_1(h9,h10,h1,p5)  =   1.0 * v2ijka(h9,h10,h1,p5)                      )
            (   ix2_4_1(h9,h10,h1,p5) +=  -0.5 * d_t1(p6,h1) * v2ijab(h9,h10,p5,p6)        ) //O3V2
            ( ix2_4_temp(h9,h10,h1,h2) =   1.0 * d_t1(p5,h1) * ix2_4_1(h9,h10,h2,p5)       ) //O4V
            ( ix2_4(h9,h10,h1,h2)     +=  -1.0 * ix2_4_temp(h9,h10,h1,h2)                  )
            ( ix2_4(h9,h10,h2,h1)     +=   1.0 * ix2_4_temp(h9,h10,h1,h2)                  ) 
            ( ix2_4(h9,h10,h1,h2)     +=   0.5 * d_t2(p5,p6,h1,h2) * v2ijab(h9,h10,p5,p6)  ) //O4V2
            ( ix2_6_2(h10,p5)          =   1.0 * d_f1(h10,p5)                              )
            ( ix2_6_2(h10,p5)         +=   1.0 * d_t1(p6,h7) * v2ijab(h7,h10,p5,p6)        )
            ( ix2_6_3(h8,h10,h1,p9)    =   1.0 * v2ijka(h8,h10,h1,p9)                      )
            ( ix2_6_3(h8,h10,h1,p9)   +=   1.0 * d_t1(p5,h1) * v2ijab(h8,h10,p5,p9)        ) //O3V2
            // IP spin explicit  
            ( lt12_o_a(h1_oa,h2_oa)                  =  1.0 * lt12_o(h1_oa,h2_oa)              )
            ( lt12_o_b(h1_ob,h2_ob)                  =  1.0 * lt12_o(h1_ob,h2_ob)              )
            ( ix1_1_1_a(h1_oa,p1_va)                 =  1.0 * ix1_1_1(h1_oa,p1_va)             )
            ( ix1_1_1_b(h1_ob,p1_vb)                 =  1.0 * ix1_1_1(h1_ob,p1_vb)             )
            ( ix2_1_aaaa(h1_oa,p1_va,h2_oa,h3_oa)    =  1.0 * ix2_1(h1_oa,p1_va,h2_oa,h3_oa)   )
            ( ix2_1_abab(h1_oa,p1_vb,h2_oa,h3_ob)    =  1.0 * ix2_1(h1_oa,p1_vb,h2_oa,h3_ob)   )
            ( ix2_1_baba(h1_ob,p1_va,h2_ob,h3_oa)    =  1.0 * ix2_1(h1_ob,p1_va,h2_ob,h3_oa)   )
            ( ix2_1_bbbb(h1_ob,p1_vb,h2_ob,h3_ob)    =  1.0 * ix2_1(h1_ob,p1_vb,h2_ob,h3_ob)   )
            ( ix2_2_a(h1_oa,h2_oa)                   =  1.0 * ix2_2(h1_oa,h2_oa)               )
            ( ix2_2_b(h1_ob,h2_ob)                   =  1.0 * ix2_2(h1_ob,h2_ob)               )
            ( ix2_3_a(p1_va,p2_va)                   =  1.0 * ix2_3(p1_va,p2_va)               )
            ( ix2_3_b(p1_vb,p2_vb)                   =  1.0 * ix2_3(p1_vb,p2_vb)               )
            ( ix2_4_aaaa(h1_oa,h2_oa,h3_oa,h4_oa)    =  1.0 * ix2_4(h1_oa,h2_oa,h3_oa,h4_oa)   )
            ( ix2_4_abab(h1_oa,h2_ob,h3_oa,h4_ob)    =  1.0 * ix2_4(h1_oa,h2_ob,h3_oa,h4_ob)   )  
            ( ix2_4_bbbb(h1_ob,h2_ob,h3_ob,h4_ob)    =  1.0 * ix2_4(h1_ob,h2_ob,h3_ob,h4_ob)   )    
            ( ix2_5_aaaa(h1_oa,p1_va,h2_oa,p2_va)    =  1.0 * ix2_5(h1_oa,p1_va,h2_oa,p2_va)   )
            ( ix2_5_abba(h1_oa,p1_vb,h2_ob,p2_va)    =  1.0 * ix2_5(h1_oa,p1_vb,h2_ob,p2_va)   )
            ( ix2_5_abab(h1_oa,p1_vb,h2_oa,p2_vb)    =  1.0 * ix2_5(h1_oa,p1_vb,h2_oa,p2_vb)   )
            ( ix2_5_bbbb(h1_ob,p1_vb,h2_ob,p2_vb)    =  1.0 * ix2_5(h1_ob,p1_vb,h2_ob,p2_vb)   )
            ( ix2_5_baab(h1_ob,p1_va,h2_oa,p2_vb)    =  1.0 * ix2_5(h1_ob,p1_va,h2_oa,p2_vb)   )
            ( ix2_5_baba(h1_ob,p1_va,h2_ob,p2_va)    =  1.0 * ix2_5(h1_ob,p1_va,h2_ob,p2_va)   )
            ( ix2_6_2_a(h1_oa,p1_va)                 =  1.0 * ix2_6_2(h1_oa,p1_va)             )
            ( ix2_6_2_b(h1_ob,p1_vb)                 =  1.0 * ix2_6_2(h1_ob,p1_vb)             )   
            ( ix2_6_3_aaaa(h1_oa,h2_oa,h3_oa,p1_va)  =  1.0 * ix2_6_3(h1_oa,h2_oa,h3_oa,p1_va) )
            ( ix2_6_3_abba(h1_oa,h2_ob,h3_ob,p1_va)  =  1.0 * ix2_6_3(h1_oa,h2_ob,h3_ob,p1_va) )
            ( ix2_6_3_abab(h1_oa,h2_ob,h3_oa,p1_vb)  =  1.0 * ix2_6_3(h1_oa,h2_ob,h3_oa,p1_vb) )
            ( ix2_6_3_bbbb(h1_ob,h2_ob,h3_ob,p1_vb)  =  1.0 * ix2_6_3(h1_ob,h2_ob,h3_ob,p1_vb) )
            ( ix2_6_3_baab(h1_ob,h2_oa,h3_oa,p1_vb)  =  1.0 * ix2_6_3(h1_ob,h2_oa,h3_oa,p1_vb) )
            ( ix2_6_3_baba(h1_ob,h2_oa,h3_ob,p1_va)  =  1.0 * ix2_6_3(h1_ob,h2_oa,h3_ob,p1_va) )
            .deallocate(lt12_o,
                        ix1_1_1,
                        ix2_1_1,ix2_1_3,ix2_1_temp,ix2_1,
                        ix2_2,
                        ix2_3,
                        ix2_4_1,ix2_4_temp,ix2_4,
                        ix2_5,
                        ix2_6_2,ix2_6_3,
                        v2ijkl,v2iabc);
            
            #ifdef USE_TALSH
              sch.execute(ExecutionHW::GPU);
            #else
              sch.execute();
            #endif    

        #if GF_IN_SG
        }
        ec.pg().barrier();
        #endif
        write_to_disk(t2v2_o,t2v2_o_file);
        write_to_disk(lt12_o_a,lt12_o_a_file);
        write_to_disk(lt12_o_b,lt12_o_b_file);
        write_to_disk(ix1_1_1_a,ix1_1_1_a_file);
        write_to_disk(ix1_1_1_b,ix1_1_1_b_file);
        write_to_disk(ix2_1_aaaa,ix2_1_aaaa_file);
        write_to_disk(ix2_1_bbbb,ix2_1_bbbb_file);
        write_to_disk(ix2_1_abab,ix2_1_abab_file);
        write_to_disk(ix2_1_baba,ix2_1_baba_file);
        write_to_disk(ix2_2_a,ix2_2_a_file);
        write_to_disk(ix2_2_b,ix2_2_b_file);
        write_to_disk(ix2_3_a,ix2_3_a_file);
        write_to_disk(ix2_3_b,ix2_3_b_file);
        write_to_disk(ix2_4_aaaa,ix2_4_aaaa_file);
        write_to_disk(ix2_4_abab,ix2_4_abab_file);
        write_to_disk(ix2_4_bbbb,ix2_4_bbbb_file);
        write_to_disk(ix2_5_aaaa,ix2_5_aaaa_file);
        write_to_disk(ix2_5_abba,ix2_5_abba_file);
        write_to_disk(ix2_5_abab,ix2_5_abab_file);
        write_to_disk(ix2_5_bbbb,ix2_5_bbbb_file);
        write_to_disk(ix2_5_baab,ix2_5_baab_file);
        write_to_disk(ix2_5_baba,ix2_5_baba_file);
        write_to_disk(ix2_6_2_a,ix2_6_2_a_file);
        write_to_disk(ix2_6_2_b,ix2_6_2_b_file);
        write_to_disk(ix2_6_3_aaaa,ix2_6_3_aaaa_file);
        write_to_disk(ix2_6_3_abba,ix2_6_3_abba_file);
        write_to_disk(ix2_6_3_abab,ix2_6_3_abab_file);
        write_to_disk(ix2_6_3_bbbb,ix2_6_3_bbbb_file);
        write_to_disk(ix2_6_3_baab,ix2_6_3_baab_file);
        write_to_disk(ix2_6_3_baba,ix2_6_3_baba_file);
      }
      
    auto inter_read_end = std::chrono::high_resolution_clock::now();
    double total_inter_time = 
        std::chrono::duration_cast<std::chrono::duration<double>>((inter_read_end - inter_read_start)).count();
    if(rank == 0) std::cout << std::endl << "GFCC: Time taken for reading/writing intermediates: " << total_inter_time << " secs" << std::endl;

      ///////////////////////////////////////
      //                                   //
      //  performing retarded_alpha first  //
      //                                   //
      ///////////////////////////////////////
      if(rank == 0) {
        cout << endl << "_____retarded_GFCCSD_on_alpha_spin______" << endl;
        //ofs_profile << endl << "_____retarded_GFCCSD_on_alpha_spin______" << endl;
      }

      while (true) {

        const std::string levelstr = std::to_string(level);      
        std::string q1_a_file    = files_prefix+".r_q1_a.l"+levelstr;
        std::string q2_aaa_file  = files_prefix+".r_q2_aaa.l"+levelstr;
        std::string q2_bab_file  = files_prefix+".r_q2_bab.l"+levelstr;
        std::string hx1_a_file   = files_prefix+".r_hx1_a.l"+levelstr;
        std::string hx2_aaa_file = files_prefix+".r_hx2_aaa.l"+levelstr;
        std::string hx2_bab_file = files_prefix+".r_hx2_bab.l"+levelstr;
        std::string hsub_a_file  = files_prefix+".r_hsub_a.l"+levelstr;
        std::string bsub_a_file  = files_prefix+".r_bsub_a.l"+levelstr;
        std::string cp_a_file    = files_prefix+".r_cp_a.l"+levelstr;
      
        bool gf_restart = fs::exists(q1_a_file)    && 
                          fs::exists(q2_aaa_file)  && fs::exists(q2_bab_file)  &&
                          fs::exists(hx1_a_file)   && 
                          fs::exists(hx2_aaa_file) && fs::exists(hx2_bab_file) &&
                          fs::exists(hsub_a_file)  && fs::exists(bsub_a_file)  && 
                          fs::exists(cp_a_file)    && ccsd_options.gf_restart;

        if(level==1) {
          omega_extra.push_back(omega_min_ip);
          omega_extra.push_back(omega_max_ip);
        }

        for(auto x: omega_extra) 
          omega_extra_finished.push_back(x);

        auto qr_rank = omega_extra_finished.size() * noa;

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
        // auto [u1] = unit_tis.labels<1>("all");

        for(auto x: omega_extra) {
          // omega_extra_finished.push_back(x);
          ndiis=ccsd_options.gf_ndiis;
          gf_omega = x;
          if(!gf_restart){
              gfccsd_driver_ip_a<T>(ec, *sub_ec, subcomm, MO, 
                              d_t1_a, d_t1_b, d_t2_aaaa, d_t2_bbbb, d_t2_abab, 
                              d_f1, t2v2_o, lt12_o_a, lt12_o_b,
                              ix1_1_1_a, ix1_1_1_b,
                              ix2_1_aaaa, ix2_1_abab, ix2_1_bbbb, ix2_1_baba,
                              ix2_2_a, ix2_2_b, 
                              ix2_3_a, ix2_3_b, 
                              ix2_4_aaaa, ix2_4_abab, ix2_4_bbbb, 
                              ix2_5_aaaa, ix2_5_abba, ix2_5_abab, 
                              ix2_5_bbbb, ix2_5_baab, ix2_5_baba,
                              ix2_6_2_a, ix2_6_2_b, 
                              ix2_6_3_aaaa, ix2_6_3_abba, ix2_6_3_abab,
                              ix2_6_3_bbbb, ix2_6_3_baab, ix2_6_3_baba,
                              v2ijab_aaaa, v2ijab_abab, v2ijab_bbbb,
                              p_evl_sorted_occ,p_evl_sorted_virt, total_orbitals, nocc, nvir,
                              nptsi,unit_tis,files_prefix,levelstr,noa);
          }
          else if(rank==0) cout << endl << "Restarting freq: " << gf_omega << endl;
        }
 
        ComplexTensor  q1_tamm_a{o_alpha,otis};
        ComplexTensor  q2_tamm_aaa{v_alpha,o_alpha,o_alpha,otis};
        ComplexTensor  q2_tamm_bab{v_beta, o_alpha,o_beta, otis};
        ComplexTensor Hx1_tamm_a{o_alpha,otis};   
        ComplexTensor Hx2_tamm_aaa{v_alpha,o_alpha,o_alpha,otis};
        ComplexTensor Hx2_tamm_bab{v_beta, o_alpha,o_beta, otis};
  
        if(!gf_restart) {
          auto cc_t1 = std::chrono::high_resolution_clock::now();
  
          sch.allocate(q1_tamm_a, q2_tamm_aaa, q2_tamm_bab,
                      Hx1_tamm_a,Hx2_tamm_aaa,Hx2_tamm_bab).execute();
          
          const std::string plevelstr = std::to_string(level-1);
  
          std::string pq1_a_file    = files_prefix+".r_q1_a.l"+plevelstr;
          std::string pq2_aaa_file  = files_prefix+".r_q2_aaa.l"+plevelstr;
          std::string pq2_bab_file  = files_prefix+".r_q2_bab.l"+plevelstr;
          
          decltype(qr_rank) ivec_start = 0;
          bool prev_q12 = fs::exists(pq1_a_file) && fs::exists(pq2_aaa_file) && fs::exists(pq2_bab_file);
  
          if(prev_q12) {
            TiledIndexSpace otis_prev{otis,range(0,prev_qr_rank)};
            auto [op1] = otis_prev.labels<1>("all");
            ComplexTensor q1_prev_a  {o_alpha,otis_prev};
            ComplexTensor q2_prev_aaa{v_alpha,o_alpha,o_alpha,otis_prev};
            ComplexTensor q2_prev_bab{v_beta, o_alpha,o_beta, otis_prev};
            sch.allocate(q1_prev_a,q2_prev_aaa,q2_prev_bab).execute();
  
            read_from_disk(q1_prev_a,pq1_a_file);
            read_from_disk(q2_prev_aaa,pq2_aaa_file);
            read_from_disk(q2_prev_bab,pq2_bab_file);
            
            ivec_start = prev_qr_rank;
  
            if(subcomm != MPI_COMM_NULL){
              sub_sch
                (q1_tamm_a(h1_oa,op1) = q1_prev_a(h1_oa,op1))
                (q2_tamm_aaa(p1_va,h1_oa,h2_oa,op1) = q2_prev_aaa(p1_va,h1_oa,h2_oa,op1))
                (q2_tamm_bab(p1_vb,h1_oa,h2_ob,op1) = q2_prev_bab(p1_vb,h1_oa,h2_ob,op1)).execute();
            }          
            sch.deallocate(q1_prev_a,q2_prev_aaa,q2_prev_bab).execute();           
          }     
  
          auto cc_t2 = std::chrono::high_resolution_clock::now();
          double time  = std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
          if(rank == 0) cout << endl << "Time to read in pre-computed Q1/Q2: " << time << " secs" << endl;
  
          ComplexTensor q1_tmp_a{o_alpha};
          ComplexTensor q2_tmp_aaa{v_alpha,o_alpha,o_alpha};
          ComplexTensor q2_tmp_bab{v_beta, o_alpha,o_beta};
        
          //TODO: optimize Q1/Q2 computation
          //Gram-Schmidt orthogonalization
          double time_gs_orth = 0.0;
          double time_gs_norm = 0.0;
          double total_time_gs  = 0.0;
  
          bool q_exist = fs::exists(q1_a_file) && fs::exists(q2_aaa_file) && fs::exists(q2_bab_file);
  
          if(!q_exist){
            sch.allocate(q1_tmp_a, q2_tmp_aaa, q2_tmp_bab).execute();
  
            for(decltype(qr_rank) ivec=ivec_start;ivec<qr_rank;ivec++) {
          
              auto cc_t0 = std::chrono::high_resolution_clock::now();
  
              auto W_read = omega_extra_finished[ivec/(noa)];
              auto pi_read = ivec%(noa);
              std::stringstream gfo;
              gfo << std::defaultfloat << W_read;
              
              std::string x1_a_wpi_file = files_prefix+".x1_a.w"+gfo.str()+".oi"+std::to_string(pi_read);
              std::string x2_aaa_wpi_file = files_prefix+".x2_aaa.w"+gfo.str()+".oi"+std::to_string(pi_read);
              std::string x2_bab_wpi_file = files_prefix+".x2_bab.w"+gfo.str()+".oi"+std::to_string(pi_read);
  
              if(fs::exists(x1_a_wpi_file) && fs::exists(x2_aaa_wpi_file) && fs::exists(x2_bab_wpi_file)){
                read_from_disk(q1_tmp_a,x1_a_wpi_file);
                read_from_disk(q2_tmp_aaa,x2_aaa_wpi_file);
                read_from_disk(q2_tmp_bab,x2_bab_wpi_file);  
              }
              else {
                nwx_terminate("ERROR: At least one of " + x1_a_wpi_file + " and " + x2_aaa_wpi_file + " and " + x2_bab_wpi_file + " do not exist!");
              }
  
              if(ivec>0){
                TiledIndexSpace tsc{otis, range(0,ivec)};
                auto [sc] = tsc.labels<1>("all");
  
                ComplexTensor oscalar{tsc};
                ComplexTensor x1c_a{o_alpha,tsc};
                ComplexTensor x2c_aaa{v_alpha,o_alpha,o_alpha,tsc};
                ComplexTensor x2c_bab{v_beta, o_alpha,o_beta, tsc};
  
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
                    (oscalar(sc)  = -1.0 * q1_tmp_a(h1_oa) * x1c_a(h1_oa,sc))
                    (oscalar(sc) += -0.5 * q2_tmp_aaa(p1_va,h1_oa,h2_oa) * x2c_aaa(p1_va,h1_oa,h2_oa,sc))
                    (oscalar(sc) += -1.0 * q2_tmp_bab(p1_vb,h1_oa,h2_ob) * x2c_bab(p1_vb,h1_oa,h2_ob,sc))
  
                    (q1_tmp_a(h1_oa) += oscalar(sc) * q1_tamm_a(h1_oa,sc))
                    (q2_tmp_aaa(p1_va,h1_oa,h2_oa) += oscalar(sc) * q2_tamm_aaa(p1_va,h1_oa,h2_oa,sc))
                    (q2_tmp_bab(p1_vb,h1_oa,h2_ob) += oscalar(sc) * q2_tamm_bab(p1_vb,h1_oa,h2_ob,sc))
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
              T newsc = 1.0/std::real(sqrt(q1norm_a*q1norm_a + 0.5*q2norm_aaa*q2norm_aaa + q2norm_bab*q2norm_bab));
  
              std::complex<T> cnewsc = static_cast<std::complex<T>>(newsc);
  
              TiledIndexSpace tsc{otis, range(ivec,ivec+1)};
              auto [sc] = tsc.labels<1>("all");
  
              if(subcomm != MPI_COMM_NULL){
                sub_sch
                (q1_tamm_a(h1_oa,sc) = cnewsc * q1_tmp_a(h1_oa))
                (q2_tamm_aaa(p2_va,h1_oa,h2_oa,sc) = cnewsc * q2_tmp_aaa(p2_va,h1_oa,h2_oa))
                (q2_tamm_bab(p2_vb,h1_oa,h2_ob,sc) = cnewsc * q2_tmp_bab(p2_vb,h1_oa,h2_ob))
                .execute();
              }
              ec.pg().barrier();
  
              auto cc_gs = std::chrono::high_resolution_clock::now();
              time_gs_norm  += std::chrono::duration_cast<std::chrono::duration<double>>((cc_gs - cc_t1)).count();
              total_time_gs   += std::chrono::duration_cast<std::chrono::duration<double>>((cc_gs - cc_t0)).count();
            } //end of Gram-Schmidt for loop over ivec
  
            sch.deallocate(q1_tmp_a,q2_tmp_aaa,q2_tmp_bab).execute();
  
            write_to_disk(q1_tamm_a,   q1_a_file);
            write_to_disk(q2_tamm_aaa, q2_aaa_file);
            write_to_disk(q2_tamm_bab, q2_bab_file);
          } //end of !gs-restart
          else { //restart GS
            read_from_disk(q1_tamm_a,   q1_a_file);
            read_from_disk(q2_tamm_aaa, q2_aaa_file);
            read_from_disk(q2_tamm_bab, q2_bab_file);
          }
  
          if(rank == 0) {
            cout << endl << "Time for orthogonalization: " << time_gs_orth << " secs" << endl;
            cout << endl << "Time for normalizing and copying back: " << time_gs_norm << " secs" << endl;
            cout << endl << "Total time for Gram-Schmidt: " << total_time_gs << " secs" << endl;
          }
          auto cc_gs_x = std::chrono::high_resolution_clock::now();
  
          bool gs_x12_restart = fs::exists(hx1_a_file) && fs::exists(hx2_aaa_file) && fs::exists(hx2_bab_file);
  
          if(!gs_x12_restart){
            #if GF_IN_SG
            if(subcomm != MPI_COMM_NULL){           
              gfccsd_x1_a(sub_sch,
            #else 
              gfccsd_x1_a(sch,
            #endif
                    MO, Hx1_tamm_a,
                    d_t1_a, d_t1_b, d_t2_aaaa, d_t2_bbbb, d_t2_abab,
                    q1_tamm_a, q2_tamm_aaa, q2_tamm_bab, 
                    d_f1, ix2_2_a, ix1_1_1_a, ix1_1_1_b,
                    ix2_6_3_aaaa, ix2_6_3_abab,
                    otis,true);
  
            #if GF_IN_SG
              gfccsd_x2_a(sub_sch,
            #else 
              gfccsd_x2_a(sch,
            #endif
                    MO, Hx2_tamm_aaa, Hx2_tamm_bab, 
                    d_t1_a, d_t1_b, d_t2_aaaa, d_t2_bbbb, d_t2_abab,
                    q1_tamm_a, q2_tamm_aaa, q2_tamm_bab, 
                    d_f1, ix2_1_aaaa, ix2_1_abab,
                    ix2_2_a, ix2_2_b,
                    ix2_3_a, ix2_3_b, 
                    ix2_4_aaaa, ix2_4_abab,
                    ix2_5_aaaa, ix2_5_abba, ix2_5_abab, 
                    ix2_5_bbbb, ix2_5_baab,
                    ix2_6_2_a, ix2_6_2_b, 
                    ix2_6_3_aaaa, ix2_6_3_abba, ix2_6_3_abab,
                    ix2_6_3_bbbb, ix2_6_3_baab,
                    v2ijab_aaaa, v2ijab_abab, v2ijab_bbbb,
                    otis,true);
  
            #if GF_IN_SG
              #ifdef USE_TALSH
                sub_sch.execute(ExecutionHW::GPU);
              #else
                sub_sch.execute();
              #endif       
            }
            ec.pg().barrier();
            #else 
              #ifdef USE_TALSH
                sch.execute(ExecutionHW::GPU);
              #else
                sch.execute();
              #endif       
            #endif
            write_to_disk(Hx1_tamm_a,  hx1_a_file);
            write_to_disk(Hx2_tamm_aaa,hx2_aaa_file);
            write_to_disk(Hx2_tamm_bab,hx2_bab_file);
          }
          else {
            read_from_disk(Hx1_tamm_a,  hx1_a_file);
            read_from_disk(Hx2_tamm_aaa,hx2_aaa_file);
            read_from_disk(Hx2_tamm_bab,hx2_bab_file);
          }      
          auto cc_q12 = std::chrono::high_resolution_clock::now();
          double time_q12  = std::chrono::duration_cast<std::chrono::duration<double>>((cc_q12 - cc_gs_x)).count();
          if(rank == 0) cout << endl << "Time to contract Q1/Q2: " << time_q12 << " secs" << endl;              
        } //if !gf_restart
  
        prev_qr_rank = qr_rank;
  
        auto cc_t1 = std::chrono::high_resolution_clock::now();

        auto [otil,otil1,otil2] = otis.labels<3>("all");
        ComplexTensor hsub_tamm_a{otis,otis};  
        ComplexTensor bsub_tamm_a{otis,o_alpha};  
        ComplexTensor Cp_a{o_alpha,otis};
        ComplexTensor::allocate(&ec,hsub_tamm_a,bsub_tamm_a,Cp_a);

        if(!gf_restart){
  
          ComplexTensor p1_k_a{v_alpha,otis};
          ComplexTensor q1_conj_a   = tamm::conj(q1_tamm_a  );
          ComplexTensor q2_conj_aaa = tamm::conj(q2_tamm_aaa);
          ComplexTensor q2_conj_bab = tamm::conj(q2_tamm_bab);      
  
          sch
             (bsub_tamm_a(otil1,h1_oa)  =       q1_conj_a(h1_oa,otil1))
             (hsub_tamm_a(otil1,otil2)  =       q1_conj_a(h1_oa,otil1) * Hx1_tamm_a(h1_oa,otil2))
             (hsub_tamm_a(otil1,otil2) += 0.5 * q2_conj_aaa(p1_va,h1_oa,h2_oa,otil1) * Hx2_tamm_aaa(p1_va,h1_oa,h2_oa,otil2))
             (hsub_tamm_a(otil1,otil2) +=       q2_conj_bab(p1_vb,h1_oa,h2_ob,otil1) * Hx2_tamm_bab(p1_vb,h1_oa,h2_ob,otil2))
             .deallocate(q1_conj_a,q2_conj_aaa,q2_conj_bab)
             
             .allocate(p1_k_a)
             ( Cp_a(h1_oa,otil)    =        q1_tamm_a(h1_oa,otil)                                     )
             ( Cp_a(h2_oa,otil)   += -1.0 * lt12_o_a(h1_oa,h2_oa) * q1_tamm_a(h1_oa,otil)               )
             ( Cp_a(h2_oa,otil)   +=        d_t1_a(p1_va,h1_oa) * q2_tamm_aaa(p1_va,h2_oa,h1_oa,otil) )
             ( Cp_a(h2_oa,otil)   +=        d_t1_b(p1_vb,h1_ob) * q2_tamm_bab(p1_vb,h2_oa,h1_ob,otil) )
             ( p1_k_a(p1_va,otil)  =        d_t2_aaaa(p1_va,p2_va,h1_oa,h2_oa) * q2_tamm_aaa(p2_va,h1_oa,h2_oa,otil) )
             ( p1_k_a(p1_va,otil) +=  2.0 * d_t2_abab(p1_va,p2_vb,h1_oa,h2_ob) * q2_tamm_bab(p2_vb,h1_oa,h2_ob,otil) )
             ( Cp_a(h1_oa,otil)   += -0.5 * p1_k_a(p1_va,otil) * d_t1_a(p1_va,h1_oa) )
             .deallocate(p1_k_a,q1_tamm_a, q2_tamm_aaa, q2_tamm_bab);

              #ifdef USE_TALSH
                sch.execute(ExecutionHW::GPU);
              #else
                sch.execute();
              #endif                   
        } //if !gf_restart

        auto cc_t2 = std::chrono::high_resolution_clock::now();
        auto time  = std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();

        if(rank == 0) cout << endl << "Time to compute Cp: " << time << " secs" << endl;

        //Write all tensors
        if(!gf_restart) {
          write_to_disk(hsub_tamm_a, hsub_a_file);
          write_to_disk(bsub_tamm_a, bsub_a_file);
          write_to_disk(Cp_a,        cp_a_file);
          sch.deallocate(Hx1_tamm_a,Hx2_tamm_aaa,Hx2_tamm_bab).execute();
        }
        else {
          read_from_disk(hsub_tamm_a, hsub_a_file);
          read_from_disk(bsub_tamm_a, bsub_a_file);
          read_from_disk(Cp_a,        cp_a_file);
        }      
  
        Complex2DMatrix hsub_a(qr_rank,qr_rank);
        Complex2DMatrix bsub_a(qr_rank,noa);
  
        tamm_to_eigen_tensor(hsub_tamm_a,hsub_a);
        tamm_to_eigen_tensor(bsub_tamm_a,bsub_a);
        Complex2DMatrix hident = Complex2DMatrix::Identity(hsub_a.rows(),hsub_a.cols());
  
        ComplexTensor xsub_local_a{otis,o_alpha};
        ComplexTensor o_local_a{o_alpha};
        ComplexTensor Cp_local_a{o_alpha,otis};
  
        ComplexTensor::allocate(&ec_l,xsub_local_a,o_local_a,Cp_local_a);
  
        Scheduler sch_l{ec_l};
        sch_l
           (Cp_local_a(h1_oa,otil) = Cp_a(h1_oa,otil))
           .execute();
  
        if(rank==0) {
          cout << endl << "spectral function (omega_npts_ip = " << omega_npts_ip << "):" <<  endl;
        }

        cc_t1 = std::chrono::high_resolution_clock::now();
        
        // Compute spectral function for designated omega regime
        for(int64_t ni=0;ni<omega_npts_ip;ni++) {
          std::complex<T> omega_tmp =  std::complex<T>(omega_min_ip + ni*omega_delta, -1.0*gf_eta);
  
          Complex2DMatrix xsub_a = (hsub_a + omega_tmp * hident).lu().solve(bsub_a);
          eigen_to_tamm_tensor(xsub_local_a,xsub_a);
            
          sch_l
             (o_local_a(h1_oa) = Cp_local_a(h1_oa,otil)*xsub_local_a(otil,h1_oa))
             .execute();
          
          auto oscalar  = std::imag(tamm::sum(o_local_a));
            
          if(level == 1) {
            omega_ip_A0[ni] = oscalar;
          }
          else {
            if (level > 1) {
              T oerr = oscalar - omega_ip_A0[ni];
              omega_ip_A0[ni] = oscalar;
              if(std::abs(oerr) < gf_threshold) omega_ip_conv_a[ni] = true; 
            }
          }
          if(rank==0){
          std::ostringstream spf;
          spf << "W = " << std::setprecision(12) << std::real(omega_tmp) << ", omega_ip_A0 =  " << omega_ip_A0[ni] << endl;
          cout << spf.str();
          }
        }
  
        cc_t2 = std::chrono::high_resolution_clock::now();
        time  = std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
        if(rank == 0) {
          cout << endl << "omegas processed in level " << level << " = " << omega_extra << endl;
          cout << "Time to compute spectral function in level " << level << " (omega_npts_ip = " << omega_npts_ip << "): " 
                    << time << " secs" << endl;
        }
  
        auto extrap_file = files_prefix+".extrapolate.retarded.alpha.txt";
        std::ostringstream spfe;
        spfe << "";
  
        // extrapolate or proceed to next level
        if(std::all_of(omega_ip_conv_a.begin(),omega_ip_conv_a.end(), [](bool x){return x;}) || gf_extrapolate_level == level) {
          if(rank==0) cout << endl << "--------------------extrapolate & converge-----------------------" << endl;
          auto cc_t1 = std::chrono::high_resolution_clock::now();
  
          AtomicCounter* ac = new AtomicCounterGA(ec.pg(), 1);
          ac->allocate(0);
          int64_t taskcount = 0;
          int64_t next = ac->fetch_add(0, 1);
  
          for(int64_t ni=0;ni<lomega_npts_ip;ni++) {
            if (next == taskcount) {
              std::complex<T> omega_tmp =  std::complex<T>(lomega_min_ip + ni*omega_delta_e, -1.0*gf_eta);
  
              Complex2DMatrix xsub_a = (hsub_a + omega_tmp * hident).lu().solve(bsub_a);
              eigen_to_tamm_tensor(xsub_local_a,xsub_a);
  
              sch_l
               (o_local_a(h1_oa) = Cp_local_a(h1_oa,otil)*xsub_local_a(otil,h1_oa))
               .execute();
            
              auto oscalar  = std::imag(tamm::sum(o_local_a));
              
              Eigen::Tensor<std::complex<T>, 1, Eigen::RowMajor> olocala_eig(noa);
              tamm_to_eigen_tensor(o_local_a,olocala_eig);
              for(TAMM_SIZE nj = 0; nj<noa; nj++){
                auto gpp = olocala_eig(nj).imag();
                spfe << "orb_index = " << nj << ", gpp_a = " << gpp << endl;
              }
  
              spfe << "w = " << std::setprecision(12) << std::real(omega_tmp) << ", A_a =  " << oscalar << endl;
              
              next = ac->fetch_add(0, 1); 
            }
            taskcount++;
          }
  
          ec.pg().barrier();
          ac->deallocate();
          delete ac;
  
          write_string_to_disk(ec,spfe.str(),extrap_file);
  
          sch.deallocate(xsub_local_a,o_local_a,Cp_local_a,
                     hsub_tamm_a,bsub_tamm_a,Cp_a).execute();  
  
          auto cc_t2 = std::chrono::high_resolution_clock::now();
          double time =
            std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
          if(rank == 0) std::cout << endl << 
          "Time taken for extrapolation (lomega_npts_ip = " << lomega_npts_ip << "): " << time << " secs" << endl;
            
          break;
        }
        else {
          if(level==1){
            auto o1 = (omega_extra[0] + omega_extra[1] ) / 2;
            omega_extra.clear();
            o1 = find_closest(o1,omega_space_ip);
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
                size_t ind = (otmp - omega_min_ip)/omega_delta;
                if (!omega_ip_conv_a[ind]) { oe_add = true; break; }
              }
  
              if(oe_add){
                T Win = (w1+w2)/2;
                
                Win = find_closest(Win,omega_space_ip);
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
                      hsub_tamm_a,bsub_tamm_a,Cp_a).execute();  

      } //end while
      //end of alpha
    }   

    sch.deallocate(cholVpr,d_f1,d_t1,d_t2).execute();

    /////////////////Free tensors////////////////////////////
    //#endif

    free_tensors(d_t1_a, d_t1_b,
                 d_t2_aaaa, d_t2_bbbb, d_t2_abab,
                 v2ijab_aaaa, v2ijab_abab, v2ijab_bbbb, v2ijab, v2ijka, v2iajb,
                 cholOO_a, cholOO_b, cholOV_a, cholOV_b, cholVV_a, cholVV_b);

    if(ccsd_options.gf_ip) {
      free_tensors(t2v2_o,
                 lt12_o_a, lt12_o_b,
                 ix1_1_1_a, ix1_1_1_b, 
                 ix2_1_aaaa, ix2_1_abab, ix2_1_bbbb, ix2_1_baba,
                 ix2_2_a, ix2_2_b, 
                 ix2_3_a, ix2_3_b, 
                 ix2_4_aaaa, ix2_4_abab, ix2_4_bbbb, 
                 ix2_5_aaaa, ix2_5_abba, ix2_5_abab, 
                 ix2_5_bbbb, ix2_5_baab, ix2_5_baba,
                 ix2_6_2_a, ix2_6_2_b,
                 ix2_6_3_aaaa, ix2_6_3_abba, ix2_6_3_abab, 
                 ix2_6_3_bbbb, ix2_6_3_baab, ix2_6_3_baba);
      }

    #ifdef USE_TALSH
    //talshStats();
    if(has_gpu) talsh_instance.shutdown();
    #endif  

  cc_t2 = std::chrono::high_resolution_clock::now();

  ccsd_time =
    std::chrono::duration_cast<std::chrono::duration<T>>((cc_t2 - cc_t1)).count();
  if(rank==0) cout << std::endl << "Time taken for GF-CCSD: " << ccsd_time << " secs" << std::endl;

  // ofs_profile.close();
  
  // --------- END GF CCSD -----------
  // GA_Summarize(0);
  ec.flush_and_sync();
  // MemoryManagerGA::destroy_coll(mgr);
  ec_l.flush_and_sync();
  // MemoryManagerLocal::destroy_coll(mgr_l);  
  if(subcomm != MPI_COMM_NULL){
    (*sub_ec).flush_and_sync();
    // MemoryManagerGA::destroy_coll(sub_mgr);
    MPI_Comm_free(&subcomm);
  }
  // delete ec;
}

#endif 
