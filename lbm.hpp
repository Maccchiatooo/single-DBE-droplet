#ifndef _LBM_H_
#define _LBM_H_

#include <cmath>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_Timer.hpp>
#include <mpi.h>

#define q 9
#define dim 2
#define ghost 3
struct CommHelper
{

    MPI_Comm comm;
    int rx, ry;
    int me;
    int px, py;
    int up, down, left, right, leftup, leftdown, rightup, rightdown;

    CommHelper(MPI_Comm comm_)
    {
        comm = comm_;
        int nranks;
        MPI_Comm_size(comm, &nranks);
        MPI_Comm_rank(comm, &me);

        ry = std::pow(1.0 * nranks, 1.0 / 2.0);
        while (nranks % ry != 0)
            ry++;

        rx = nranks / ry;

        px = me % rx;
        py = (me / rx) % ry;

        printf("rx=%d,ry=%d,px=%d,py=%d\n", rx, ry, px, py);

        left = px == 0 ? me+rx-1 : me - 1;
        right = px == rx - 1 ? me-rx+1 : me + 1;
        down = py == 0 ? px+(ry-1)*rx : me - rx;
        up = py == ry - 1 ? px : me + rx;
                
                
        leftup = (py == ry - 1) ? left%rx : left + rx;
        
        rightup = (py == ry - 1) ? right%rx : right + rx;

        leftdown = (py == 0) ? left%rx+(ry-1)*rx : left - rx;
        rightdown = ( py == 0) ? right%rx+(ry-1)*rx : right - rx;

        printf("Me:%i MyNeibors: %i %i %i %i %i %i %i %i\n", me, left, right, up, down, leftup, leftdown, rightup, rightdown);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    template <class ViewType>
    void isend_irecv(int partner, ViewType send_buffer, ViewType recv_buffer, MPI_Request *request_send, MPI_Request *request_recv)
    {
        MPI_Irecv(recv_buffer.data(), recv_buffer.size(), MPI_DOUBLE, partner, 1, comm, request_recv);
        MPI_Isend(send_buffer.data(), send_buffer.size(), MPI_DOUBLE, partner, 1, comm, request_send);
    }
};

struct LBM
{
    typedef Kokkos::RangePolicy<> range_policy;
    typedef Kokkos::MDRangePolicy<Kokkos::Rank<3>> mdrange_policy3;
    typedef Kokkos::MDRangePolicy<Kokkos::Rank<2>> mdrange_policy2;
    using buffer_ft = Kokkos::View<double ***, Kokkos::LayoutLeft, Kokkos::HostSpace>;
    using buffer_t = Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::HostSpace>;
    using buffer_ut = Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace>;

    CommHelper comm;
    
    MPI_Request mpi_requests_recv[8];
    MPI_Request mpi_requests_send[8];
    int mpi_active_requests;

    int glx;
    int gly;
    int lx = glx / comm.rx + 2 * ghost;
    int ly = gly / comm.ry + 2 * ghost;

    //Compute c1(int lx,int ly);
    int x_lo, x_hi, y_lo, y_hi;
    double rho0;
    double u0 = 0.01;
    double mu;
    double cs2=1.0/3.0;
    double tau0;
    double r0;
    double beta, kappa;
    double delta;
    double sigma=0.001*2.187;
    double rho_l, rho_v;



    buffer_ft m_left, m_right, m_down, m_up;
    buffer_ft m_leftout, m_rightout, m_downout, m_upout;
    buffer_ft m_leftup, m_rightup, m_leftdown, m_rightdown;
    buffer_ft m_leftupout, m_rightupout, m_leftdownout, m_rightdownout;
    buffer_t u_left, u_right, u_down, u_up;
    buffer_t u_leftout, u_rightout, u_downout, u_upout;
    buffer_t u_leftup, u_rightup, u_leftdown, u_rightdown;
    buffer_t u_leftupout, u_rightupout, u_leftdownout, u_rightdownout;
    Kokkos::View<double ***, Kokkos::CudaUVMSpace> f = Kokkos::View<double ***, Kokkos::CudaUVMSpace>("f", q, lx, ly);
    Kokkos::View<double ***, Kokkos::CudaUVMSpace> f_tem = Kokkos::View<double ***, Kokkos::CudaUVMSpace>("ft", q, lx, ly);
    Kokkos::View<double ***, Kokkos::CudaUVMSpace> fb = Kokkos::View<double ***, Kokkos::CudaUVMSpace>("fb", q, lx, ly);

    Kokkos::View<double **, Kokkos::CudaUVMSpace> ua = Kokkos::View<double **, Kokkos::CudaUVMSpace>("u", lx, ly);
    Kokkos::View<double **, Kokkos::CudaUVMSpace> va = Kokkos::View<double **, Kokkos::CudaUVMSpace>("v", lx, ly);
    Kokkos::View<double **, Kokkos::CudaUVMSpace> rho = Kokkos::View<double **, Kokkos::CudaUVMSpace>("rho", lx, ly);
    Kokkos::View<double **, Kokkos::CudaUVMSpace> cp = Kokkos::View<double **, Kokkos::CudaUVMSpace>("mu", lx, ly);
    Kokkos::View<double **, Kokkos::CudaUVMSpace> p = Kokkos::View<double **, Kokkos::CudaUVMSpace>("p", lx, ly);

    Kokkos::View<double ***, Kokkos::CudaUVMSpace> dc_cp = Kokkos::View<double ***, Kokkos::CudaUVMSpace>("dcmu", dim, lx, ly);
    Kokkos::View<double ***, Kokkos::CudaUVMSpace> dc_rho = Kokkos::View<double ***, Kokkos::CudaUVMSpace>("dcrho", dim, lx, ly);
    Kokkos::View<double ***, Kokkos::CudaUVMSpace> dm_cp = Kokkos::View<double ***, Kokkos::CudaUVMSpace>("dmmu", dim, lx, ly);
    Kokkos::View<double ***, Kokkos::CudaUVMSpace> dm_rho = Kokkos::View<double ***, Kokkos::CudaUVMSpace>("dmrho", dim, lx, ly);

    Kokkos::View<double **, Kokkos::CudaUVMSpace> la_rho = Kokkos::View<double **, Kokkos::CudaUVMSpace>("la", lx, ly);

    Kokkos::View<double ***, Kokkos::CudaUVMSpace> edc_cp = Kokkos::View<double ***, Kokkos::CudaUVMSpace>("edcmu", q, lx, ly);
    Kokkos::View<double ***, Kokkos::CudaUVMSpace> edc_rho = Kokkos::View<double ***, Kokkos::CudaUVMSpace>("edcrho", q, lx, ly);
    Kokkos::View<double ***, Kokkos::CudaUVMSpace> edm_cp = Kokkos::View<double ***, Kokkos::CudaUVMSpace>("edmmu", q, lx, ly);
    Kokkos::View<double ***, Kokkos::CudaUVMSpace> edm_rho = Kokkos::View<double ***, Kokkos::CudaUVMSpace>("edmrho", q, lx, ly);

    Kokkos::View<int **, Kokkos::CudaUVMSpace> e = Kokkos::View<int **, Kokkos::CudaUVMSpace>("e", q, dim);
    Kokkos::View<double *, Kokkos::CudaUVMSpace> t = Kokkos::View<double *, Kokkos::CudaUVMSpace>("t", q);
    Kokkos::View<int **, Kokkos::CudaUVMSpace> usr = Kokkos::View<int **, Kokkos::CudaUVMSpace>("usr", lx, ly);
    Kokkos::View<int **, Kokkos::CudaUVMSpace> ran = Kokkos::View<int **, Kokkos::CudaUVMSpace>("ran", lx, ly);
    Kokkos::View<int *, Kokkos::CudaUVMSpace> bb = Kokkos::View<int *, Kokkos::CudaUVMSpace>("b", q);

    LBM(MPI_Comm comm_, int sx, int sy, double &tau, double &rho0,double &rho1, double &R, double &delta) : comm(comm_), glx(sx), gly(sy), tau0(tau), rho_l(rho0),rho_v(rho1), r0(R),delta(delta){

                                                                                                                                       };

    void Initialize();
    void Collision();
    void setup_subdomain();
    void setup_u();
    void u_pack(Kokkos::View<double **,Kokkos::CudaUVMSpace> c);
    void u_unpack(Kokkos::View<double **,Kokkos::CudaUVMSpace> c);
    void u_exchange();
    void pass(Kokkos::View<double **,Kokkos::CudaUVMSpace> c);
    void pack();
    void exchange();
    void unpack();
    void Streaming();
    void Update();
    void Output(int n);
    void MPIoutput(int n);


    Kokkos::View<double***,Kokkos::CudaUVMSpace> d_c(Kokkos::View<double**,Kokkos::CudaUVMSpace> c);
    Kokkos::View<double***,Kokkos::CudaUVMSpace> d_b(Kokkos::View<double**,Kokkos::CudaUVMSpace> c);
    Kokkos::View<double***,Kokkos::CudaUVMSpace> d_m(Kokkos::View<double**,Kokkos::CudaUVMSpace> c);
    Kokkos::View<double**,Kokkos::CudaUVMSpace> laplace(Kokkos::View<double**,Kokkos::CudaUVMSpace> c);
    Kokkos::View<double***,Kokkos::CudaUVMSpace> edc(Kokkos::View<double**,Kokkos::CudaUVMSpace> c);
    Kokkos::View<double***,Kokkos::CudaUVMSpace> edb(Kokkos::View<double**,Kokkos::CudaUVMSpace> c);
    Kokkos::View<double***,Kokkos::CudaUVMSpace> edm(Kokkos::View<double**,Kokkos::CudaUVMSpace> c);
};
#endif