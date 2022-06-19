#include "mpi.h"
#include "lbm.hpp"
#include "System.hpp"
#include <Kokkos_Core.hpp>

int main(int argc, char *argv[])
{
    int nx = 128;
    int ny = 128;

    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);
    {

        System s1(nx, ny);
        s1.Initialize();
        s1.Monitor();
        MPI_Barrier(MPI_COMM_WORLD);
        LBM l1(MPI_COMM_WORLD, s1.sx, s1.sy, s1.tau, s1.rho0, s1.rho1, s1.R,s1.delta);

        l1.Initialize();

        l1.MPIoutput(0);
        l1.setup_subdomain();
        
        for (int it = 1; it <= s1.Time; it++)
        {

            l1.Collision();
            l1.pack();
            l1.exchange();
            l1.unpack();
            l1.Streaming();
            l1.Update();

            if (it % s1.inter == 0)
            {
                l1.MPIoutput(it / s1.inter);
            }
        }
    }
    Kokkos::finalize();
    MPI_Finalize();

    return 0;
}
