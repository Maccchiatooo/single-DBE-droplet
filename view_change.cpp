#include "lbm.hpp"

void LBM::setup_subdomain()
{

    m_left = buffer_ft("m_left", q, ghost, ly);

    m_right = buffer_ft("m_right", q, ghost, ly);

    m_leftout = buffer_ft("m_leftout", q,ghost, ly);

    m_rightout = buffer_ft("m_rightout", q,ghost, ly);

    m_up = buffer_ft("m_up", q,  lx,ghost);

    m_down = buffer_ft("m_down", q,  lx,ghost);

    m_upout = buffer_ft("m_upout", q, lx,ghost);

    m_downout = buffer_ft("m_downout", q, lx,ghost);


    m_leftdown= buffer_ft("m_leftdown", q,  ghost,ghost);

    m_leftdownout= buffer_ft("m_leftdownout", q,  ghost,ghost);

    m_leftup= buffer_ft("m_leftup", q,  ghost,ghost);

    m_leftupout= buffer_ft("m_leftupout", q,  ghost,ghost);

    m_rightdown= buffer_ft("m_rightdown", q,  ghost,ghost);

    m_rightdownout= buffer_ft("m_rightdownout", q,  ghost,ghost);

    m_rightup= buffer_ft("m_rightup", q,  ghost,ghost);

    m_rightupout= buffer_ft("m_rightupout", q,  ghost,ghost);

};
void LBM::pack()
{
    if (x_lo != 0)
        Kokkos::deep_copy(m_leftout, Kokkos::subview(f, Kokkos::ALL, std::make_pair(ghost, 2*ghost), Kokkos::ALL));

    if (x_hi !=glx)
        Kokkos::deep_copy(m_rightout, Kokkos::subview(f, Kokkos::ALL, std::make_pair(lx-2*ghost, lx-ghost), Kokkos::ALL));
    
    if (y_lo != 0)
        Kokkos::deep_copy(m_downout, Kokkos::subview(f, Kokkos::ALL,  Kokkos::ALL,std::make_pair(ghost, 2*ghost)));

    if (y_hi !=gly)
        Kokkos::deep_copy(m_upout, Kokkos::subview(f, Kokkos::ALL,  Kokkos::ALL,std::make_pair(ly-2*ghost, ly-ghost)));


    if (x_lo != 0&&y_lo!=0)
        Kokkos::deep_copy(m_leftdownout, Kokkos::subview(f, Kokkos::ALL, std::make_pair(ghost, 2*ghost), std::make_pair(ghost, 2*ghost)));

    if (x_hi !=glx&&y_lo!=0)
        Kokkos::deep_copy(m_rightdownout, Kokkos::subview(f, Kokkos::ALL, std::make_pair(lx-2*ghost, lx-ghost), std::make_pair(ghost, 2*ghost)));
    
    if (x_lo!=0&&y_hi != gly)
        Kokkos::deep_copy(m_leftupout, Kokkos::subview(f, Kokkos::ALL,  std::make_pair(ghost, 2*ghost),std::make_pair(ly-2*ghost, ly-ghost)));

    if (x_hi!=glx&&y_hi !=gly)
        Kokkos::deep_copy(m_rightupout, Kokkos::subview(f, Kokkos::ALL,  std::make_pair(lx-2*ghost, lx-ghost),std::make_pair(ly-2*ghost, ly-ghost)));
};

void LBM::exchange()
{
    int mar = 1;

    if (x_lo != 0)
        MPI_Send(m_leftout.data(), m_leftout.size(), MPI_DOUBLE, comm.left, mar, comm.comm);

    if (x_hi != glx)
        MPI_Recv(m_right.data(), m_right.size(), MPI_DOUBLE, comm.right, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 2;
    if (x_hi != glx)
        MPI_Send(m_rightout.data(), m_rightout.size(), MPI_DOUBLE, comm.right, mar, comm.comm);

    if (x_lo != 0)
        MPI_Recv(m_left.data(),m_left.size(), MPI_DOUBLE, comm.left, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 3;

    if (y_lo != 0)
        MPI_Send(m_downout.data(), m_downout.size(), MPI_DOUBLE, comm.down, mar, comm.comm);

    if (y_hi != gly)
        MPI_Recv(m_up.data(), m_up.size(), MPI_DOUBLE, comm.up, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 4;
    if (y_hi != gly)
        MPI_Send(m_upout.data(), m_upout.size(), MPI_DOUBLE, comm.up, mar, comm.comm);

    if (y_lo != 0)
        MPI_Recv(m_down.data(),m_down.size(), MPI_DOUBLE, comm.down, mar, comm.comm, MPI_STATUSES_IGNORE);




    mar = 5;

    if (x_lo != 0&&y_lo!=0)
        MPI_Send(m_leftdownout.data(), m_leftdownout.size(), MPI_DOUBLE, comm.leftdown, mar, comm.comm);

    if (x_hi != glx&&y_hi!=gly)
        MPI_Recv(m_rightup.data(), m_rightup.size(), MPI_DOUBLE, comm.rightup, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 6;
    if (x_hi != glx&&y_lo!=0)
        MPI_Send(m_rightdownout.data(), m_rightdownout.size(), MPI_DOUBLE, comm.rightdown, mar, comm.comm);

    if (x_lo != 0&&y_hi!=gly)
        MPI_Recv(m_leftup.data(),m_leftup.size(), MPI_DOUBLE, comm.leftup, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 7;

    if (x_hi!=glx&&y_hi != gly)
        MPI_Send(m_rightupout.data(), m_rightupout.size(), MPI_DOUBLE, comm.rightup, mar, comm.comm);

    if (x_lo!=0&&y_lo != 0)
        MPI_Recv(m_leftdown.data(), m_leftdown.size(), MPI_DOUBLE, comm.leftdown, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 8;
    if (x_lo!=0&&y_hi != gly)
        MPI_Send(m_leftupout.data(), m_leftupout.size(), MPI_DOUBLE, comm.leftup, mar, comm.comm);

    if (x_hi!=glx&&y_lo != 0)
        MPI_Recv(m_rightdown.data(),m_rightdown.size(), MPI_DOUBLE, comm.rightdown, mar, comm.comm, MPI_STATUSES_IGNORE);

  
    MPI_Barrier(MPI_COMM_WORLD);
};

void LBM::unpack()
{
    if (x_lo != 0){
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, std::make_pair(0, ghost), Kokkos::ALL), m_left);}

    if (x_hi != glx ){
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, std::make_pair(lx-ghost, lx), Kokkos::ALL), m_right);}

    if (y_lo != 0){
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, Kokkos::ALL, std::make_pair(0, ghost)), m_down);}

    if (y_hi != gly ){
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, Kokkos::ALL, std::make_pair(ly-ghost, ly)), m_up);}

    if (x_lo != 0&&y_lo!=0){
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, std::make_pair(0, ghost), std::make_pair(0, ghost)), m_leftdown);}

    if (x_hi != glx&&y_lo!=0 ){
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, std::make_pair(lx-ghost, lx), std::make_pair(0, ghost)), m_rightdown);}

    if (x_lo!=0&&y_hi != gly){
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, std::make_pair(0, ghost), std::make_pair(ly-ghost, ly)), m_leftup);}

    if (x_hi!=glx&&y_hi != gly ){
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, std::make_pair(lx-ghost, lx), std::make_pair(ly-ghost, ly)), m_rightup);}

};



void LBM::setup_u()
{


    u_left = buffer_t("u_left",  ghost, ly);

    u_right = buffer_t("u_right",  ghost, ly);

    u_leftout = buffer_t("u_leftout", ghost, ly);

    u_rightout = buffer_t("u_rightout", ghost, ly);

    u_up = buffer_t("u_up",  lx,ghost);

    u_down = buffer_t("u_down",  lx,ghost);

    u_upout = buffer_t("u_upout",lx,ghost);

    u_downout = buffer_t("u_downout", lx,ghost);


    u_leftdown= buffer_t("u_leftdown", ghost,ghost);

    u_leftdownout= buffer_t("u_leftdownout",  ghost,ghost);

    u_leftup= buffer_t("u_leftup",  ghost,ghost);

    u_leftupout= buffer_t("u_leftupout",  ghost,ghost);

    u_rightdown= buffer_t("u_rightdown",  ghost,ghost);

    u_rightdownout= buffer_t("u_rightdownout",  ghost,ghost);

    u_rightup= buffer_t("u_rightup",  ghost,ghost);

    u_rightupout= buffer_t("m_rightupout",  ghost,ghost);


};

void LBM::u_pack(Kokkos::View<double **,Kokkos::CudaUVMSpace> u)
{


    if (x_lo != 0)
        Kokkos::deep_copy(u_leftout, Kokkos::subview(u, std::make_pair(ghost, 2*ghost), Kokkos::ALL));

    if (x_hi !=glx)
        Kokkos::deep_copy(u_rightout, Kokkos::subview(u,  std::make_pair(lx-2*ghost, lx-ghost), Kokkos::ALL));

    if (y_lo != 0){
        Kokkos::deep_copy(u_downout, Kokkos::subview(u, Kokkos::ALL, std::make_pair(ghost, 2 * ghost)));
    }

    if (y_hi !=gly)
        Kokkos::deep_copy(u_upout, Kokkos::subview(u,  Kokkos::ALL,std::make_pair(ly-2*ghost, ly-ghost)));


    if (x_lo != 0&&y_lo!=0)
        Kokkos::deep_copy(u_leftdownout, Kokkos::subview(u,  std::make_pair(ghost, 2*ghost), std::make_pair(ghost, 2*ghost)));

    if (x_hi !=glx&&y_lo!=0)
        Kokkos::deep_copy(u_rightdownout, Kokkos::subview(u, std::make_pair(lx-2*ghost, lx-ghost), std::make_pair(ghost, 2*ghost)));
    
    if (x_lo!=0&&y_hi != gly)
        Kokkos::deep_copy(u_leftupout, Kokkos::subview(u,   std::make_pair(ghost, 2*ghost),std::make_pair(ly-2*ghost, ly-ghost)));

    if (x_hi!=glx&&y_hi !=gly)
        Kokkos::deep_copy(u_rightupout, Kokkos::subview(u,   std::make_pair(lx-2*ghost, lx-ghost),std::make_pair(ly-2*ghost, ly-ghost)));

    

};

void LBM::u_exchange()
{
    int mar = 1;

    if (x_lo != 0)
        MPI_Send(u_leftout.data(), u_leftout.size(), MPI_DOUBLE, comm.left, mar, comm.comm);

    if (x_hi != glx)
        MPI_Recv(u_right.data(), u_right.size(), MPI_DOUBLE, comm.right, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 2;
    if (x_hi != glx)
        MPI_Send(u_rightout.data(), u_rightout.size(), MPI_DOUBLE, comm.right, mar, comm.comm);

    if (x_lo != 0)
        MPI_Recv(u_left.data(),u_left.size(), MPI_DOUBLE, comm.left, mar, comm.comm, MPI_STATUSES_IGNORE);


    mar = 3;

    if (y_lo != 0)
        MPI_Send(u_downout.data(), u_downout.size(), MPI_DOUBLE, comm.down, mar, comm.comm);

    if (y_hi != gly)
        MPI_Recv(u_up.data(), u_up.size(), MPI_DOUBLE, comm.up, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 4;
    if (y_hi != gly)
        MPI_Send(u_upout.data(), u_upout.size(), MPI_DOUBLE, comm.up, mar, comm.comm);

    if (y_lo != 0)
        MPI_Recv(u_down.data(),u_down.size(), MPI_DOUBLE, comm.down, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 5;

    if (x_lo != 0&&y_lo!=0)
        MPI_Send(u_leftdownout.data(), u_leftdownout.size(), MPI_DOUBLE, comm.leftdown, mar, comm.comm);

    if (x_hi != glx&&y_hi!=gly)
        MPI_Recv(u_rightup.data(), u_rightup.size(), MPI_DOUBLE, comm.rightup, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 6;
    if (x_hi != glx&&y_lo!=0)
        MPI_Send(u_rightdownout.data(), u_rightdownout.size(), MPI_DOUBLE, comm.rightdown, mar, comm.comm);

    if (x_lo != 0&&y_hi!=gly)
        MPI_Recv(u_leftup.data(),u_leftup.size(), MPI_DOUBLE, comm.leftup, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 7;

    if (x_hi!=glx&&y_hi != gly)
        MPI_Send(u_rightupout.data(), u_rightupout.size(), MPI_DOUBLE, comm.rightup, mar, comm.comm);

    if (x_lo!=0&&y_lo != 0)
        MPI_Recv(u_leftdown.data(), u_leftdown.size(), MPI_DOUBLE, comm.leftdown, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 8;
    if (x_lo!=0&&y_hi != gly)
        MPI_Send(u_leftupout.data(), u_leftupout.size(), MPI_DOUBLE, comm.leftup, mar, comm.comm);

    if (x_hi!=glx&&y_lo != 0)
        MPI_Recv(u_rightdown.data(),u_rightdown.size(), MPI_DOUBLE, comm.rightdown, mar, comm.comm, MPI_STATUSES_IGNORE);

  
    MPI_Barrier(MPI_COMM_WORLD);
};

void LBM::u_unpack(Kokkos::View<double **,Kokkos::CudaUVMSpace> u)
{
    if (x_lo != 0){
        Kokkos::deep_copy(Kokkos::subview(u,  std::make_pair(0, ghost), Kokkos::ALL), u_left);}

    if (x_hi != glx ){
        Kokkos::deep_copy(Kokkos::subview(u,  std::make_pair(lx-ghost, lx), Kokkos::ALL), u_right);}

    if (y_lo != 0){
        Kokkos::deep_copy(Kokkos::subview(u,  Kokkos::ALL, std::make_pair(0, ghost)), u_down);}

    if (y_hi != gly ){
        Kokkos::deep_copy(Kokkos::subview(u,  Kokkos::ALL, std::make_pair(ly-ghost, ly)), u_up);}

    if (x_lo != 0&&y_lo!=0){
        Kokkos::deep_copy(Kokkos::subview(u,  std::make_pair(0, ghost), std::make_pair(0, ghost)), u_leftdown);}

    if (x_hi != glx&&y_lo!=0 ){
        Kokkos::deep_copy(Kokkos::subview(u,  std::make_pair(lx-ghost, lx), std::make_pair(0, ghost)), u_rightdown);}

    if (x_lo!=0&&y_hi != gly){
        Kokkos::deep_copy(Kokkos::subview(u,  std::make_pair(0, ghost), std::make_pair(ly-ghost, ly)), u_leftup);}

    if (x_hi!=glx&&y_hi != gly ){
        Kokkos::deep_copy(Kokkos::subview(u,  std::make_pair(lx-ghost, lx), std::make_pair(ly-ghost, ly)), u_rightup);}
    
};

void LBM::pass(Kokkos::View<double **,Kokkos::CudaUVMSpace> u){

    
    Kokkos::fence();
    u_pack(u);
    Kokkos::fence();
    u_exchange();
    Kokkos::fence();
    u_unpack(u);
    Kokkos::fence();
    MPI_Barrier(MPI_COMM_WORLD);

};
