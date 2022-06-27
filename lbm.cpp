#include "lbm.hpp"
#include <cstring>
#include <stdexcept>
#include <iostream>

void LBM::Initialize()
{

    x_lo = (lx - 2 * ghost) * comm.px;
    x_hi = (lx - 2 * ghost) * (comm.px + 1);
    y_lo = (ly - 2 * ghost) * comm.py;
    y_hi = (ly - 2 * ghost) * (comm.py + 1);



    // weight and discrete velocity
    t(0) = 4.0 / 9.0;
    t(1) = 1.0 / 9.0;
    t(2) = 1.0 / 9.0;
    t(3) = 1.0 / 9.0;
    t(4) = 1.0 / 9.0;
    t(5) = 1.0 / 36.0;
    t(6) = 1.0 / 36.0;
    t(7) = 1.0 / 36.0;
    t(8) = 1.0 / 36.0;

    bb(0) = 0;
    bb(1) = 3;
    bb(3) = 1;
    bb(2) = 4;
    bb(4) = 2;
    bb(5) = 7;
    bb(7) = 5;
    bb(6) = 8;
    bb(8) = 6;

    e(0, 0) = 0;
    e(1, 0) = 1;
    e(2, 0) = 0;
    e(3, 0) = -1;
    e(4, 0) = 0;
    e(5, 0) = 1;
    e(6, 0) = -1;
    e(7, 0) = -1;
    e(8, 0) = 1;

    e(0, 1) = 0;
    e(1, 1) = 0;
    e(2, 1) = 1;
    e(3, 1) = 0;
    e(4, 1) = -1;
    e(5, 1) = 1;
    e(6, 1) = 1;
    e(7, 1) = -1;
    e(8, 1) = -1;

    beta = 12.0 * sigma / delta;
    kappa = 3.0 * sigma * delta / 2.0;
    setup_u();

    Kokkos::parallel_for(
        "init", mdrange_policy2({ghost, ghost}, {lx-ghost, ly-ghost}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
            int global_x = x_lo + i - ghost;
            int global_y = y_lo + j - ghost;


            double dist = 2.0 * (sqrt(pow((global_x - 0.5 * glx), 2) + pow(global_y - 0.5 * gly, 2)) - r0) / delta;

            p(i, j) = 0.0;
            rho(i, j) = 0.5 * (rho_l + rho_v) + 0.5 * (rho_l - rho_v) * tanh(dist);

            ua(i, j) = 0.0;
            va(i, j) = 0.0;
        });
    Kokkos::fence();

    pass(rho);
    la_rho = laplace(rho);
    dm_rho = d_m(rho);
    dc_rho = d_c(rho);


    Kokkos::parallel_for(
        "cp_init", mdrange_policy2({ghost, ghost}, {lx-ghost, ly-ghost}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
            
            double mu0 = 2 * beta * (rho(i, j) - rho_l) * (rho(i, j) - rho_v) * (2 * rho(i, j) - rho_l - rho_v);

            cp(i, j) = mu0 - kappa * la_rho(i, j);

        });
    Kokkos::fence();

    pass(cp);
    dm_cp = d_m(cp);
    dc_cp = d_c(cp);


    Kokkos::parallel_for(
        "initf", mdrange_policy3({0, 0, 0}, {q, lx, ly}), KOKKOS_CLASS_LAMBDA(const int ii, const int i, const int j) {
            double udc_ = ua(i, j) * (dc_rho (0, i, j)  * cs2 - rho(i, j) * dc_cp(0, i, j)) -
                          va(i, j) * (dc_rho (1, i, j)  * cs2 - rho(i, j) * dc_cp(1, i, j));

            double edc__rho = 0.5 * (rho(i + e(ii, 0), j + e(ii, 1)) - rho(i - e(ii, 0), i - e(ii, 1)));
            double edc__cp = 0.5 * (cp(i + e(ii, 0), j + e(ii, 1)) - cp(i - e(ii, 0), i - e(ii, 1)));   

            
            double gamma = t(ii) * (1.0+3.0 *    (e(ii, 0) * ua(i, j) + e(ii, 1) * va(i, j)) +
                                        4.5 * pow(e(ii, 0) * ua(i, j) + e(ii, 1) * va(i, j), 2) -
                                        1.5 * (pow(ua(i, j), 2) + pow(va(i, j), 2)));                     

            double fors_c =  edc__rho * cs2 - rho(i, j) * edc__cp - udc_;

            f(ii, i, j) = gamma*rho(i,j)- 0.5 * fors_c * gamma / cs2;
                                   
            f_tem(ii, i, j) = 0.0;
        });
    Kokkos::fence();
};
void LBM::Collision()
{
    dm_rho = d_m(rho);
    dm_cp = d_m(cp); 

    Kokkos::parallel_for(
        "collision", mdrange_policy3({0, ghost, ghost}, {q, lx - ghost, ly - ghost}), KOKKOS_CLASS_LAMBDA(const int ii, const int i, const int j) {
            
            double udm_ = ua(i, j) * (dm_rho (0, i, j)  * cs2 - rho(i, j) * dm_cp (0, i, j)) -
                          va(i, j) * (dm_rho (1, i, j)  * cs2 - rho(i, j) * dm_cp (1, i, j));
            
            double udc_ = ua(i, j) * (dc_rho (0, i, j)  * cs2 - rho(i, j) * dc_cp(0, i, j)) -
                          va(i, j) * (dc_rho (1, i, j)  * cs2 - rho(i, j) * dc_cp(1, i, j));

            double edc__rho = 0.5 * (rho(i + e(ii, 0), j + e(ii, 1)) - rho(i - e(ii, 0), i - e(ii, 1)));
            double edc__cp = 0.5 * (cp(i + e(ii, 0), j + e(ii, 1)) - cp(i - e(ii, 0), i - e(ii, 1)));

            double edm__rho = 0.25 * (5.0*rho(i + e(ii, 0), j + e(ii, 1)) - 3.0*rho(i,j)- rho(i - e(ii, 0), i - e(ii, 1))-rho(i+2*e(ii,0),j+2*e(ii,1)));
            double edm__cp = 0.25 * (5.0*cp(i + e(ii, 0), j + e(ii, 1)) - 3.0*cp(i,j)- cp(i - e(ii, 0), i - e(ii, 1))-cp(i+2*e(ii,0),j+2*e(ii,1)));


            double fors_m =  edm__rho * cs2 - rho(i, j) * edm__cp - udm_;

            double fors_c =  edc__rho * cs2 - rho(i, j) * edc__cp - udc_;

            double gamma = t(ii) * (1.0+3.0 *    (e(ii, 0) * ua(i, j) + e(ii, 1) * va(i, j)) +
                                        4.5 * pow(e(ii, 0) * ua(i, j) + e(ii, 1) * va(i, j), 2) -
                                        1.5 * (pow(ua(i, j), 2) + pow(va(i, j), 2)));

            double feq = rho(i, j) * gamma - 0.5 * fors_c * gamma / cs2;

            f(ii, i, j) = f(ii, i, j) - (f(ii, i, j) - feq) / (tau0 + 0.5) + fors_m * gamma / cs2;
        });
    Kokkos::fence();
};

void LBM::Streaming()
{
    /*if (x_lo == 0){
        Kokkos::parallel_for(
            "bc1", mdrange_policy2({0,ghost-1},{q,ly-ghost+1}), KOKKOS_CLASS_LAMBDA(const int ii,const int j) {
                if(e(ii,0)>0){
                f(ii, ghost-1, j) = f(bb(ii), ghost, j+e(ii,1));}
            });}
    Kokkos::fence();
    if (x_hi == glx){
        Kokkos::parallel_for(
            "bc2", mdrange_policy2({0,ghost-1},{q,ly-ghost+1}), KOKKOS_CLASS_LAMBDA(const int ii,const int j) {
                if(e(ii,0)<0){
                f(ii, lx - ghost, j) = f(bb(ii), lx - ghost-1, j+e(ii,1));}
            });}
    Kokkos::fence();
    if (y_lo == 0)
        Kokkos::parallel_for(
            "bc3", mdrange_policy2({0,ghost-1},{q,lx-ghost+1}), KOKKOS_CLASS_LAMBDA(const int ii,const int i) {
                if(e(ii,1)>0){
                f(ii, i, ghost-1) = f(bb(ii), i+e(ii,0), ghost);}
            });
    Kokkos::fence();
    if (y_hi == gly)
        Kokkos::parallel_for(
            "bc4", mdrange_policy2({0,ghost-1},{q,lx-ghost+1}), KOKKOS_CLASS_LAMBDA(const int ii, const int i) {
                if(e(ii,1)<0){
                f(ii, i, ly - ghost) = f(bb(ii), i+e(ii,0), ly - ghost-1);}
            });

    Kokkos::fence();*/
    Kokkos::parallel_for(
        "stream1", mdrange_policy3({0, ghost, ghost}, {q, lx - ghost, ly - ghost}), KOKKOS_CLASS_LAMBDA(const int ii, const int i, const int j) {
            f_tem(ii, i, j) = f(ii, i - e(ii, 0), j - e(ii, 1));
        });

    Kokkos::parallel_for(
        "stream2", mdrange_policy3({0, ghost, ghost}, {q, lx - ghost, ly - ghost}), KOKKOS_CLASS_LAMBDA(const int ii, const int i, const int j) {
            f(ii, i, j) = f_tem(ii, i, j);
        });
    Kokkos::fence();
};

void LBM::Update()
{

    typedef Kokkos::TeamPolicy<> team_policy;
    typedef Kokkos::TeamPolicy<>::member_type member_type;
    Kokkos::parallel_for(
        "update", team_policy(ly-2*ghost, Kokkos::AUTO), KOKKOS_CLASS_LAMBDA(const member_type &team_member) {
            const int j = team_member.league_rank()+ghost;
            Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, ghost,lx-ghost), [&](const int &i)
            {
                            ua(i, j) = 0.0;
                            va(i, j) = 0.0;
                            rho(i, j) = 0.0;

                        Kokkos::parallel_reduce(
                                         Kokkos::ThreadVectorRange(team_member, q),[&](const int& ii, double &u_tem) {
                        u_tem += f(ii, i, j) * e(ii, 0);},
                        ua(i, j));

                        Kokkos::parallel_reduce(
                                         Kokkos::ThreadVectorRange(team_member, q),[&](const int&ii, double &v_tem) {
                        v_tem += f(ii, i, j) * e(ii, 1);},
                        va(i, j));

                        Kokkos::parallel_reduce(
                                         Kokkos::ThreadVectorRange(team_member, q),[&](const int &ii, double &rho_tem) {
                        rho_tem += f(ii, i, j);},
                        rho(i, j)); 

             }); });
    Kokkos::fence();

    pass(rho);

    dc_rho = d_c(rho);
    la_rho = laplace(rho);


    Kokkos::parallel_for(
        "cp_cal", mdrange_policy2({ghost, ghost}, {lx-ghost, ly-ghost}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
            
            double mu0 = 2 * beta * (rho(i, j) - rho_l) * (rho(i, j) - rho_v) * (2 * rho(i, j) - rho_l - rho_v);

            cp(i, j) = mu0 - kappa * la_rho(i, j);
        });
    Kokkos::fence();

    pass(cp);

    dc_cp = d_c(cp);




    Kokkos::fence();
    Kokkos::parallel_for(
        "stream1", mdrange_policy2({ghost, ghost}, {lx - ghost, ly - ghost}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
            ua(i, j) = ua(i,j)+ 0.5 * (cs2 * dc_rho(0, i, j) - rho(i, j) * dc_cp(0, i, j));
            va(i, j) = va(i,j)+ 0.5 * (cs2 * dc_rho(1, i, j) - rho(i, j) * dc_cp(1, i, j));
        });
    Kokkos::fence();
    Kokkos::parallel_for(
        "stream1", mdrange_policy2({ghost, ghost}, {lx - ghost, ly - ghost}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
            ua(i, j) = ua(i,j) / rho(i, j);
            va(i, j) = va(i,j) / rho(i, j);
        });
    Kokkos::fence();
};


void LBM::MPIoutput(int n)
{
    // MPI_IO
    MPI_File fh;
    MPI_Status status;
    MPI_Offset offset = 0;

    MPI_Datatype FILETYPE, DATATYPE;
    // buffer
    int tp;
    float ttp;
    double fp;
    // min max
    double umin, umax, vmin, vmax, pmin, pmax,rhomin,rhomax;
    double uumin, uumax, vvmin, vvmax, ppmin, ppmax,rhomin_,rhomax_;
    // transfer
    double *uu, *vv, *pp, *xx, *yy, *rr;
    uu = (double *)malloc((lx - 2 * ghost) * (ly - 2 * ghost) * sizeof(double));
    vv = (double *)malloc((lx - 2 * ghost) * (ly - 2 * ghost) * sizeof(double));
    rr = (double *)malloc((lx - 2 * ghost) * (ly - 2 * ghost) * sizeof(double));
    xx = (double *)malloc((lx - 2 * ghost) * (ly - 2 * ghost) * sizeof(double));
    yy = (double *)malloc((lx - 2 * ghost) * (ly - 2 * ghost) * sizeof(double));

    for (int j = 0; j < (ly - 2 * ghost); j++)
    {
        for (int i = 0; i < (lx - 2 * ghost); i++)
        {

            uu[i + j * (lx - 2 * ghost)] = ua(i + ghost, j + ghost);
            vv[i + j * (lx - 2 * ghost)] = va(i + ghost, j + ghost);
            rr[i + j * (lx - 2 * ghost)] = rho(i + ghost, j + ghost);
            xx[i + j * (lx - 2 * ghost)] = (double)(x_lo + i) / (glx - 1);
            yy[i + j * (lx - 2 * ghost)] = (double)(y_lo + j) / (gly - 1);
        }
    }

    parallel_reduce(
        " Label", mdrange_policy2({ghost, ghost}, {lx - ghost, ly - ghost}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, double &valueToUpdate) {
         double my_value = ua(i,j);
         if(my_value > valueToUpdate ) valueToUpdate = my_value; }, Kokkos ::Max<double>(umax));
    Kokkos::fence();
    parallel_reduce(
        " Label", mdrange_policy2({ghost, ghost}, {lx - ghost, ly - ghost}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, double &valueToUpdate) {
         double my_value = va(i,j);
         if(my_value > valueToUpdate ) valueToUpdate = my_value; }, Kokkos ::Max<double>(vmax));
    Kokkos::fence();

    parallel_reduce(
        " Label", mdrange_policy2({ghost, ghost}, {lx - ghost, ly - ghost}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, double &valueToUpdate) {
         double my_value =rho(i,j);
         if(my_value > valueToUpdate ) valueToUpdate = my_value; }, Kokkos ::Max<double>(rhomax));
    Kokkos::fence();
    parallel_reduce(
        " Label", mdrange_policy2({ghost, ghost}, {lx - ghost, ly - ghost}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, double &valueToUpdate) {
         double my_value = ua(i,j);
         if(my_value < valueToUpdate ) valueToUpdate = my_value; }, Kokkos ::Min<double>(umin));
    Kokkos::fence();
    parallel_reduce(
        " Label", mdrange_policy2({ghost, ghost}, {lx - ghost, ly - ghost}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, double &valueToUpdate) {
         double my_value = va(i,j);
         if(my_value < valueToUpdate ) valueToUpdate = my_value; }, Kokkos ::Min<double>(vmin));
    Kokkos::fence();

    parallel_reduce(
        " Label", mdrange_policy2({ghost, ghost}, {lx - ghost, ly - ghost}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, double &valueToUpdate) {
         double my_value = rho(i,j);
         if(my_value < valueToUpdate ) valueToUpdate = my_value; }, Kokkos ::Min<double>(rhomin));
    Kokkos::fence();
    std::string str1 = "output" + std::to_string(n) + ".plt";
    const char *na = str1.c_str();
    std::string str2 = "#!TDV112";
    const char *version = str2.c_str();
    MPI_File_open(MPI_COMM_WORLD, na, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

    MPI_Reduce(&umin, &uumin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&umax, &uumax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Reduce(&vmin, &vvmin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&vmax, &vvmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Reduce(&rhomin, &rhomin_, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&rhomax, &rhomax_, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (comm.me == 0)
    {

        MPI_File_seek(fh, offset, MPI_SEEK_SET);
        // header !version number
        MPI_File_write(fh, version, 8, MPI_CHAR, &status);
        // INTEGER 1
        tp = 1;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);

        // 3*4+8=20
        // variable name
        tp = 5;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 120;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 121;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 117;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 118;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 112;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);

        // 20+11*4=64
        // Zone Marker
        ttp = 299.0;
        MPI_File_write(fh, &ttp, 1, MPI_REAL, &status);
        // Zone Name
        tp = 90;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 79;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 78;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 69;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 32;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 48;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 48;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 49;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);

        // 64 + 10 * 4 = 104

        // paraents
        tp = -1;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // strendid
        tp = -2;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // SOLUTION TIME
        double nn = (double)n;
        fp = nn;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        // zone color
        tp = -1;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // ZONE type
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // specify var location
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // are raw local
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // number of miscellaneous
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // ordered zone
        tp = 0;

        tp = glx;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = gly;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 1;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // AUXILIARY
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // 104 + 13 * 4 = 156
        // EOHMARKER
        ttp = 357.0;
        MPI_File_write(fh, &ttp, 1, MPI_REAL, &status);
        // DATA SECTION
        ttp = 299.0;
        MPI_File_write(fh, &ttp, 1, MPI_REAL, &status);
        // VARIABLE DATA FORMAT
        tp = 2;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);

        MPI_File_write(fh, &tp, 1, MPI_INT, &status);

        MPI_File_write(fh, &tp, 1, MPI_INT, &status);

        MPI_File_write(fh, &tp, 1, MPI_INT, &status);

        MPI_File_write(fh, &tp, 1, MPI_INT, &status);

        // PASSIVE VARIABLE
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // SHARING VARIABLE
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // ZONE NUMBER
        tp = -1;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // 156 + 10 * 4 = 196
        // MIN AND MAX VALUE FLOAT 64
        fp = 0.0;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = 1.0;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = 0.0;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = 1.0;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = uumin;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = uumax;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = vvmin;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = vvmax;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = rhomin_;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = rhomax_;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);

        // 196 + 10 * 8 = 276
    }

    offset = 276;

    int glolen[2] = {glx, gly};
    int localstart[2] = {x_lo, y_lo};
    int l_l[2] = {lx - 2 * ghost, ly - 2 * ghost};
    MPI_Type_create_subarray(dim, glolen, l_l, localstart, MPI_ORDER_FORTRAN, MPI_DOUBLE, &DATATYPE);

    MPI_Type_commit(&DATATYPE);

    MPI_Type_contiguous(5, DATATYPE, &FILETYPE);

    MPI_Type_commit(&FILETYPE);

    MPI_File_set_view(fh, offset, MPI_DOUBLE, FILETYPE, "native", MPI_INFO_NULL);

    MPI_File_write_all(fh, xx, (lx - 2 * ghost) * (ly - 2 * ghost), MPI_DOUBLE, MPI_STATUS_IGNORE);

    MPI_File_write_all(fh, yy, (lx - 2 * ghost) * (ly - 2 * ghost), MPI_DOUBLE, MPI_STATUS_IGNORE);

    MPI_File_write_all(fh, uu, (lx - 2 * ghost) * (ly - 2 * ghost), MPI_DOUBLE, MPI_STATUS_IGNORE);

    MPI_File_write_all(fh, vv, (lx - 2 * ghost) * (ly - 2 * ghost), MPI_DOUBLE, MPI_STATUS_IGNORE);

    MPI_File_write_all(fh, rr, (lx - 2 * ghost) * (ly - 2 * ghost), MPI_DOUBLE, MPI_STATUS_IGNORE);

    MPI_File_close(&fh);

    free(uu);
    free(vv);
    free(rr);
    free(xx);
    free(yy);

    MPI_Barrier(MPI_COMM_WORLD);
};
void LBM::Output(int n)
{
    std::ofstream outfile;
    std::string str = "output" + std::to_string(n) + std::to_string(comm.me);
    outfile << std::setiosflags(std::ios::fixed);
    outfile.open(str + ".dat", std::ios::out);

    outfile << "variables=x,y,u,v,p" << std::endl;
    outfile << "zone I=" << lx  << ",J=" << ly  << std::endl;

    for (int j = 0; j < ly ; j++)
    {
        for (int i = 0; i < lx ; i++)
        {

            outfile << std::setprecision(8) << setiosflags(std::ios::left) << (i - ghost + x_lo) / (glx - 1.0) << " " << (j - ghost + y_lo) / (gly - 1.0) << " " << ua(i, j) << " " << va(i, j) << " " << rho(i, j) << std::endl;
        }
    }

    outfile.close();
    if (comm.me == 0)
    {
        printf("\n");
        printf("The result %d is writen\n", n);
        printf("\n");
        printf("============================\n");
    }
};


Kokkos::View<double**,Kokkos::CudaUVMSpace> LBM::laplace(Kokkos::View<double**,Kokkos::CudaUVMSpace> c)
{

    Kokkos::View<double **, Kokkos::CudaUVMSpace> la_ = Kokkos::View<double **, Kokkos::CudaUVMSpace>("la_", lx, ly);

    typedef Kokkos::TeamPolicy<> team_policy;
    typedef Kokkos::TeamPolicy<>::member_type member_type;

    /*if (x_lo == 0){
        Kokkos::parallel_for(
            "bc1_", range_policy(ghost-1,ly-ghost+1), KOKKOS_CLASS_LAMBDA(const int j) {
                c(ghost-1, j) = c(ghost+1,j);
                c(ghost-2, j) = c(ghost+2,j);
            });}



    if (x_hi == glx){
        Kokkos::parallel_for(
            "bc2_", range_policy(ghost-1,ly-ghost+1), KOKKOS_CLASS_LAMBDA(const int j) {
                c(lx-ghost, j) = c(lx-ghost-2,j);
                c(lx-ghost+1, j) = c(lx-ghost-3,j);
            });}

    if (y_lo == 0){
        Kokkos::parallel_for(
            "bc3_", range_policy(ghost-1,lx-ghost+1), KOKKOS_CLASS_LAMBDA(const int i) {        
                 c(i, ghost-1) = c(i,ghost+1);
                 c(i, ghost-2) = c(i,ghost+2);
            });}

    if (y_hi == gly){
        Kokkos::parallel_for(
            "bc4_", range_policy(ghost-1,lx-ghost+1), KOKKOS_CLASS_LAMBDA(const int i) {      
                c(i, ly-ghost) = c(i,ly-ghost-2);
                c(i, ly-ghost+1) = c(i,ly-ghost-3);
            });}*/
    Kokkos::parallel_for(
        "laplace", team_policy(ly-2*ghost, Kokkos::AUTO), KOKKOS_CLASS_LAMBDA(const member_type &team_member) {
            const int j = team_member.league_rank()+ghost;
            Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, ghost,lx-ghost), [&](const int &i)
            {
                            la_(i, j) = 0.0;


                        Kokkos::parallel_reduce(
                                         Kokkos::ThreadVectorRange(team_member, q),[&](const int& ii, double &la_tem) {
                        la_tem += t(ii) * (c(i +  e(ii, 0), j +  e(ii, 1)) + c(i -  e(ii, 0), j -  e(ii, 1)) - 2 * c(i, j)) / 2.0 /cs2;},
                        la_(i, j));


             }); });


    Kokkos::fence();
    return la_;
};

Kokkos::View<double***,Kokkos::CudaUVMSpace> LBM::d_c(Kokkos::View<double**,Kokkos::CudaUVMSpace> c)
{
    Kokkos::View<double ***, Kokkos::CudaUVMSpace> dc= Kokkos::View<double ***, Kokkos::CudaUVMSpace>("dc_", dim, lx, ly);
    typedef Kokkos::TeamPolicy<> team_policy;
    typedef Kokkos::TeamPolicy<>::member_type member_type;

    /*if (x_lo == 0){
        Kokkos::parallel_for(
            "bc1_", range_policy(ghost-1,ly-ghost+1), KOKKOS_CLASS_LAMBDA(const int j) {
                c(ghost-1, j) = c(ghost+1,j);

            });}

    if (x_hi == glx){
        Kokkos::parallel_for(
            "bc2_", range_policy(ghost-1,ly-ghost+1), KOKKOS_CLASS_LAMBDA(const int j) {
                c(lx-ghost, j) = c(lx-ghost-2,j);
            });}

    if (y_lo == 0){
        Kokkos::parallel_for(
            "bc3_", range_policy(ghost-1,lx-ghost+1), KOKKOS_CLASS_LAMBDA(const int i) {        
                 c(i, ghost-1) = c(i,ghost+1);
            });}

    if (y_hi == gly){
        Kokkos::parallel_for(
            "bc4_", range_policy(ghost-1,lx-ghost+1), KOKKOS_CLASS_LAMBDA(const int i) {      
                c(i, ly-ghost) = c(i,ly-ghost-2);

            });}*/

    Kokkos::parallel_for(
        "dc", team_policy(ly-2*ghost, Kokkos::AUTO), KOKKOS_CLASS_LAMBDA(const member_type &team_member) {
            const int j = team_member.league_rank()+ghost;
            Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, ghost,lx-ghost), [&](const int &i)
            {
                            dc(0, i, j) = 0.0;
                            dc(1, i, j) = 0.0;

                        Kokkos::parallel_reduce(
                                         Kokkos::ThreadVectorRange(team_member, q),[&](const int& ii, double &dc0_tem) {
                        dc0_tem += t(ii) *  e(ii, 0) * (c(i + e(ii, 0), j + e(ii, 1)) - c(i - e(ii, 0), j - e(ii, 1))) / 2.0 / cs2;},
                        dc(0,i,j));

                        Kokkos::parallel_reduce(
                                         Kokkos::ThreadVectorRange(team_member, q),[&](const int& ii, double &dc1_tem) {
                        dc1_tem += t(ii) *  e(ii, 1) * (c(i + e(ii, 0), j + e(ii, 1)) - c(i - e(ii, 0), j - e(ii, 1))) / 2.0 / cs2;},
                        dc(1,i,j));


             }); });

Kokkos::fence();
    return dc;
};

Kokkos::View<double***,Kokkos::CudaUVMSpace> LBM::d_m(Kokkos::View<double**,Kokkos::CudaUVMSpace> c)
{

    Kokkos::View<double ***, Kokkos::CudaUVMSpace> dm = Kokkos::View<double ***, Kokkos::CudaUVMSpace>("dm_", dim, lx, ly);

    typedef Kokkos::TeamPolicy<> team_policy;
    typedef Kokkos::TeamPolicy<>::member_type member_type;


    /*if (x_lo == 0){
        Kokkos::parallel_for(
            "bc1__", range_policy(ghost-2,ly-ghost+2), KOKKOS_CLASS_LAMBDA(const int j) {
                c(ghost-1, j) = c(ghost+1,j);
                c(ghost-2, j) = c(ghost+2,j);

            });}

    if (x_hi == glx){
        Kokkos::parallel_for(
            "bc2__", range_policy(ghost-2,ly-ghost+2), KOKKOS_CLASS_LAMBDA(const int j) {
                c(lx-ghost, j) = c(lx-ghost-2,j);
                c(lx-ghost+1, j) = c(lx-ghost-3,j);
            });}

    if (y_lo == 0){
        Kokkos::parallel_for(
            "bc3__", range_policy(ghost-2,lx-ghost+2), KOKKOS_CLASS_LAMBDA(const int i) {        
                 c(i, ghost-1) = c(i,ghost+1);
                 c(i, ghost-2) = c(i,ghost+2);
            });}

    if (y_hi == gly){
        Kokkos::parallel_for(
            "bc4__", range_policy(ghost-2,lx-ghost+2), KOKKOS_CLASS_LAMBDA(const int i) {      
                c(i, ly-ghost) = c(i,ly-ghost-2);
                c(i, ly-ghost+1) = c(i,ly-ghost-3);

            });}*/

    Kokkos::parallel_for(
        "dm", team_policy(ly-2*ghost, Kokkos::AUTO), KOKKOS_CLASS_LAMBDA(const member_type &team_member) {
            const int j = team_member.league_rank()+ghost;
            Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, ghost,lx-ghost), [&](const int &i)
            {
                            dm(0, i, j) = 0.0;
                            dm(1, i, j) = 0.0;

                            Kokkos::parallel_reduce(
                                Kokkos::ThreadVectorRange(team_member, q), [&](const int &ii, double &db0_tem)
                                { double temp = 0.25 * (5.0 * c(i + e(ii, 0), j + e(ii, 1)) - 3.0 * c(i, j) - c(i - e(ii, 0), j - e(ii, 1)) - c(i + 2 * e(ii, 0), j + 2 * e(ii, 1)));
                                    db0_tem += t(ii) * e(ii, 0) * temp / cs2; },
                                dm(0, i, j));

                            Kokkos::parallel_reduce(
                                Kokkos::ThreadVectorRange(team_member, q), [&](const int &ii, double &db1_tem)
                                { double temp = 0.25 * (5.0 * c(i + e(ii, 0), j + e(ii, 1)) - 3.0 * c(i, j) - c(i - e(ii, 0), j - e(ii, 1)) - c(i + 2 * e(ii, 0), j + 2 * e(ii, 1)));
                                    db1_tem += t(ii) * e(ii, 1) * temp / cs2; },
                                dm(1, i, j));


             }); });

    Kokkos::fence();
    return dm;
};
