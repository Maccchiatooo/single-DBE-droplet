#ifndef _SYSTEM_H_
#define _SYSTEM_H_
#include "lbm.hpp"
#include <cmath>
#include <iostream>

class System
{

public:
    System(int &nx, int &ny) : sx(nx), sy(ny){};
    void Initialize();
    void Monitor();

    //  domain size
    int sx, sy;
    // speed of sound
    double cs2, cs;
    // beta,surface tension, interface thickness
    double beta, kappa, sigma, delta;
    // circle radius
    double R;
    // density0,density1
    double rho0, rho1;

    // total time
    int Time;
    // time interval
    int inter;
    // relaxation time connect to macro to micro
    double tau;
};
#endif