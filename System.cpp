#include "System.hpp"

void System::Initialize()
{
    // system defination
    this->cs2 = 1.0 / 3.0;
    this->cs = sqrt(cs2);

    this->rho0 = 0.1;
    this->rho1 = 1.0;
    this->beta = 0.01;
    this->sigma = 2.187 * 0.001;
    this->delta = 4;
    this->kappa = this->beta * this->delta * this->delta / 8 * (this->rho0 - this->rho1) * (this->rho0 - this->rho1);
    this->R = 0.25 * this->sy;
    this->Time = 100000;
    this->inter = 10000;
    this->tau = 0.5;
    this->delta = 4.0;
};

void System::Monitor()
{

    std::cout << "2D Cylinder Flow" << std::endl
              << "rho0   =" << this->rho0 << std::endl
              << "rho1   =" << this->rho1 << std::endl
              << "beta   =" << this->beta << std::endl
              << "kappa  =" << this->kappa << std::endl
              << "sigma  =" << this->sigma << std::endl
              << "tau    =" << this->tau << std::endl
              << "Time   =" << this->Time << std::endl
              << "inter  =" << this->inter << std::endl
              << "============================" << std::endl;
};
