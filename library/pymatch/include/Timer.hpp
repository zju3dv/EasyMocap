#pragma once
#include <iostream>
#include <chrono>

class Timer
{
  public:
    std::string name_;
    Timer(std::string name):name_(name){};
    Timer(){};
    void start();
    void tic();
    void toc();
    void toc(std::string things);
    void end();
    double now();

    
  private:
    std::chrono::steady_clock::time_point t_s_; //start time ponit
    std::chrono::steady_clock::time_point t_tic_; //tic time ponit
    std::chrono::steady_clock::time_point t_toc_; //toc time ponit
    std::chrono::steady_clock::time_point t_e_; //stop time point
};


void Timer::tic()
{
    t_tic_ = std::chrono::steady_clock::now();
}

void Timer::toc()
{
    t_toc_ = std::chrono::steady_clock::now();
    auto tt = std::chrono::duration_cast<std::chrono::duration<double>>(t_toc_ - t_tic_).count();
    std::cout << "Time spend: " << tt << " seconds" << std::endl;
}

void Timer::toc(std::string things)
{
    t_toc_ = std::chrono::steady_clock::now();
    auto tt = std::chrono::duration_cast<std::chrono::duration<double>>(t_toc_ - t_tic_).count();
    std::cout << "Time spend: " << tt << " seconds when doing "<<things << std::endl;
}

void Timer::start()
{
    t_s_ = std::chrono::steady_clock::now();
}

void Timer::end()
{
    t_e_ = std::chrono::steady_clock::now();
    auto tt = std::chrono::duration_cast<std::chrono::duration<double>>(t_e_ - t_s_).count();
    std::cout << "< "<<this->name_<<" > Time total spend: " << tt << " seconds" << std::endl;
}

double Timer::now()
{
    t_e_ = std::chrono::steady_clock::now();
    auto tt = std::chrono::duration_cast<std::chrono::duration<double>>(t_e_ - t_s_).count();
    return tt;
}
