#ifndef SSK
#define SSK

#include <string>
#include <cmath>
#include <boost/python.hpp>

double kb(std::string s, std::string t, const double & l, const int n);

double kp(std::string, std::string t, const double & l, const int n);

double ssk(const std::string s, const std::string t, const double & l, const int & n);

#endif