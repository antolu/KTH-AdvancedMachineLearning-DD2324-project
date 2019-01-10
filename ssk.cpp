#include "ssk.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>

double kb(std::string s, std::string t, const double & l, const int n) {
    // std::cout << "kb" << std::endl;
    // std::cout << s << " " << t << std::endl;
    char x = s.back();

    int u_len = -1.0;

    if (n == 0) 
        return 1;
    if (n > std::min(s.size(), t.size()))
        return 0;

    std::string s_new = s.substr(0, s.size()-1);
    std::string t_new = t.substr(0, t.size()-1);

    double kb_res = 0.0;
    
    // std::cout << t.back() << std::endl;
    if (x == t.back()) {
        // std::cout << s << t << n << std::endl;
        kb_res = l * ( kb(s, t_new, l, n) + l * kp(s_new, t_new, l, n-1) );
        // std::cout << kb_res << std::endl;
        return kb_res;
    }
    for (int i = t.size()-1; i >= 0; i--) {
        if (t[i] == x) {
            u_len = i;
            // std::cout << u_len << std::endl;
            break;
        }
    }
    // std::cout << s << t << u_len << n  << " " << t.substr(0, u_len+1) << std::endl;
    
    if (u_len == -1.0) {
        // std::cout << "-1" << std::endl;
        return 0;
    }

    if (u_len == 0.0)
        kb_res = pow(l, u_len+1) * kb(s, t.substr(0, 1), l, n);
    else
        kb_res = pow(l, u_len+1) * kb(s, t.substr(0, u_len), l, n);
    // std::cout << kb_res << std::endl;
    return kb_res;
}

double kp(std::string s, std::string t, const double & l, const int n) {
    // std::cout << "kp" << std::endl;
    // std::cout << s << " " << t << std::endl;
    if (n == 0) 
        return 1;
    if (n > std::min(s.size(), t.size()))
        return 0;
    
    char x = s.back();

    double kp_res = l * kp(s.substr(0, s.size()-1), t, l, n);
    double kb_res = kb(s, t, l, n);

    return kp_res + kb_res;
}

double ssk(std::string s, std::string t, const double & l, const int n) {
    // std::cout << "k" << std::endl;
    if (n > std::min(s.size(), t.size()))
        return 0;

    double kp_sum = 0;
    char x = s.back();

    std::string s_new = s.substr(0, s.size()-1);

    double k_res = ssk(s_new, t, l, n);

    for (int i = 0; i < t.size(); i++) {
        if (t[i] == x) {
            kp_sum += kp(s_new, t.substr(0, i), l, n-1) * pow(l, 2.0);
        }
    }

    return k_res + kp_sum;
}

main(int argc, char const *argv[])
{
    std::string s = "science is organized knowledge";
    std::string t = "wisdom is organized life";
    // std::string s = "cat";
    // std::string t = "car";
    double l = 0.2;
    int n = 2;

    auto start = std::chrono::high_resolution_clock::now();
    double kern = ssk(s, t, l, n);
    auto duration = std::chrono::high_resolution_clock::now() - start;

    std::cout << std::setprecision(9) << kern << std::endl;
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(duration).count() << " microseconds";
    return 0;
}

// BOOST_PYTHON_MODULE(ssk) {
//     boost::python::def("SSK", ssk);
// }