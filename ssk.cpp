#include "ssk.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>

double kb(std::string s, std::string t, const double & l, const int & n) {
    char x = s.back();

    if (n == 0) 
        return 1;
    if (n > std::min(s.size(), t.size()))
        return 0;

    std::string s_new = s.substr(0, s.size()-1);
    std::string t_new = t.substr(0, t.size()-1);

    double kb_res = 0.0;

    if (s.back() == t.back()) {
        kb_res = l * (kb(s, t.substr(0, t.size()-1), l, n) + l * kp(s.substr(0, s.size()-1), t.substr(0, t.size()-1), l, n-1));
    } else {
        int u_start = t.size() - 1;
        std::string u = "";
        while (u_start > 0) {
            if (t[u_start] == x) {
                break;
            } else {
                u += t[u_start];
                u_start--;
            }
        }
        if (u_start == 0 && t[0] != x) {
            return 0;
        }

        kb_res = pow(l, u.size()) * kb(s, t.substr(0, t.size() - u.size()), l, n);
    }

    return kb_res;
}

double kp(std::string s, std::string t, const double & l, const int & n) {
    if (n == 0) 
        return 1;
    if (n > std::min(s.size(), t.size()))
        return 0;
    
    char x = s.back();

    double kp_res = l * kp(s.substr(0, s.size()-1), t, l, n);
    double kb_res = kb(s, t, l, n);

    return kp_res + kb_res;
}

double ssk(std::string s, std::string t, const double & l, const int & n) {
    if (n > std::min(s.size(), t.size()))
        return 0;

    double kp_sum = 0;
    char x = s.back();

    std::string s_new = s.substr(0, s.size()-1);

    double k_res = ssk(s_new, t, l, n);

    for (int i = 0; i < t.size(); i++) {
        if (t[i] == x) {
            kp_sum += kp(s_new, t.substr(0, i), l, n-1) * pow(l, 2);
        }
    }

    return k_res + kp_sum;
}

// main(int argc, char const *argv[])
// {
//     // std::string s = "i stockholm hej jag heter gustav kjellberg jag kommer fran goteborg men bor i stockholm hej jag heter gustav kjellberg jag kommer fran goteborg men bor i stockholm hej jag heter gustav kjellberg jag kommer fran goteborg men bor i stockholm";
//     // std::string t = "i stockholm hej jag heter gustav kjellberg jag kommer fran goteborg men bor i stockholm hej jag heter gustav kjellberg jag kommer fran goteborg men bor i stockholm hej jag heter gustav kjellberg jag kommer fran goteborg men bor i stockholm";
//     // std::string t = "here is different version of library being used than the one was used to build the python module";
//     // std::string s = "science is organized knowledge";
//     // std::string t = "wisdom is organized life";
//     std::string s = "cat";
//     std::string t = "cat";
//     double l = 0.2;
//     int n = 2;

//     auto start = std::chrono::high_resolution_clock::now();
//     double kern = ssk(s, t, l, n);
//     double norm1 = ssk(s, s, l, n);
//     double norm2 = ssk(t, t, l, n);
//     auto duration = std::chrono::high_resolution_clock::now() - start;
//     std::cout << std::setprecision(9) << kern << std::endl;
//     std::cout << std::setprecision(9) << kern/sqrt(norm1*norm2) << std::endl;
//     std::cout << std::chrono::duration_cast<std::chrono::microseconds>(duration).count() << " microseconds";
//     return 0;
// }

BOOST_PYTHON_MODULE(ssk) {
    boost::python::def("SSK", ssk);
}