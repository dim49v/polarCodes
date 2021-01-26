#pragma once

#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <thread>
#include <mutex>
#include <string> 
#include <time.h>
#include <math.h>

std::random_device r;
std::seed_seq seed{ r(), r(), r(), r(), r(), r(), r(), r() };
std::mt19937 eng(seed);

template <typename T>
inline void PrintVector(const std::vector<T>& m) {
    for (int i = 0; i < m.size(); i++) {
        std::cout << m[i] << ' ';
    }
    std::cout << '\n';
}

template <typename T>
std::vector<T> ReverseShuffle(const std::vector<T>& a) {
    int n = a.size();
    std::vector<T> c(n);
    for (int i = 0; i < n; i++) {
        c[i] = i * 2 < n
            ? a[i * 2]
            : a[1 + i * 2 - n];
    }

    return c;
}
