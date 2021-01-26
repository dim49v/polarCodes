#pragma once

#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <thread>
#include <mutex>
#include <string> 
#include <time.h>
#include "Resources.h"

double Awgn(std::normal_distribution<double>& distributionN, int s) {
    return distributionN(eng) + s;
}

std::vector<int> AddFrozenBits(const std::vector<int>& v, const std::vector<int>& infIndexes, const std::vector<int>& frozenBits) {
    std::vector<int> c(frozenBits.size() + v.size());
    int u = 0;
    int y = 0;
    for (int i = 0; i < c.size(); i++)
    {
        if (u < infIndexes.size() && i == infIndexes[u]) {
            c[i] = v[u];
            u++;
        }
        else {
            c[i] = frozenBits[y];
            y++;
        }
    }

    return c;
}

std::vector<double> Transform(const std::vector<int>& v, std::normal_distribution<double>& distributionN) {
    std::vector<double> c(v.size());
    for (int i = 0; i < c.size(); i++) {
        c[i] = Awgn(distributionN, (v[i] == 0 ? 1 : -1));
    }

    return c;
}

std::vector<int> FrozenBits(int n, const std::vector<int>& infIndexes, const std::vector<int>& frozenBits) {
    std::vector<int> c(infIndexes.size(), -1);
    return AddFrozenBits(c, infIndexes, frozenBits);
}

std::vector<int> EncodeNode(const std::vector<int>& c) {
    if (c.size() == 2) {
        return { c[0] ^ c[1], c[1] };
    }
    std::vector<int> v(c);
    v = ReverseShuffle(v);
    int n = c.size() / 2;
    std::vector<int> x1 = EncodeNode(std::vector<int>(v.begin(), v.begin() + n));
    std::vector<int> x2 = EncodeNode(std::vector<int>(v.begin() + n, v.end()));
    for (int i = 0; i < n; i++) {
        x1[i] ^= x2[i];
    }
    x1.insert(x1.end(), x2.begin(), x2.end());
    return x1;
}
std::vector<int> Encode(const std::vector<int>& c) {
    std::vector<int> v(c);
    return EncodeNode(v);;
}