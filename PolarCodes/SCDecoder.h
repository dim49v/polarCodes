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
#include "Resources.h"
namespace SCDecoder {

    double LLR(double y, double sigma) {
        return y * 2. / (sigma * sigma);
    }

    double L1(double y1, double y2) {
        double a = 2. * atanh(tanh(y1 / 2) * tanh(y2 / 2));
        if (isinf(a) || isnan(a)) {
            return std::min(abs(y1), abs(y2)) * (a < 0 ? -1 : 1);
        }
        return a;
    }

    double L2(double y1, double y2, int b) {
        double a;
        if (b == 0) {
            a = y1 + y2;
        }
        else {
            a = y2 - y1;
        }
        return a;
    }

    std::vector<int> DecodeNode(const std::vector<double>& l, const std::vector<int>& FrozenBits, std::vector<int>& sol) {
        if (l.size() == 2) {
            if (std::isnan(l[0]) || std::isnan(l[1]) || std::isinf(l[0]) || std::isinf(l[1])) {
                return {};
            }
            int u1, u2;
            u1 = FrozenBits[sol.size()] != -1
                ? FrozenBits[sol.size()]
                : L1(l[0], l[1]) > 0 ? 0 : 1;
            sol.push_back(u1);
            u2 = FrozenBits[sol.size()] != -1
                ? FrozenBits[sol.size()]
                : L2(l[0], l[1], u1) > 0 ? 0 : 1;
            sol.push_back(u2);
            return { u1 ^ u2, u2 };
        }
        else {
            std::vector<double> l1(l.size() / 2);
            std::vector<double> l2(l.size() / 2);
            for (int i = 0; i < l.size(); i+=2) {
                l1[i / 2] = L1(l[i], l[i + 1]);
            }
            std::vector<int> v1 = DecodeNode(l1, FrozenBits, sol);
            for (int i = 0; i < l.size(); i+=2) {
                l2[i / 2] = L2(l[i], l[i + 1], v1[i / 2]);
            }
            std::vector<int> v2 = DecodeNode(l2, FrozenBits, sol);
            std::vector<int> v3(v1.size() * 2);
            for (int i = 0; i < v3.size(); i+=2) {
                v3[i] = v1[i / 2] ^ v2[i / 2];
                v3[i + 1] = v2[i / 2];
            }
            return v3;
        }
    }

    std::vector<int> Decode(const std::vector<double>& c, const std::vector<int>& FrozenBits, double sigma) {
        std::vector<int> sol(0);
        std::vector<double> v(c.size());
        for (int i = 0; i < v.size(); i++) {
            v[i] = LLR(c[i], sigma);
            if (std::isinf(v[i])) {
                v[i] = DBL_MAX;
            }
        }
        DecodeNode(v, FrozenBits, sol);

        return sol;
    }
}