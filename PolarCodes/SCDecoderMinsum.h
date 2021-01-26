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

namespace SCDecoderMinsum {
    inline double Decodef(double r1, double r2) {
        return std::min(abs(r1), abs(r2)) * (r1 * r2 < 0 ? -1. : 1.);
    }
    inline double Decodeg(double r1, double r2, int b) {
        return r2 + r1 * (1 - b * 2);
    }

    std::vector<int> DecodeNode(const std::vector<double>& l, const std::vector<int>& FrozenBits, std::vector<int>& sol) {
        if (l.size() == 2) {
            int u1, u2;
            u1 = FrozenBits[sol.size()] != -1
                ? FrozenBits[sol.size()]
                : Decodef(l[0], l[1]) >= 0 ? 0 : 1;
            sol.push_back(u1);
            u2 = FrozenBits[sol.size()] != -1
                ? FrozenBits[sol.size()]
                : Decodeg(l[0], l[1], u1) >= 0 ? 0 : 1;
            sol.push_back(u2);
            return { u1 ^ u2, u2 };
        }
        else {
            std::vector<double> l1(l.size() / 2);
            std::vector<double> l2(l.size() / 2);
            for (int i = 0; i < l.size(); i+=2) {
                l1[i/2] = Decodef(l[i], l[i + 1]);
            }
            std::vector<int> v1 = DecodeNode(l1, FrozenBits, sol);
            for (int i = 0; i < l.size(); i+=2) {
                l2[i/2] = Decodeg(l[i], l[i + 1], v1[i/2]);
            }
            std::vector<int> v2 = DecodeNode(l2, FrozenBits, sol);
            std::vector<int> v3(v1.size() * 2);
            for (int i = 0; i < v3.size(); i += 2) {
                v3[i] = v1[i / 2] ^ v2[i / 2];
                v3[i + 1] = v2[i / 2];
            }
            return v3;
        }
    }

    std::vector<int> Decode(const std::vector<double>& c, const std::vector<int>& FrozenBits) {
        std::vector<int> sol(0);
        DecodeNode(c, FrozenBits, sol);

        return sol;
    }
}