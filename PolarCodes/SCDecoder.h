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

    void DecodeNode(int n, int m, int& solSize, double** decTree1, int** decTree2, int* sol, const std::vector<int>& frozenBits) {
        if (2 << m == n) {
            int u1, u2;
            u1 = frozenBits[solSize] != -1
                ? frozenBits[solSize]
                : L1(decTree1[m][0], decTree1[m][1]) > 0 ? 0 : 1;
            sol[solSize] = u1;
            solSize++;
            u2 = frozenBits[solSize] != -1
                ? frozenBits[solSize]
                : L2(decTree1[m][0], decTree1[m][1], u1) > 0 ? 0 : 1;
            sol[solSize] = u2;
            solSize++;
            decTree2[m][0] = u1 ^ u2;
            decTree2[m][1] = u2;
            return;
        }
        else {
            int nodeSize = n >> m;
            for (int i = 0; i < nodeSize; i+=2) {
                decTree1[m + 1][i / 2] = L1(decTree1[m][i], decTree1[m][i + 1]);
            }
            DecodeNode(n, m + 1, solSize, decTree1, decTree2, sol, frozenBits);
            for (int i = 0; i < nodeSize; i+=2) {
                decTree2[m][i] = decTree2[m + 1][i / 2];
                decTree1[m + 1][i / 2] = L2(decTree1[m][i], decTree1[m][i + 1], decTree2[m][i]);
            }
            DecodeNode(n, m + 1, solSize, decTree1, decTree2, sol, frozenBits);
            for (int i = 0; i < nodeSize; i+=2) {
                decTree2[m][i] = decTree2[m][i] ^ decTree2[m + 1][i / 2];
                decTree2[m][i + 1] = decTree2[m + 1][i / 2];
            }
            return;
        }
    }

    void Decode(int n, double* c, double** decTree1, int** decTree2, int* sol, const std::vector<int>& frozenBits, double sigma) {
        for (int i = 0; i < n; i++) {
            decTree1[0][i] = LLR(c[i], sigma);
            if (std::isinf(decTree1[0][i])) {
                decTree1[0][i] = DBL_MAX;
            }
        }
        int solSize = 0;
        DecodeNode(n, 0, solSize, decTree1, decTree2, sol, frozenBits);

        return;
    }
}