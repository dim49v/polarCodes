#pragma once

#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <thread>
#include <mutex>
#include <string> 
#include <time.h>
#include "Resources.cpp"

namespace SCDecoderMinsum {
    inline double Decodef(double r1, double r2) {
        return std::min(abs(r1), abs(r2)) * (r1 * r2 < 0 ? -1. : 1.);
    }
    char Decodef_char(char r1, char r2) {
        unsigned char mr1 = (r1 > 0) ? r1 : -r1;
        unsigned char mr2 = (r2 > 0) ? r2 : -r2;
        unsigned char mm = (mr1 < mr2) ? mr1 : mr2;
        unsigned char sr = (r1 ^ r2) & 0x80;
        return (sr == 0) ? mm : -mm;
    }
    inline double Decodeg(double r1, double r2, int b) {
        return r2 + r1 * (1 - b * 2);
    }

    char Decodeg_char(char r1, char r2, int b) {
        return r2 + r1 * (1 - b * 2);
    }

    void DecodeNode(int n, int m, int& solSize, double** decTree1, int** decTree2, int* sol, const std::vector<int>& frozenBits) {
        if (2 << m == n) {
            int u1, u2;
            u1 = frozenBits[solSize] != -1
                ? frozenBits[solSize]
                : Decodef(decTree1[m][0], decTree1[m][1]) > 0 ? 0 : 1;
            sol[solSize] = u1;
            solSize++;
            u2 = frozenBits[solSize] != -1
                ? frozenBits[solSize]
                : Decodeg(decTree1[m][0], decTree1[m][1], u1) > 0 ? 0 : 1;
            sol[solSize] = u2;
            solSize++;
            decTree2[m][0] = u1 ^ u2;
            decTree2[m][1] = u2;
            return;
        }
        else {
            int nodeSize = n >> m;
            for (int i = 0; i < nodeSize; i += 2) {
                decTree1[m + 1][i / 2] = Decodef(decTree1[m][i], decTree1[m][i + 1]);
            }
            DecodeNode(n, m + 1, solSize, decTree1, decTree2, sol, frozenBits);
            for (int i = 0; i < nodeSize; i += 2) {
                decTree2[m][i] = decTree2[m + 1][i / 2];
                decTree1[m + 1][i / 2] = Decodeg(decTree1[m][i], decTree1[m][i + 1], decTree2[m][i]);
            }
            DecodeNode(n, m + 1, solSize, decTree1, decTree2, sol, frozenBits);
            for (int i = 0; i < nodeSize; i += 2) {
                decTree2[m][i] = decTree2[m][i] ^ decTree2[m + 1][i / 2];
                decTree2[m][i + 1] = decTree2[m + 1][i / 2];
            }
            return;
        }
    }

    void Decode(int n, double* c, double** decTree1, int** decTree2, int* sol, const std::vector<int>& frozenBits) {
        for (int i = 0; i < n; i++) {
            decTree1[0][i] = c[i];
        }
        int solSize = 0;
        DecodeNode(n, 0, solSize, decTree1, decTree2, sol, frozenBits);

        return;
    }
}