#pragma once

#include <immintrin.h>
#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <thread>
#include <mutex>
#include <string> 
#include <time.h>
#include <stack>
#include "Resources.cpp"
#include "Encoder.cpp"

namespace SCLDecoderMinsum {
    std::vector<int> inactivePathIndices;
    bool* activePath;
    char*** arrayPointer_P;
    int**** arrayPointer_C;
    int** pathIndexToArrayIndex;
    std::vector<int>* inactiveArrayIndices;
    int** arrayReferenceCount;
    int L_;
    int n_;
    int m_;
    int k_;
    int crc_;
    unsigned int** probForks;
    bool** constForks;
    unsigned int* metric;
    unsigned int* sortedMetric;

    inline double Decodef(double r1, double r2) {
        return std::min(abs(r1), abs(r2)) * (r1 * r2 < 0 ? -1. : 1.);
    }     
    char Decodef_char(char r1, char r2) {
        unsigned char mr1 = (r1 > 0) ?  r1 : -r1;
        unsigned char mr2 = (r2 > 0) ?  r2 : -r2;
        unsigned char mm = (mr1 < mr2) ? mr1 : mr2;
        unsigned char sr = (r1 ^ r2) & 0x80;
        return (sr == 0) ? mm : -mm;
    }
    __m256i f_avx(__m256i r1, __m256i r2) {
        const __m256i mask = _mm256_set1_epi8(0x40);
        __m256i mr1 = _mm256_abs_epi8(r1);
        __m256i mr2 = _mm256_abs_epi8(r2);
        __m256i mm = _mm256_min_epi8(mr1, mr2);
        __m256i ss = _mm256_xor_si256(r1, r2);
        __m256i sr = _mm256_or_si256(ss, mask);
        return _mm256_sign_epi8(mm, sr);
    }
    inline double Decodeg(double r1, double r2, int b) {
        return r2 + r1 * (1 - b * 2);
    }  
    char Decodeg_char(char r1, char r2, int b) {
        int res = r2 + r1 * (1 - b * 2);
        return abs(res) > 127 ? (res > 0 ? 127 : -127) : res;
    }

    void InitializeDataStructures(int n, int L, int k, int crc) {
        L_ = L;
        n_ = n;
        m_ = std::log2(n);
        k_ = k;
        crc_ = crc;
        inactivePathIndices.reserve(L_);
        activePath = new bool[L_];
        arrayPointer_P = new char** [m_ + 1];
        arrayPointer_C = new int*** [m_ + 1];
        pathIndexToArrayIndex = new int* [m_ + 1];
        inactiveArrayIndices = new std::vector<int>[m_ + 1];
        arrayReferenceCount = new int* [m_ + 1];
        probForks = new unsigned int* [L_];
        constForks = new bool* [L_];
        metric = new unsigned int[L_];
        sortedMetric = new unsigned int[2 * L_];
        unsigned int size = n;
        for (int i = 0; i < m_ + 1; i++) {
            pathIndexToArrayIndex[i] = new int[L_];
            inactiveArrayIndices[i].reserve(L_);
            arrayReferenceCount[i] = new int[L_];
            arrayPointer_P[i] = new char* [L_];
            arrayPointer_C[i] = new int** [L_];
            for (int u = 0; u < L_; u++) {
                arrayPointer_P[i][u] = new char[size];
                arrayPointer_C[i][u] = new int* [size];
                for (int j = 0; j < size; j++) {
                    arrayPointer_C[i][u][j] = new int[2];
                }
                arrayReferenceCount[i][u] = 0;
                inactiveArrayIndices[i].push_back(u);
            }
            size >>= 1;
        }
        for (int i = 0; i < L_; i++) {
            activePath[i] = false;
            inactivePathIndices.push_back(i);
            probForks[i] = new unsigned int[2];
            constForks[i] = new bool[2];
        }
    }
    void resetData() {
        inactivePathIndices.clear();
        for (int i = 0; i <= m_; i++) {
            inactiveArrayIndices[i].clear();
            for (int u = 0; u < L_; u++) {
                arrayReferenceCount[i][u] = 0;
                inactiveArrayIndices[i].push_back(u);
            }
        }
        for (int i = 0; i < L_; i++) {
            activePath[i] = false;
            inactivePathIndices.push_back(i);
        }
    }
    int assignInitialPath() {
        int l = inactivePathIndices.back();
        inactivePathIndices.pop_back();
        activePath[l] = true;
        int s;
        for (int i = 0; i < m_ + 1; i++) {
            s = inactiveArrayIndices[i].back();
            inactiveArrayIndices[i].pop_back();
            pathIndexToArrayIndex[i][l] = s;
            arrayReferenceCount[i][s] = 1;
        }
        metric[l] = 0;
        return l;
    }
    int clonePath(int l) {
        int ll = inactivePathIndices.back();
        inactivePathIndices.pop_back();
        activePath[ll] = true;
        int s;
        for (int i = 0; i < m_ + 1; i++) {
            s = pathIndexToArrayIndex[i][l];
            pathIndexToArrayIndex[i][ll] = s;
            arrayReferenceCount[i][s]++;
        }   
        metric[ll] = metric[l];
        return ll;
    }
    void killPath(int l) {
        activePath[l] = false;
        inactivePathIndices.push_back(l);
        int s;
        for (int i = 0; i < m_ + 1; i++) {
            s = pathIndexToArrayIndex[i][l];
            arrayReferenceCount[i][s]--;
            if (arrayReferenceCount[i][s] == 0) {
                inactiveArrayIndices[i].push_back(s);
            }
        }
    }
    int copyContent(int m, int l) {
        unsigned int size = n_ >> m;
        int ss;
        int s = pathIndexToArrayIndex[m][l];
        if (arrayReferenceCount[m][s] == 1) {
            ss = s;
        }
        else {
            ss = inactiveArrayIndices[m].back();
            inactiveArrayIndices[m].pop_back();
            for (int i = 0; i < size; i++) {
                arrayPointer_P[m][ss][i] = arrayPointer_P[m][s][i];
                arrayPointer_C[m][ss][i][0] = arrayPointer_C[m][s][i][0];
                arrayPointer_C[m][ss][i][1] = arrayPointer_C[m][s][i][1];
            }
            arrayReferenceCount[m][s]--;
            arrayReferenceCount[m][ss] = 1;
            pathIndexToArrayIndex[m][l] = ss;
        }
        
        return ss;
    }
    char* getArrayPointer_P(int m, int l) {
        int ss = copyContent(m, l);
        return arrayPointer_P[m][ss];
    }
    int** getArrayPointer_C(int m, int l) {
        int ss = copyContent(m, l);
        return arrayPointer_C[m][ss];
    }

    void recursivelyCalcP(int m, int phase) {
        if (m == 0) {
            return;
        }
        int w = phase / 2;
        if (phase % 2 == 0) {
            recursivelyCalcP(m - 1, w);
        }
        char* p1;
        char* p2;
        int** c1;
        int size = n_ >> m;;
        //char minP = 255;
        for (int i = 0; i < L_; i++) {
            if (!activePath[i]) {
                continue;
            }
            p1 = getArrayPointer_P(m, i);
            p2 = getArrayPointer_P(m - 1, i);
            c1 = getArrayPointer_C(m, i);
            for (int u = 0; u < size; u++) {
                if (phase % 2 == 0) {
                    p1[u] = Decodef_char(p2[2 * u], p2[2 * u + 1]);
                }
                else {
                    p1[u] = Decodeg_char(p2[2 * u], p2[2 * u + 1], c1[u][0]);
                }
            }
        }
    }
    void recursivelyUpdateC(int m, int phase) {
        unsigned int size = n_ >> m;
        int w = phase / 2;
        int** c1;
        int** c2;
        for (int i = 0; i < L_; i++) {
            if (!activePath[i]) {
                continue;
            }
            c1 = getArrayPointer_C(m, i);
            c2 = getArrayPointer_C(m - 1, i);
            for (int u = 0; u < size; u++) {
                c2[2 * u][w % 2] = c1[u][0] ^ c1[u][1];
                c2[2 * u + 1][w % 2] = c1[u][1];
            }
        }
        if (w % 2 == 1) {
            recursivelyUpdateC(m - 1, w);
        }
    }
    unsigned char additionPM(char llr, int u) {
        char x = -(1 - 2 * u) * llr;
        return x >= 0 ? x : 0;
    }

    void continuePaths_UnfrozenBit(int phase) {
        char* Pm;
        int** cm;
        int activeCount = 0;
        if (phase == n_ - 1) {
            phase = phase;
        }
        for (int i = 0; i < L_; i++) {
            if (activePath[i]) {
                Pm = getArrayPointer_P(m_, i);
                probForks[i][0] = metric[i] + additionPM(Pm[0], 0);
                probForks[i][1] = metric[i] + additionPM(Pm[0], 1);
                sortedMetric[2 * i] = probForks[i][0];
                sortedMetric[2 * i + 1] = probForks[i][1];
                activeCount++;
            }
        }
        if (activeCount < L_) {
            for (int i = 0; i < L_; i++) {
                if (activePath[i]) {       
                    cm = getArrayPointer_C(m_, i);
                    cm[0][phase % 2] = 0;
                    int ii = clonePath(i);
                    cm = getArrayPointer_C(m_, ii);
                    cm[0][phase % 2] = 1;              
                    metric[i] = probForks[i][0];
                    metric[ii] = probForks[i][1];
                }
            }
        }
        else {
            std::sort(sortedMetric, sortedMetric + 2 * L_);
            unsigned int med = (sortedMetric[L_ - 1] + sortedMetric[L_]) / 2;
            int constCount = 0;
            for (int i = 0; i < L_; i++) {
                if (probForks[i][0] <= med && constCount < L_) {
                    constForks[i][0] = true;
                    constCount++;
                }
                else {
                    constForks[i][0] = false;
                }
                if (probForks[i][1] <= med && constCount < L_) {
                    constForks[i][1] = true;
                    constCount++;
                }
                else {
                    constForks[i][1] = false;
                }      
                if (!constForks[i][0] && !constForks[i][1]) {
                    killPath(i);
                }
            }
            for (int i = 0; i < L_; i++) {
                if (!activePath[i] || (!constForks[i][0] && !constForks[i][1])) {
                    continue;
                }
                cm = getArrayPointer_C(m_, i);
                if (constForks[i][0] && constForks[i][1]) {
                    cm[0][phase % 2] = 0;
                    int ii = clonePath(i);
                    cm = getArrayPointer_C(m_, ii);
                    cm[0][phase % 2] = 1;
                    metric[i] = probForks[i][0];
                    metric[ii] = probForks[i][1];
                }
                else if (constForks[i][0]) {
                    cm[0][phase % 2] = 0;
                    metric[i] = probForks[i][0];
                }
                else {
                    cm[0][phase % 2] = 1;
                    metric[i] = probForks[i][1];
                }
            }
        }
    }
    void Decode(int n, int L, double* c, int** decTree2, int* sol, const std::vector<int>& frozenBits) {
        int l = assignInitialPath();
        char* pm = getArrayPointer_P(0, l);
        int** cm;
        for (int i = 0; i < n; i++) {
            pm[i] = abs(c[i] * 16 + 0.5) > 127 ? (c[i] > 0 ? 127 : -127) : c[i] * 16 + 0.5;
        }          
        for (int i = 0; i < n; i++) {
            recursivelyCalcP(m_, i);
            if (frozenBits[i] != -1) {
                for (int u = 0; u < L_; u++) {
                    if (!activePath[u]) {
                        continue;
                    }
                    cm = getArrayPointer_C(m_, u);
                    pm = getArrayPointer_P(m_, u);
                    cm[0][i % 2] = frozenBits[i];
                    metric[u] += additionPM(pm[0], frozenBits[i]);
                }
            }
            else {
                continuePaths_UnfrozenBit(i);
            }
            if (i % 2 == 1) {
                recursivelyUpdateC(m_, i);
            }
        } 
        for (int i = 0; i < L_; i++) { 
            cm = getArrayPointer_C(0, i);
            for (int u = 0; u < n_; u++) {
                sol[u] = cm[u][0]; 
            }
            for (int u = 0; u < n; u++) {
                std::cout << sol[u];
            }
            std::cout << "\n";
            Encode(n_, sol, decTree2);
            int j = 0;
            for (int u = 0; u < n_; u++) {
                if (frozenBits[u] == -1) {      
                    sol[j++] = decTree2[0][u];
                }
            }
            bool isTrue = true;
            unsigned short crcRes;
            switch (crc_)
            {
            case 8:
                crcRes = Crc8(sol, k_);
                for (int u = 1; u <= 8; u++) {
                    if (sol[k_ + crc_ - u] != (crcRes & 0x01)) {
                        isTrue = false;
                        break;
                    }
                    crcRes >>= 1;
                }
                break;
            case 16:
                crcRes = Crc16(sol, k_);
                for (int u = 1; u <= 16; u++) {
                    if (sol[k_ + crc_ - u] != (crcRes & 0x0001)) {
                        isTrue = false;
                        break;
                    }
                    crcRes >>= 1;
                }
                break;
            }
            if (!isTrue) {
                continue;
            }
            break;
        }

        std::cout << "\n";
        resetData();

        return;
    }
}