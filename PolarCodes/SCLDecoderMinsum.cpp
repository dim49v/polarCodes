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
    char**** arrayPointer_C;
    int** pathIndexToArrayIndex;
    std::vector<int>* inactiveArrayIndices;
    int** arrayReferenceCount;
    int L_;
    int n_;
    int m_;
    int k_;
    int crc_;
    int** probForks;
    bool** constForks;
    int* metric;
    int* sortedMetric;
    char avxMaskArr[] {0xff,0x00,0xff,0x00,0xff,0x00,0xff,0x00,0xff,0x00,0xff,0x00,0xff,0x00,0xff,0x00,0xff,0x00,0xff,0x00,0xff,0x00,0xff,0x00,0xff,0x00,0xff,0x00,0xff,0x00,0xff,0x00, 0xff };
    //char* avxMaskArr1 = new char[32]{ 0, 0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff };
    //char* avxMaskArr2 = new char[32]{ 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
    //char avxMaskArr1[32]{ 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff,0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff };
    //char avxMaskArr2[32]{ 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff,0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00 };
    //char avxMaskArr1[32]{ 0x0f, 0x0d, 0x0b, 0x09, 0x07, 0x05, 0x03, 0x01, 0x0e, 0x0c, 0x0a, 0x08, 0x06, 0x04, 0x03, 0x00, 0x0f, 0x0d, 0x0b, 0x09, 0x07, 0x05, 0x03, 0x01, 0x0e, 0x0c, 0x0a, 0x08, 0x06, 0x04, 0x03, 0x00 };
    char avxMaskArr1[32]{ 0x00, 0x02, 0x04, 0x06, 0x08, 0x0a, 0x0c, 0x0e, 0x01, 0x03, 0x05, 0x07, 0x09, 0x0b, 0x0d, 0x0f, 0x00, 0x02, 0x04, 0x06, 0x08, 0x0a, 0x0c, 0x0e, 0x01, 0x03, 0x05, 0x07, 0x09, 0x0b, 0x0d, 0x0f };
    char avxMaskArr2[32]{ 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff,0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00 };
    __m256i avxMask1, avxMask2, reserve;// = 0xaaaaaaaa, avxMask2 = 0x55555555;
    const __m256i maskF1 = _mm256_set1_epi8(0x40);
    const __m256i maskG1 = _mm256_set1_epi8(0x00);
    const __m256i maskG2 = _mm256_set1_epi8(0x0f);
    const __m256i maskG3 = _mm256_set1_epi8(0x81);
    const __m256i maskG4 = _mm256_set1_epi8(0x01);
    double Decodef(double r1, double r2) {
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
        __m256i mr1 = _mm256_abs_epi8(r1);
        __m256i mr2 = _mm256_abs_epi8(r2);
        __m256i mm = _mm256_min_epi8(mr1, mr2);
        __m256i ss = _mm256_xor_si256(r1, r2);
        __m256i sr = _mm256_or_si256(ss, maskF1);
        return _mm256_sign_epi8(mm, sr);
    }
    double Decodeg(double r1, double r2, int b) {
        return r2 + r1 * (1 - b * 2);
    }  
    char Decodeg_char(char r1, char r2, char b) {
        int res = r2 + r1 * (1 - b * 2);
        return abs(res) > 127 ? (res > 0 ? 127 : -127) : res;
    }
    __m256i g_avx(__m256i r1, __m256i r2, __m256i b) {
        __m256i bb = _mm256_cmpgt_epi8(b, maskG1);        
        __m256i bbb = _mm256_xor_si256(bb, maskG2);
        __m256i mr1 = _mm256_sign_epi8(r1, bbb);
        __m256i res = _mm256_adds_epi8(mr1, r2);
        __m256i mr2 = _mm256_cmpgt_epi8(maskG3, res);
        __m256i mr3 = _mm256_and_si256(mr2, maskG4);
        __m256i mr4 = _mm256_adds_epi8(res, mr3);
        return mr4;
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
        arrayPointer_C = new char*** [m_ + 1];
        pathIndexToArrayIndex = new int* [m_ + 1];
        inactiveArrayIndices = new std::vector<int>[m_ + 1];
        arrayReferenceCount = new int* [m_ + 1];
        probForks = new int* [L_];
        constForks = new bool* [L_];
        metric = new int[L_];
        sortedMetric = new int[2 * L_];
        unsigned int size = n;
        for (int i = 0; i < m_ + 1; i++) {
            pathIndexToArrayIndex[i] = new int[L_];
            inactiveArrayIndices[i].reserve(L_);
            arrayReferenceCount[i] = new int[L_];
            arrayPointer_P[i] = new char* [L_];
            arrayPointer_C[i] = new char** [L_];
            for (int u = 0; u < L_; u++) {
                arrayPointer_P[i][u] = new char[size > 32 ? size : 32];
                arrayPointer_C[i][u] = new char* [2];
                for (int j = 0; j < 2; j++) {
                    arrayPointer_C[i][u][j] = new char[size > 32 ? size : 32];
                }
                arrayReferenceCount[i][u] = 0;
                inactiveArrayIndices[i].push_back(u);
            }
            size >>= 1;
        }
        for (int i = 0; i < L_; i++) {
            activePath[i] = false;
            inactivePathIndices.push_back(i);
            probForks[i] = new int[2];
            constForks[i] = new bool[2];
        }

        avxMask1 = _mm256_loadu_epi8(avxMaskArr1);
        avxMask2 = _mm256_loadu_epi8(avxMaskArr2);
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
                arrayPointer_C[m][ss][0][i] = arrayPointer_C[m][s][0][i];
                arrayPointer_C[m][ss][1][i] = arrayPointer_C[m][s][1][i];
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
    char** getArrayPointer_C(int m, int l) {
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
        char** c1;
        __m256i r1, r2, b, res1;
        __m128i rr;
        for (int i = 0; i < L_; i++) {
            if (!activePath[i]) {
                continue;
            } 
            int size = n_ >> m;
            int size2 = size >> 1;
            p1 = getArrayPointer_P(m, i);
            p2 = getArrayPointer_P(m - 1, i);
            c1 = getArrayPointer_C(m, i);
            if (phase % 2 == 0) {
                for (int u = 0; u < size; u += 32) {
                    r1 = _mm256_loadu_epi8(p2 + u);
                    r2 = _mm256_loadu_epi8(p2 + size + u);
                    res1 = f_avx(r1, r2);
                    r1 = _mm256_shuffle_epi8(res1, avxMask1);
                    res1 = _mm256_permute4x64_epi64(r1, 0xf8); // [1, 2, 3, 4] -> [1, 3 (, 4, 4)]
                    rr = _mm256_castsi256_si128(res1);
                    _mm_storeu_epi8(p1 + (u >> 1), rr);
                    if (size2 > 0) {
                        res1 = _mm256_permute4x64_epi64(r1, 0xfd); // [1, 2, 3, 4] -> [2, 4 (, 4, 4)]
                        rr = _mm256_castsi256_si128(res1);
                        _mm_storeu_epi8(p1 + (u >> 1) + size2, rr);
                    }
                }
            } 
            else {
                for (int u = 0; u < size; u += 32) {
                    r1 = _mm256_loadu_epi8(p2 + u);
                    r2 = _mm256_loadu_epi8(p2 + size + u);
                    b = _mm256_loadu_epi8(c1[0] + u);
                    res1 = g_avx(r1, r2, b);
                    r1 = _mm256_shuffle_epi8(res1, avxMask1);
                    res1 = _mm256_permute4x64_epi64(r1, 0xf8); // [1, 2, 3, 4] -> [1, 3 (, 4, 4)]
                    rr = _mm256_castsi256_si128(res1);
                    _mm_storeu_epi8(p1 + (u >> 1), rr);
                    if (size2 > 0) {
                        res1 = _mm256_permute4x64_epi64(r1, 0xfd); // [1, 2, 3, 4] -> [2, 4 (, 4, 4)]
                        rr = _mm256_castsi256_si128(res1);
                        _mm_storeu_epi8(p1 + (u >> 1) + size2, rr);
                    }
                }
            }
            //for (int u = 0; u < size; u++) {
            //    if (phase % 2 == 0) {
            //        p1[(u % 2 * size2) + u / 2] = Decodef_char(p2[u], p2[size + u]);
            //    }
            //    else {
            //        p1[(u % 2 * size2) + u / 2] = Decodeg_char(p2[u], p2[size + u], c1[0][u]);
            //    }
            //}
        }
    }
    void recursivelyUpdateC(int m, int phase) {
        unsigned int size = n_ >> m;
        int w = phase / 2;
        char** c1;
        char** c2;
        for (int i = 0; i < L_; i++) {
            if (!activePath[i]) {
                continue;
            }
            c1 = getArrayPointer_C(m, i);
            c2 = getArrayPointer_C(m - 1, i);
            for (int u = 0; u < size; u++) {
                c2[w % 2][2 * u] = c1[0][u] ^ c1[1][u];
                c2[w % 2][2 * u + 1] = c1[1][u];
            }
        }
        if (w % 2 == 1) {
            recursivelyUpdateC(m - 1, w);
        }
    }
    char additionPM(char llr, char u) {
        char x = -(1 - 2 * u) * llr;
        return x >= 0 ? x : 0;
    }

    bool sortCMP(int a, int b) {
        return (probForks[a / 2][a % 2] == probForks[b / 2][b % 2]) 
            ? metric[a] < metric[b] 
            : probForks[a / 2][a % 2] < probForks[b / 2][b % 2];
    }
    void continuePaths_UnfrozenBit(int phase) {
        char* Pm;
        char** cm;
        if (phase == n_ - 1) {
            phase = phase;
        }
        int activeCount = 0;
        for (int i = 0; i < L_; i++) {
            if (activePath[i]) {
                Pm = getArrayPointer_P(m_, i);
                if (Pm[0] == 0) {
                    Pm = Pm;
                }
                probForks[i][0] = metric[i] + additionPM(Pm[0], 0);
                probForks[i][1] = metric[i] + additionPM(Pm[0], 1);
                sortedMetric[2 * i] = 2 * i;
                sortedMetric[2 * i + 1] = 2 * i + 1;
                activeCount++;
            }
        }
        if (activeCount < L_) {
            for (int i = 0; i < L_; i++) {
                if (activePath[i]) {       
                    cm = getArrayPointer_C(m_, i);
                    cm[phase % 2][0] = 0;
                    int ii = clonePath(i);
                    cm = getArrayPointer_C(m_, ii);
                    cm[phase % 2][0] = 1;              
                    metric[i] = probForks[i][0];
                    metric[ii] = probForks[i][1];
                }
            }
        }
        else {
            std::sort(sortedMetric, sortedMetric + 2 * L_, sortCMP);
            double med = (double)(sortedMetric[L_ - 1] + sortedMetric[L_]) / 2;
            for (int i = 0; i < L_; i++) {
                constForks[sortedMetric[i]/2][sortedMetric[i] % 2] = true;
                constForks[sortedMetric[L_ + i] / 2][sortedMetric[L_ + i] % 2] = false;
            }
            for (int i = 0; i < L_; i++) {
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
                    cm[phase % 2][0] = 0;
                    int ii = clonePath(i);
                    cm = getArrayPointer_C(m_, ii);
                    cm[phase % 2][0] = 1;
                    metric[i] = probForks[i][0];
                    metric[ii] = probForks[i][1];
                }
                else if (constForks[i][0]) {
                    cm[phase % 2][0] = 0;
                    metric[i] = probForks[i][0];
                }
                else {
                    cm[phase % 2][0] = 1;
                    metric[i] = probForks[i][1];
                }
            }
        }
    }
    void Decode(int n, int L, double* c, int** decTree2, int* sol, const std::vector<int>& frozenBits, int qFraction) {
        int l = assignInitialPath();
        char* pm = getArrayPointer_P(0, l);
        char** cm;
        int fraction = 2 << qFraction;
        int fillN = n_ >> 1;
        for (int i = 0; i < fillN; i++) {
            pm[i] = round(c[2 * i] * fraction);
            pm[fillN + i] = round(c[2 * i + 1] * fraction);
        }          
        for (int i = 0; i < n_; i++) {
            recursivelyCalcP(m_, i);
            if (frozenBits[i] != -1) {
                for (int u = 0; u < L_; u++) {
                    if (!activePath[u]) {
                        continue;
                    }
                    cm = getArrayPointer_C(m_, u);
                    pm = getArrayPointer_P(m_, u);
                    cm[i % 2][0] = frozenBits[i];
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
        bool isTrue;
        for (int i = 0; i < L_; i++) { 
            cm = getArrayPointer_C(0, i);
            for (int u = 0; u < n_; u++) {
                sol[u] = cm[0][u]; 
            }
            Encode(n_, sol, decTree2);
            int j = 0;
            for (int u = 0; u < n_; u++) {
                if (frozenBits[u] == -1) {      
                    sol[j++] = decTree2[0][u];
                }
            }
            isTrue = true;
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
        resetData();

        return;
    }
}