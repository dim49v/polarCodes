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
    int* inactivePathIndices;
    int inactivePathIndicesEnd = 0;
    bool* activePath;
    float** arrayPointer_P;
    int*** arrayPointer_C;
    int* pathIndexToArrayIndex;
    int* inactiveArrayIndices;
    int* inactiveArrayIndicesEnd;
    int* arrayReferenceCount;
    int L_, L2_;
    int n_;
    int m_;
    int k_;
    int crc_;
    float* probForks;
    bool* constForks;
    float* metric;
    float* sortedMetric;
    __m256 avxMaskAbs, avxMaskSign;

    float Decodef(float r1, float r2) {
        return std::min(abs(r1), abs(r2)) * (r1 * r2 < 0 ? -1.f : 1.f);
    }
    __m256 f_avx_float(__m256 r1, __m256 r2) {
        __m256 mr1 = _mm256_and_ps(r1, avxMaskAbs);
        __m256 mr2 = _mm256_and_ps(r2, avxMaskAbs);
        __m256 mm = _mm256_min_ps(mr1, mr2);
        __m256 ss = _mm256_and_ps(_mm256_xor_ps(r1, r2), avxMaskSign);
        return _mm256_or_ps(mm, ss);
    }

    float Decodeg(float r1, float r2, int b) {
        return r2 + r1 * (1 - b * 2);
    }
    __m256 g_avx_float(__m256 r1, __m256 r2, __m256i b) {
        __m256 bb = _mm256_castsi256_ps(_mm256_slli_epi32(b, 31));
        __m256 rr1 = _mm256_xor_ps(r1, bb);
        return _mm256_add_ps(r2, rr1);
    }

    void InitializeDataStructures(int n, int L, int k, int crc) {
        L_ = L;
        L2_ = 2 * L_;
        n_ = n;
        m_ = std::log2(n);
        k_ = k;
        crc_ = crc;
        inactivePathIndices = new int[L_];
        activePath = new bool[L_];
        arrayReferenceCount = new int [(m_ + 1) * L_];
        arrayPointer_P = new float* [(m_ + 1) * L_];
        arrayPointer_C = new int** [(m_ + 1) * L_];
        pathIndexToArrayIndex = new int [(m_ + 1) * L_];
        inactiveArrayIndices = new int[(m_ + 1) * L_];
        inactiveArrayIndicesEnd = new int[m_ + 1];
        probForks = new float [L_ * 2];
        constForks = new bool [L_ * 2];
        metric = new float[L_];
        sortedMetric = new float[L2_];
        unsigned int size = n;
        for (int i = 0; i < m_ + 1; i++) {
            inactiveArrayIndicesEnd[i] = 0;
            for (int u = 0; u < L_; u++) {
                arrayPointer_P[i * L_ + u] = new float[size > 8 ? size : 8];
                arrayPointer_C[i * L_ + u] = new int* [2];
                for (int j = 0; j < 2; j++) {
                    arrayPointer_C[i * L_ + u][j] = new int[size > 8 ? size : 8];
                }
                arrayReferenceCount[i * L_ + u] = 0;
                inactiveArrayIndices[i * L_ + inactiveArrayIndicesEnd[i]++] = u;
            }
            size >>= 1;
        }
        for (int i = 0; i < L_; i++) {
            activePath[i] = false;
            inactivePathIndices[inactivePathIndicesEnd++] = i;
        }
        avxMaskAbs = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
        avxMaskSign = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));
    }
    void resetData() {
        inactivePathIndicesEnd = 0;
        for (int i = 0; i <= m_; i++) {
            inactiveArrayIndicesEnd[i] = 0;
            for (int u = 0; u < L_; u++) {
                arrayReferenceCount[i * L_ + u] = 0;
                inactiveArrayIndices[i * L_ + inactiveArrayIndicesEnd[i]++] = u;
            }
        }
        for (int i = 0; i < L_; i++) {
            activePath[i] = false;
            inactivePathIndices[inactivePathIndicesEnd++] = i;
        }
    }
    int assignInitialPath() {
        int l = inactivePathIndices[--inactivePathIndicesEnd];
        activePath[l] = true;
        int s;
        for (int i = 0; i < m_ + 1; i++) {
            s = inactiveArrayIndices[i * L_ + (--inactiveArrayIndicesEnd[i])];
            pathIndexToArrayIndex[i * L_ + l] = s;
            arrayReferenceCount[i * L_ + s] = 1;
        }
        metric[l] = 0;
        return l;
    }
    int clonePath(int l) {
        int ll = inactivePathIndices[--inactivePathIndicesEnd];
        activePath[ll] = true;
        int s;
        for (int i = 0; i < m_ + 1; i++) {
            s = pathIndexToArrayIndex[i * L_ + l];
            pathIndexToArrayIndex[i * L_ + ll] = s;
            arrayReferenceCount[i * L_ + s]++;
        }   
        metric[ll] = metric[l];
        return ll;
    }
    void killPath(int l) {
        activePath[l] = false;
        inactivePathIndices[inactivePathIndicesEnd++] = l;
        int s;
        for (int i = 0; i < m_ + 1; i++) {
            s = pathIndexToArrayIndex[i * L_ + l];
            arrayReferenceCount[i * L_ + s]--;
            if (arrayReferenceCount[i * L_ + s] == 0) {
                inactiveArrayIndices[i * L_ + inactiveArrayIndicesEnd[i]++] = s;
            }
        }
    }
    int copyContent(int m, int l) {
        int size = n_ >> m;
        int ss;
        int s = pathIndexToArrayIndex[m * L_ + l];
        if (arrayReferenceCount[m * L_ + s] == 1) {
            ss = s;
        }
        else {
            ss = inactiveArrayIndices[m * L_ + (--inactiveArrayIndicesEnd[m])];
            std::copy(arrayPointer_P[m * L_ + s ], arrayPointer_P[m * L_ + s] + size, arrayPointer_P[m * L_ + ss]);
            std::copy(arrayPointer_C[m * L_ + s][0], arrayPointer_C[m * L_ + s][0] + size, arrayPointer_C[m * L_ + ss][0]);
            std::copy(arrayPointer_C[m * L_ + s][1], arrayPointer_C[m * L_ + s][1] + size, arrayPointer_C[m * L_ + ss][1]);
            arrayReferenceCount[m * L_ + s]--;
            arrayReferenceCount[m * L_ + ss] = 1;
            pathIndexToArrayIndex[m * L_ + l] = ss;
        }
        
        return ss;
    }
    float* getArrayPointer_P(int m, int l) {
        int ss = copyContent(m, l);
        return arrayPointer_P[m * L_ + ss];
    }
    int** getArrayPointer_C(int m, int l) {
        int ss = copyContent(m, l);
        return arrayPointer_C[m * L_ + ss];
    }

    void recursivelyCalcP(int m, int phase) {
        if (m == 0) {
            return;
        }
        int w = phase / 2;
        if (phase % 2 == 0) {
            recursivelyCalcP(m - 1, w);
        }
        float* p1;
        float* p2;
        int** c1;
        __m256 r1, r2, res1;
        __m128 rr;
        __m256i b;
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
                for (int u = 0; u < size; u += 8) {
                    r1 = _mm256_loadu_ps(p2 + u);
                    r2 = _mm256_loadu_ps(p2 + size + u);
                    res1 = f_avx_float(r1, r2);
                    _mm256_storeu_ps(p1 + u, res1);
                }
            } 
            else {
                for (int u = 0; u < size; u += 8) {
                    r1 = _mm256_loadu_ps(p2 + u);
                    r2 = _mm256_loadu_ps(p2 + size + u); 
                    b = _mm256_loadu_epi32(c1[0] + u);
                    res1 = g_avx_float(r1, r2, b);
                    _mm256_storeu_ps(p1 + u, res1);
                }
            }
            //for (int u = 0; u < size; u++) {
            //    if (phase % 2 == 0) {
            //        p1[u] = Decodef(p2[u], p2[size + u]);
            //    }
            //    else {
            //        p1[u] = Decodeg(p2[u], p2[size + u], c1[0][u]);
            //    }
            //}
        }
    }
    void recursivelyUpdateC(int m, int phase) {
        int size = n_ >> m;
        int w = phase / 2;
        int** c1;
        int** c2;
        __m256i cc1, cc2;
        for (int i = 0; i < L_; i++) {
            if (!activePath[i]) {
                continue;
            }
            c1 = getArrayPointer_C(m, i);
            c2 = getArrayPointer_C(m - 1, i);
            for (int u = 0; u < size; u += 8) {
                cc1 = _mm256_loadu_epi32(c1[0] + u);
                cc2 = _mm256_loadu_epi32(c1[1] + u);
                _mm256_storeu_epi32(c2[w % 2] + u, _mm256_xor_si256(cc1, cc2));
            }
            std::copy(c1[1], c1[1] + size, c2[w % 2] + size);
            //for (int u = 0; u < size; u++) {
            //    c2[w % 2][u] = c1[0][u] ^ c1[1][u];
            //    c2[w % 2][u + size] = c1[1][u];
            //}
        }
        if (w % 2 == 1) {
            recursivelyUpdateC(m - 1, w);
        }
    }
    float additionPM(float llr, char u) {
        if (u == 0 && llr < 0 || u == 1 && llr > 0) {
            return std::abs(llr);
        }
        return 0;
    }

    void continuePaths_UnfrozenBit(int phase) {
        float* Pm;
        int** cm;
        if (phase == n_ - 1) {
            phase = phase;
        }
        int activeCount = 0;
        for (int i = 0; i < L_; i++) {
            if (activePath[i]) {
                Pm = getArrayPointer_P(m_, i);
                probForks[i] = metric[i] + additionPM(Pm[0], 0);
                probForks[i + L_] = metric[i] + additionPM(Pm[0], 1);
                sortedMetric[2 * i] = probForks[i];
                sortedMetric[2 * i + 1] = probForks[i + L_];
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
                    metric[i] = probForks[i];
                    metric[ii] = probForks[i + L_];
                }
            }
        }
        else {
            std::nth_element(sortedMetric, sortedMetric + L_, sortedMetric + L2_);
            float med = sortedMetric[L_];
            int activeCount = 0;
            for (int i = 0; i < L_; i++) {
                if (probForks[i] < med && activeCount < L_) {
                    constForks[i] = true;
                    activeCount++;
                }
                else {
                    constForks[i] = false;
                }
                if (probForks[i + L_] < med && activeCount < L_) {
                    constForks[i + L_] = true;
                    activeCount++;
                }
                else {
                    constForks[i + L_] = false;
                }
            }
            for (int i = 0; i < L_ && activeCount < L_; i++) {
                if (probForks[i] == med) {
                    constForks[i] = true;
                    activeCount++;
                }
                if (probForks[i + L_] == med && activeCount < L_) {
                    constForks[i + L_] = true;
                    activeCount++;
                }
            }
            for (int i = 0; i < L_; i++) {
                if (!constForks[i] && !constForks[i + L_]) {
                    killPath(i);
                }
            }
            for (int i = 0; i < L_; i++) {
                if (!activePath[i] || (!constForks[i] && !constForks[i + L_])) {
                    continue;
                }
                cm = getArrayPointer_C(m_, i);
                if (constForks[i] && constForks[i + L_]) {
                    cm[phase % 2][0] = 0;
                    int ii = clonePath(i);
                    cm = getArrayPointer_C(m_, ii);
                    cm[phase % 2][0] = 1;
                    metric[i] = probForks[i];
                    metric[ii] = probForks[i + L_];
                }
                else if (constForks[i]) {
                    cm[phase % 2][0] = 0;
                    metric[i] = probForks[i];
                }
                else {
                    cm[phase % 2][0] = 1;
                    metric[i] = probForks[i + L_];
                }
            }
        }
    }
    void Decode(int n, int L, double* c, int** decTree2, int* sol, const std::vector<int>& frozenBits, int qFraction) {
        int l = assignInitialPath();
        float* pm = getArrayPointer_P(0, l);
        int** cm;
        int fraction = 2 << qFraction;
        for (int i = 0; i < n; i++) {
            pm[i] = c[i];
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
            Encode(n_, cm[0], decTree2, false);
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