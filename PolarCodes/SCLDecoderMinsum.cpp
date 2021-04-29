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
    int** arrayPointer_C;
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
    int* dethCW;
    int* dethP;
    int* sizeN;

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
        arrayPointer_C = new int* [(m_ + 1) * L_];
        pathIndexToArrayIndex = new int [(m_ + 1) * L_];
        inactiveArrayIndices = new int[(m_ + 1) * L_];
        inactiveArrayIndicesEnd = new int[m_ + 1];
        probForks = new float [L_ * 2];
        constForks = new bool [L_ * 2];
        metric = new float[L_];
        sortedMetric = new float[L2_];
        sizeN = new int[m_ + 1];
        unsigned int size = n;
        for (int i = 0; i < m_ + 1; i++) {
            sizeN[i] = n >> i;
            inactiveArrayIndicesEnd[i] = 0;
            for (int u = 0; u < L_; u++) {
                arrayPointer_P[i * L_ + u] = new float[size > 8 ? size : 8];
                arrayPointer_C[i * L_ + u] = new int[size > 8 ? size : 8];
                arrayReferenceCount[i * L_ + u] = 0;
                inactiveArrayIndices[i * L_ + inactiveArrayIndicesEnd[i]++] = u;
                pathIndexToArrayIndex[i * L_ + u] = -1;
            }
            size >>= 1;
        }
        for (int i = 0; i < L_; i++) {
            activePath[i] = false;
            inactivePathIndices[inactivePathIndicesEnd++] = i;
        }
        dethCW = new int[n];
        dethP = new int[n * m_];
        bool sw = false;
        int d, dd, ff;
        for (int i = 0; i < n; i++) {
            dd = 1 << m_;
            ff = i + 1;
            for (d = m_; ff % dd != 0; d--) {
                dd >>= 1;
            };
            dethCW[i] = d;
            for (int u = 0; u < m_; u++) {
                dd = 1 << u;
                for (d = u; i % dd != 0; d--) {
                    dd >>= 1;
                };
                dethP[u * n + i] = d;
            }
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

    int allocate(int m, int l, int L_l) {
        int t = inactiveArrayIndices[L_l + (--inactiveArrayIndicesEnd[l])];
        arrayReferenceCount[L_l + t] = 1;
        pathIndexToArrayIndex[L_l + m] = t;
        return t;
    }

    float* getArrayPointerPW(int m, int l) {
        int L_l = L_ * l;;
        int p = pathIndexToArrayIndex[L_l + m];
        if (p == -1) {  
            p = allocate(m, l, L_l);
        }
        else {
            if (arrayReferenceCount[L_l + p] > 1) {
                arrayReferenceCount[L_l + p]--;
                p = allocate(m, l, L_l);
            }
        }
        return arrayPointer_P[L_l + p];
    }  

    float* getArrayPointerPR(int m, int l) {
        int L_l = L_ * l;
        return arrayPointer_P[L_l + pathIndexToArrayIndex[L_l + m]];
    }

    int* getArrayPointerCR(int m, int l) {
        int L_l = L_ * l;
        return arrayPointer_C[L_l + pathIndexToArrayIndex[L_l + m]];
    }

    int* getArrayPointerCW(int m, int l, int f) {
        int d = dethCW[f];
        int dd = l - d;
        int L_dd = L_ * dd;
        int p = pathIndexToArrayIndex[L_dd + m];
        if (p == -1) {
            p = allocate(m, dd, L_dd);
        }
        else {
            if (arrayReferenceCount[L_dd + p] > 1) {
                arrayReferenceCount[L_dd + p]--;
                p = allocate(m, dd, L_dd);
            }
        }
        int* c;
        if (f % 2 == 0) {
            c = arrayPointer_C[l * L_ + p];
        } else {
            c = arrayPointer_C[L_dd + p] + ((1 << d) - 1);
        }
        return c;
    }

    void iterativelyCalcP(int m, int l, int f) {
        int d = dethP[(l - 1) * n_ + f];
        int ll = l - d;
        float* p = getArrayPointerPR(m, ll - 1);
        int n = sizeN[ll];
        float* pn = p + n;
        float* pp;
        __m256 r1, r2, res1;
        __m128 rr;
        __m256i b;
        if (f / (1 << d) % 2 == 1) {
            int* c = getArrayPointerCR(m, ll);
            pp = getArrayPointerPW(m, ll);
            //if (n >= 8) {
                //for (int u = 0; u < n; u += 8) {
                //    r1 = _mm256_loadu_ps(p + u);
                //    r2 = _mm256_loadu_ps(pn + u);
                //    b = _mm256_loadu_epi32(c + u);
                //    res1 = g_avx_float(r1, r2, b);
                //    _mm256_storeu_ps(pp + u, res1);
                //}
            //}
            //else {
                for (int i = 0; i < n; i++) {
                    pp[i] = Decodeg(p[i], p[i + n], c[i]);
                }
            //}
            p = pp;
            ll++;
            n >>= 1;
        }
        while (ll <= l) {
            pn = p + n;
            pp = getArrayPointerPW(m, ll);
            //if (n >= 8) {
                //for (int u = 0; u < n; u += 8) {
                //    r1 = _mm256_loadu_ps(p + u);
                //    r2 = _mm256_loadu_ps(pn + u);
                //    res1 = f_avx_float(r1, r2);
                //    _mm256_storeu_ps(pp + u, res1);
                //}
            //}
            //else {
                for (int i = 0; i < n; i++) {
                    pp[i] = Decodef(p[i], p[i + n]);
                }
            //}
            p = pp;
            ll++;
            n >>= 1;
        }
    }

    void iterativelyUpdateC(int m, int l, int f) {
        int d = dethCW[f];
        int* ccc = getArrayPointerCW(m, l - d, 0);
        int n = sizeN[l];
        ccc += n * ((1 << d) - 2);
        int* cc = ccc + n; 
        int ll = l - d;
        int* c;
        __m256i cc1, cc2;
        while (l > ll) {
            c = getArrayPointerCR(m, l);
            //if (n >= 8) {
            //    for (int u = 0; u < n; u += 8) {
            //        cc1 = _mm256_loadu_epi32(c + u);
            //        cc2 = _mm256_loadu_epi32(cc + u);
            //        _mm256_storeu_epi32(ccc + u, _mm256_xor_si256(cc1, cc2));
            //    }
            //}
            //else {
                for (int i = 0; i < n; i++) {
                    ccc[i] = c[i] ^ cc[i];
                }
            //}
            n <<= 1;
            cc = ccc;
            ccc -= n;
            l--;
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
        int* cm;
        int activeCount = 0;
        for (int i = 0; i < L_; i++) {
            if (activePath[i]) {
                Pm = getArrayPointerPR(i, m_);
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
                    cm = getArrayPointerCW(i, m_, phase);
                    cm[0] = 0;
                    int ii = clonePath(i);
                    cm = getArrayPointerCW(ii, m_, phase);
                    cm[0] = 1;              
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
                cm = getArrayPointerCW(i, m_, phase);
                if (constForks[i] && constForks[i + L_]) {
                    cm[0] = 0;
                    int ii = clonePath(i);
                    cm = getArrayPointerCW(ii, m_, phase);
                    cm[0] = 1;
                    metric[i] = probForks[i];
                    metric[ii] = probForks[i + L_];
                }
                else if (constForks[i]) {
                    cm[0] = 0;
                    metric[i] = probForks[i];
                }
                else {
                    cm[0] = 1;
                    metric[i] = probForks[i + L_];
                }
            }
        }
    }
    void Decode(int n, int L, double* c, int** decTree2, int* sol, const std::vector<int>& frozenBits, int qFraction) {
        int l = assignInitialPath();
        float* pm = getArrayPointerPW(l, 0);
        int* cm;
        int fraction = 2 << qFraction;
        for (int i = 0; i < n; i++) {
            pm[i] = c[i];
        }
        for (int i = 0; i < n_; i++) {  
            for (int u = 0; u < L_; u++) {
                if (activePath[u]) {
                    iterativelyCalcP(u, m_, i);
                    if (frozenBits[i] != -1) {
                        cm = getArrayPointerCW(u, m_, i);
                        pm = getArrayPointerPR(u, m_);
                        cm[0] = frozenBits[i];
                        metric[u] += additionPM(pm[0], frozenBits[i]);
                    }
                }
            }
            if (frozenBits[i] == -1) {
                continuePaths_UnfrozenBit(i);
            }
            for (int u = 0; u < L_; u++) {
                if (activePath[u]) {
                    if (i > 0) {
                        iterativelyUpdateC(u, m_, i);
                    }
                }
            }
        } 
        bool isTrue;
        for (int i = 0; i < L_; i++) { 
            cm = getArrayPointerCR(i, 0);    
            Encode(n_, cm, decTree2, false);
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
            default:
                isTrue = false;
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