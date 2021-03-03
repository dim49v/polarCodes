#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <thread>
#include <mutex>
#include <string> 
#include <time.h>

#include "Resources.h"
#include "SCLDecoderMinsum.h"
#include "SCDecoderMinsum.h"
#include "SCDecoder.h"
#include "Encoder.h"

std::mutex mu;

std::vector<double> ComputeZBEC(int n) {
    if (n == 1) {
        return { 0.5 };
    }
    std::vector<double> c(n);
    std::vector<double> zn2 = ComputeZBEC(n / 2);
    for (int i = 0; i < n / 2; i++) {
        c[2 * i] = 2. * zn2[i] - zn2[i] * zn2[i];
        c[2 * i + 1] = zn2[i] * zn2[i];
    }
    return c;
}

double AWGNReliabilityApproximation(double x) {
    if (x > 12) {
        return 0.9861 * x - 2.3152;
    }
    else if (x > 3.5) {
        return x * (0.009005 * x + 0.7694) - 0.9507;
    }
    else if (x > 1) {
        return x * (0.062883 * x + 0.3678) - 0.1627;
    }
    else {
        return x * (0.2202 * x + 0.06448);
    }
}
std::vector<double> ComputeZAWGN(int n, double sigma) {
    if (n == 1) {
        return { 2. / (sigma * sigma) };
    }
    std::vector<double> c(n);
    std::vector<double> zn2 = ComputeZAWGN(n / 2, sigma);
    for (int i = 0; i < n / 2; i++) {
        c[2 * i] = AWGNReliabilityApproximation(zn2[i]);
        c[2 * i + 1] = 2. * zn2[i];
    }
    return c;
}


std::vector<int> InfIndexesAWGN(int n, int k, double sigma) {
    std::vector<int> c(k);
    std::vector<double> z = ComputeZAWGN(n, sigma);
    std::vector<std::pair<double, int>> p(n);
    for (int i = 0; i < n; i++) {
        p[i].first = z[i];
        p[i].second = i;
    }
    std::sort(p.begin(), p.end());
    for (int i = n - 1; i >= n - k; i--) {
        c[n - i - 1] = p[i].second;
    }
    std::sort(c.begin(), c.end());

    return c;
}


std::vector<int> InfIndexesBEC(int n, int k) {
    std::vector<int> c(k);
    std::vector<double> z = ComputeZBEC(n);
    std::vector<std::pair<double, int>> p(n);
    for (int i = 0; i < n; i++) {
        p[i].first = z[i];
        p[i].second = i;
    }
    std::sort(p.begin(), p.end());
    for (int i = 0; i < k; i++) {
        c[i] = p[i].second;
    }
    std::sort(c.begin(), c.end());

    return c;
}

void myThread(int n, int k, std::vector<std::pair<double, std::pair<double, double>>>& res, int decoder, int zComputer, int loop, std::vector<double>& snr, int L=1, int crc = 0) {
    std::uniform_int_distribution<int> distributionU(0, 1);
    double sigma;
    std::normal_distribution<double> distributionN;
    int m = (int)log2(n);
    int* xInf = new int[k + crc];
    int* x = new int[n];
    double* xTransf = new double[n];
    int* dec = new int[n];
    int** encTree = new int* [m];
    for (int i = m - 1; i >= 0; i--) {
        int s = n >> i;
        encTree[i] = new int[s];
    }
    double** decTree1 = new double* [m];
    for (int i = m - 1; i >= 0; i--) {
        int s = n >> i;
        decTree1[i] = new double[s];
    }
    int** decTree2 = new int* [m];
    for (int i = m - 1; i >= 0; i--) {
        int s = n >> i;
        decTree2[i] = new int[s];
    }
    int tSt, tAll = 0;
    if (decoder == 3) {
        SCLDecoderMinsum::InitializeDataStructures(n, L, k, crc);
    }
    for (int snri = 0; snri < snr.size(); snri++) {
        sigma = sqrt(n / (2. * k * pow(10, snr[snri] / 10)));
        distributionN = std::normal_distribution<double>(0, sigma);
        std::vector<int> infIndexes = zComputer == 1 ? InfIndexesBEC(n, k) : InfIndexesAWGN(n, k + crc, sigma);
        std::vector<int> frozenBits(n - k - crc);
        std::vector<int> shuffledFrozenBits = FrozenBits(n, infIndexes, frozenBits);
        int er = 0;
        for (int i = 0; i < loop; i++) {
            for (int u = 0; u < k; u++) {
                xInf[u] = distributionU(eng);
            }
            if (crc > 0) {
                unsigned short crcRes;
                switch (crc)
                {
                case 8:
                    crcRes = Crc8(xInf, k);
                    for (int u = 1; u <= 8; u++) {
                        xInf[k + crc - u] = crcRes & 0x01;
                        crcRes >>= 1;
                    }
                    break;
                case 16:
                    crcRes = Crc16(xInf, k);
                    for (int u = 1; u <= 16; u++) {
                        xInf[k + crc - u] = crcRes & 0x0001;
                        crcRes >>= 1;
                    }
                    break;
                }
            }

            AddFrozenBits(xInf, x, n, infIndexes, frozenBits);
            Encode(n, x, encTree);
            Transform(n, encTree[0], xTransf, distributionN);
            tSt = clock();
            switch (decoder)
            {
            case 1:
                SCDecoder::Decode(n, xTransf, decTree1, decTree2, dec, shuffledFrozenBits, sigma);
                for (int u = 0; u < n; u++) {
                    if (x[u] != dec[u]) {
                        er++;
                        break;
                    }
                }
                break;
            case 2:
                SCDecoderMinsum::Decode(n, xTransf, decTree1, decTree2, dec, shuffledFrozenBits); 
                for (int u = 0; u < n; u++) {
                    if (x[u] != dec[u]) {
                        er++;
                        break;
                    }
                }
                break;
            case 3:
                SCLDecoderMinsum::Decode(n, L, xTransf, decTree2, dec, shuffledFrozenBits);
                for (int u = 0; u < k; u++) {
                    if (xInf[u] != dec[u]) {
                        er++;
                        break;
                    }
                }
                break;
            }
            tAll += clock() - tSt;
            if ((i + 1) % (loop / 10) == 0) {
                std::cout << (loop - i) / (loop / 10) << " ";
            }
        }
        res.push_back(std::pair<double, std::pair<double, double>>(snr[snri], std::pair<double, double>((double)er / loop, tAll)));
        std::cout << '\n' << snr[snri] << " " << (double)er / loop << '\n';
    }
}

int main()
{
    int n, k, decoder, zComputer, L, crc, loop, nth;
    
    std::ifstream fin;
    fin.open("conf.txt");
    std::cout << "Input n, k, number of iteration\n";
    fin >> n >> k >> loop;
    std::cout << "Select decoder: SC (1), SCMinsum (2), SCLMinsum (3)\n";
    fin >> decoder;
    std::cout << "Select zComputer: BEC (1), AWGN (2)\n";
    fin >> zComputer;
    std::cout << "Input L\n";
    fin >> L;
    std::cout << "Input CRC\n";
    fin >> crc;
    std::cout << "Input number of SNRb\n";
    fin >> nth;
    std::vector<double> snr(nth);
    std::cout << "Input "<< nth << " SNRbs\n";
    for (int i = 0; i < nth; i++) {
        fin >> snr[i];
    }
    std::cout << "Running...\n";
    reverseShuffleMem = new int[n];
    std::vector<std::pair<double, std::pair<double, double>>> res(0);
    int startT = clock();
    myThread(n, k, res, decoder, zComputer, loop, snr, L, crc);
    int endT = clock();
    std::sort(res.begin(), res.end());
    std::ofstream fout;
    fout.open(
        "SC_" +
        std::to_string(n) +
        "_" + std::to_string(k) +
        "_L" + std::to_string(L) +
        "_CRC" + std::to_string(crc) +
        "_" + std::to_string(time(0)) + 
        ".txt");
    fout << endT - startT << '\n';
    for (int i = 0; i < res.size(); i++) {
        fout << res[i].first << '\t' << res[i].second.first << '\t' << res[i].second.second << '\n';
    }
}
