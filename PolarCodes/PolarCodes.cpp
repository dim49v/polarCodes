#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <thread>
#include <mutex>
#include <string> 
#include <time.h>

#include "Resources.h"
#include "SCDecoderMinsum.h"
#include "SCDecoder.h"
#include "Encoder.h"

std::mutex mu;

std::vector<double> ComputeZ(int n) {
    if (n == 1) {
        return std::vector<double>(1, 0.5);
    }
    std::vector<double> c(n);
    std::vector<double> zn2 = ComputeZ(n / 2);
    for (int i = 0; i < n / 2; i++) {
        c[2 * i] = 2. * zn2[i] - zn2[i] * zn2[i];
        c[2 * i + 1] = zn2[i] * zn2[i];
    }
    return c;
}

std::vector<int> InfIndexes(int n, int k) {
    std::vector<int> c(k);
    std::vector<double> z = ComputeZ(n);
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

void myThread(double snr, int n, int k, std::vector<std::pair<double, double>> &res, int decoder, int loop) {
    std::uniform_int_distribution<int> distributionU(0, 1);
    double sigma = sqrt(n / (2. * k * pow(10, snr / 10)));
    std::normal_distribution<double> distributionN = std::normal_distribution<double>(0, sigma);
    std::vector<int> infIndexes = InfIndexes(n, k);
    std::vector<int> frozenBits(n - k);
    std::vector<int> shuffledFrozenBits = FrozenBits(n, infIndexes, frozenBits);
    int er = 0;
    for (int i = 0; i < loop; i++) {
        std::vector<int> x(k);
        for (int u = 0; u < k; u++) {
            x[u] = distributionU(eng);
        }
        x = AddFrozenBits(x, infIndexes, frozenBits);
        std::vector<int> en = Encode(x);
        std::vector<int> dec;
        switch (decoder)
        {
            case 1:
                dec = SCDecoder::Decode(Transform(en, distributionN), shuffledFrozenBits, sigma);
                break;
            case 2:
            default:
                dec = SCDecoderMinsum::Decode(Transform(en, distributionN), shuffledFrozenBits);
                break;
        }
        for (int u = 0; u < n; u++) {
            if (x[u] != dec[u]) {
                er++;
                break;
            }
        }
    }
    mu.lock();
    res.push_back(std::pair<double, double>(snr,(double)er / loop));
    mu.unlock();
    std::cout << snr << '\n';
}

int main()
{
    int n, k, decoder, loop, nth;
    
    std::cout << "Input n, k, number of iteration\n";
    std::cin >> n >> k >> loop;
    std::cout << "Select decoder: SC (1), SCMinsum (2)\n";
    std::cin >> decoder;
    std::cout << "Input number of SNRb\n";
    std::cin >> nth;
    std::vector<double> snr(nth);
    std::cout << "Input "<< nth << " SNRbs\n";
    for (int i = 0; i < nth; i++) {
        std::cin >> snr[i];
    }
    std::cout << "Running...\n";
    std::vector<std::pair<double, double>> res(0);
    std::vector<std::thread> threads;
    for (int i = 0; i < nth; i++) {
        threads.push_back(std::thread(myThread, snr[i], n, k, std::ref(res), decoder, loop));
    }

    for (int i = 0; i < threads.size(); i++) {
        threads[i].join();
    }
    std::sort(res.begin(), res.end());
    std::ofstream fout;
    fout.open("SC_"+std::to_string(n) + "_" + std::to_string(k) + "_" + std::to_string(time(0)) + ".txt");
    for (int i = 0; i < res.size(); i++) {
        fout << res[i].first << '\t' << res[i].second << '\n';
    }
}
