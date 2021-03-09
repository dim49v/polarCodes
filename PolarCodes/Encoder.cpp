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

double Awgn(std::normal_distribution<double>& distributionN) {
    return distributionN(eng);
}

void AddFrozenBits(int* xInf, int* x, int n, const std::vector<int>& infIndexes, const std::vector<int>& frozenBits) {
    int u = 0;
    int y = 0;
    for (int i = 0; i < n; i++)
    {
        if (u < infIndexes.size() && i == infIndexes[u]) {
            x[i] = xInf[u];
            u++;
        }
        else {
            x[i] = frozenBits[y];
            y++;
        }
    }

    return;
}

void Transform(int n, int* c, double* res, std::normal_distribution<double>& distributionN) {
    for (int i = 0; i < n; i++) {
        res[i] = (c[i] == 0 ? 1 : -1) + Awgn(distributionN);
    }

    return;
}

std::vector<int> FrozenBits(int n, const std::vector<int>& infIndexes, const std::vector<int>& frozenBits) {
    std::vector<int> c(infIndexes.size() + frozenBits.size());
    int* xInf = new int[infIndexes.size()];
    int* x = new int[infIndexes.size() + frozenBits.size()];
    for (int i = 0; i < infIndexes.size(); i++) {
        xInf[i] = -1;
    }
    AddFrozenBits(xInf, x, n, infIndexes, frozenBits);
    for (int i = 0; i < c.size(); i++) {
        c[i] = x[i];
    }
    return c;
}

void EncodeNode(int n, int m, int** encTree) {
    if (2 << m == n) {
        encTree[m][0] ^= encTree[m][1];
        return ;
    }
    int nodeSize = n >> (m + 1);
    ReverseShuffle(nodeSize * 2, encTree[m]);
    for (int i = 0; i < nodeSize; i++) {
        encTree[m + 1][i] = encTree[m][i];
    }
    EncodeNode(n, m + 1, encTree);
    for (int i = 0; i < nodeSize; i++) {
        encTree[m][i] = encTree[m + 1][i];
        encTree[m + 1][i] = encTree[m][i + nodeSize];
    }
    EncodeNode(n, m + 1, encTree);
    for (int i = 0; i < nodeSize; i++) {
        encTree[m][i] ^= encTree[m + 1][i];
        encTree[m][i + nodeSize] = encTree[m + 1][i];
    }
    return;
}
void Encode(int n, int* c, int** encTree) {
    for (int i = 0; i < n; i++) {
        encTree[0][i] = c[i];
    }
    EncodeNode(n, 0, encTree);
    return;
}