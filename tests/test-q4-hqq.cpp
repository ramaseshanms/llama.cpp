#include <cassert>
#include <cmath>
#include <vector>

extern "C" {
#include "../ggml/src/ggml-quants.h"
}

int main() {

    const int N = 64;

    std::vector<float> input(N);

    for (int i = 0; i < N; i++)
        input[i] = sin(i);

    int nb = (N + QK4_HQQ - 1) / QK4_HQQ;

    std::vector<block_q4_hqq> q(nb);
    std::vector<float> output(N);

    quantize_row_q4_hqq_ref(input.data(), q.data(), N);
    dequantize_row_q4_hqq(q.data(), output.data(), nb * QK4_HQQ);

    float mse = 0;

    for (int i = 0; i < N; i++) {
        float diff = input[i] - output[i];
        mse += diff * diff;
    }

    mse /= N;

    assert(mse < 0.05);
    printf("MSE = %f\n", mse);
    printf("Q4_HQQ test passed\n");

    return 0;
}