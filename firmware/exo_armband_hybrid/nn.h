#ifndef NN_H
#define NN_H

#include <Arduino.h>
#include "MODEL.h"

// Activation codes (must match export_teensy_headers.py)
#define ACT_RELU     0
#define ACT_SOFTMAX  1
#define ACT_SIGMOID  2
#define ACT_TANH     3
#define ACT_LINEAR   4

// Max layer width — must be >= the widest entry of MODEL_TOPOLOGY[].
// 현재 토폴로지 [132, 64, 64, 6]에서는 입력층 132가 가장 넓다.
//
// 이 값은 4채널 시절 토폴로지 [58, 64, 64, 4] 기준의 128이었고 4ch→8ch
// (58→132 features) 마이그레이션에서 갱신되지 않았다. 그 상태에서는
// predict()가 buf_a[128]에 132개를 써서 buf_b를 4개 덮고, 이어지는
// denseLayer()가 buf_b에 출력을 쓰면서 아직 읽어야 할 입력 특징
// 128~131(= CH4~CH7 TSD 에너지)을 파괴했다. 크래시 없이 추론 정확도만
// 조용히 떨어지는 종류의 버그였다.
#define NN_MAX_WIDTH 160

// 같은 실수가 재발하지 않게 컴파일 단계에서 막는다.
static_assert(NN_MAX_WIDTH >= MODEL_TOPOLOGY[0],
              "NN_MAX_WIDTH가 입력 특징 수보다 작다 — buf_a/buf_b 오버플로 발생");

class Preprocessor;  // full definition in preprocessor.h — only referenced
                      // by pointer/reference here, so a forward decl is enough


class NeuralNet {
public:
    NeuralNet();
    ~NeuralNet();

    // Load weights + biases + scaler from LittleFS. Expects a file of
    // exactly (weights+biases+2*MODEL_TOPOLOGY[0]) floats, laid out as
    // W0 b0 W1 b1 W2 b2 means stds (see docs/FIRMWARE_PROTOCOL.md 4-1).
    // On success, also feeds means/stds into `preproc` via
    // Preprocessor::setStandardizer() and sets isLoaded()==true.
    // Returns false (isLoaded() stays false) if the file is missing or
    // its size doesn't match the current MODEL_TOPOLOGY.
    bool loadFromLittleFS(const char* path, Preprocessor& preproc);

    bool isLoaded() const { return _loaded; }

    // Forward pass. input must have MODEL_TOPOLOGY[0] elements,
    // output must have MODEL_TOPOLOGY[MODEL_N_LAYERS-1] elements.
    // No-op (output left untouched) if !isLoaded().
    void predict(const float* input, float* output);

    // Return argmax index of a probability vector of length n.
    int  argmax(const float* output, int n);

    // Debug: number of features expected as input.
    int  inputSize()  const { return MODEL_TOPOLOGY[0]; }
    int  outputSize() const { return MODEL_TOPOLOGY[MODEL_N_LAYERS - 1]; }

private:
    bool   _loaded  = false;
    float* _weights = nullptr;  // W0+W1+W2 concatenated, heap-allocated in loadFromLittleFS()
    float* _biases  = nullptr;  // b0+b1+b2 concatenated, heap-allocated in loadFromLittleFS()

    // Two ping-pong buffers for layer outputs.
    float buf_a[NN_MAX_WIDTH];
    float buf_b[NN_MAX_WIDTH];

    // Apply activation in-place.
    void applyActivation(float* data, int n, int code);

    // Dense layer: out = activation(W^T * in + b)
    // weights are stored row-major with shape (in_dim, out_dim)
    //   weights[i * out_dim + j] = W[i, j]
    // matches Keras Dense layer weight format (kernel shape (in_dim, out_dim)).
    void denseLayer(const float* in, int in_dim,
                    float* out, int out_dim,
                    const float* weights, const float* biases,
                    int activation_code);
};

#endif // NN_H