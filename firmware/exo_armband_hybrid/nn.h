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

// Max layer width — sized for [58, 64, 64, 4], with margin.
#define NN_MAX_WIDTH 128

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