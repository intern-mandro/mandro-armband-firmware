#include "nn.h"
#include "preprocessor.h"
#include <LittleFS.h>
#include <math.h>


NeuralNet::NeuralNet() {
    // Nothing to init; weights are loaded later via loadFromLittleFS().
}

NeuralNet::~NeuralNet() {
    delete[] _weights;
    delete[] _biases;
}


// ═══════════════════════════════════════════════════════════════════════
// Load weights + biases + scaler from LittleFS
//
// File layout (see docs/FIRMWARE_PROTOCOL.md 4-1):
//   W0 b0 W1 b1 W2 b2 means stds   (float32, interleaved per layer)
//
// _weights/_biases keep the layout predict()/denseLayer() already expect
// (all weights concatenated, then all biases concatenated) — this function
// is what translates between the two layouts while reading.
// ═══════════════════════════════════════════════════════════════════════
bool NeuralNet::loadFromLittleFS(const char* path, Preprocessor& preproc) {
    // Element counts derived from MODEL_TOPOLOGY so this doesn't need to
    // change if the topology ever does.
    int totalWeights = 0;
    int totalBiases  = 0;
    for (int L = 0; L < MODEL_N_LAYERS - 1; L++) {
        totalWeights += MODEL_TOPOLOGY[L] * MODEL_TOPOLOGY[L + 1];
        totalBiases  += MODEL_TOPOLOGY[L + 1];
    }
    const int nFeatures = MODEL_TOPOLOGY[0];  // scaler mean/std length

    const size_t EXPECTED_BYTES =
        (size_t)(totalWeights + totalBiases + 2 * nFeatures) * sizeof(float);

    File f = LittleFS.open(path, "r");
    if (!f) return false;

    if ((size_t)f.size() != EXPECTED_BYTES) {
        f.close();
        return false;
    }

    float* newWeights = new float[totalWeights];
    float* newBiases  = new float[totalBiases];
    float* means      = new float[nFeatures];
    float* stds       = new float[nFeatures];

    bool ok = true;
    int wOff = 0, bOff = 0;
    for (int L = 0; L < MODEL_N_LAYERS - 1 && ok; L++) {
        int wCount = MODEL_TOPOLOGY[L] * MODEL_TOPOLOGY[L + 1];
        int bCount = MODEL_TOPOLOGY[L + 1];
        ok = ok && f.read((uint8_t*)(newWeights + wOff), wCount * sizeof(float)) == wCount * sizeof(float);
        ok = ok && f.read((uint8_t*)(newBiases  + bOff), bCount * sizeof(float)) == bCount * sizeof(float);
        wOff += wCount;
        bOff += bCount;
    }
    ok = ok && f.read((uint8_t*)means, nFeatures * sizeof(float)) == nFeatures * sizeof(float);
    ok = ok && f.read((uint8_t*)stds,  nFeatures * sizeof(float)) == nFeatures * sizeof(float);
    f.close();

    if (ok) {
        // Swap in the newly loaded weights only once everything above
        // succeeded, and only now free whatever was loaded before.
        delete[] _weights;
        delete[] _biases;
        _weights = newWeights;
        _biases  = newBiases;
        preproc.setStandardizer(means, stds);
        _loaded = true;
    } else {
        delete[] newWeights;
        delete[] newBiases;
    }
    delete[] means;
    delete[] stds;
    return ok;
}


// ═══════════════════════════════════════════════════════════════════════
// Activation functions
// ═══════════════════════════════════════════════════════════════════════
void NeuralNet::applyActivation(float* data, int n, int code) {
    switch (code) {
        case ACT_RELU:
            for (int i = 0; i < n; i++) {
                if (data[i] < 0.0f) data[i] = 0.0f;
            }
            break;

        case ACT_SOFTMAX: {
            // Numerically stable softmax: subtract max before exp
            float max_val = data[0];
            for (int i = 1; i < n; i++) {
                if (data[i] > max_val) max_val = data[i];
            }
            float sum = 0.0f;
            for (int i = 0; i < n; i++) {
                data[i] = expf(data[i] - max_val);
                sum += data[i];
            }
            float inv_sum = 1.0f / sum;
            for (int i = 0; i < n; i++) data[i] *= inv_sum;
            break;
        }

        case ACT_SIGMOID:
            for (int i = 0; i < n; i++) {
                data[i] = 1.0f / (1.0f + expf(-data[i]));
            }
            break;

        case ACT_TANH:
            for (int i = 0; i < n; i++) data[i] = tanhf(data[i]);
            break;

        case ACT_LINEAR:
        default:
            // No-op
            break;
    }
}


// ═══════════════════════════════════════════════════════════════════════
// Dense layer: out[j] = activation( sum_i(in[i] * W[i,j]) + b[j] )
// weights layout: row-major (in_dim, out_dim), matching Keras flatten()
// ═══════════════════════════════════════════════════════════════════════
void NeuralNet::denseLayer(const float* in, int in_dim,
                           float* out, int out_dim,
                           const float* weights, const float* biases,
                           int activation_code) {
    for (int j = 0; j < out_dim; j++) {
        float acc = biases[j];
        for (int i = 0; i < in_dim; i++) {
            acc += in[i] * weights[i * out_dim + j];
        }
        out[j] = acc;
    }
    applyActivation(out, out_dim, activation_code);
}


// ═══════════════════════════════════════════════════════════════════════
// Full forward pass
// ═══════════════════════════════════════════════════════════════════════
void NeuralNet::predict(const float* input, float* output) {
    if (!_loaded) return;  // no weights yet — caller should check isLoaded() first

    // Copy input to buf_a
    int in_dim = MODEL_TOPOLOGY[0];
    for (int i = 0; i < in_dim; i++) buf_a[i] = input[i];

    // Ping-pong through hidden layers
    const float* w_ptr = _weights;
    const float* b_ptr = _biases;

    float* cur_in  = buf_a;
    float* cur_out = buf_b;

    for (int L = 0; L < MODEL_N_LAYERS - 1; L++) {
        int in_d  = MODEL_TOPOLOGY[L];
        int out_d = MODEL_TOPOLOGY[L + 1];
        int act   = MODEL_ACTIVATIONS[L];

        denseLayer(cur_in, in_d, cur_out, out_d, w_ptr, b_ptr, act);

        // Advance pointers
        w_ptr += in_d * out_d;
        b_ptr += out_d;

        // Swap buffers
        float* tmp = cur_in;
        cur_in = cur_out;
        cur_out = tmp;
    }

    // Final output is in cur_in (after last swap)
    int out_dim = MODEL_TOPOLOGY[MODEL_N_LAYERS - 1];
    for (int i = 0; i < out_dim; i++) output[i] = cur_in[i];
}


// ═══════════════════════════════════════════════════════════════════════
// Argmax
// ═══════════════════════════════════════════════════════════════════════
int NeuralNet::argmax(const float* output, int n) {
    int best = 0;
    float best_val = output[0];
    for (int i = 1; i < n; i++) {
        if (output[i] > best_val) {
            best_val = output[i];
            best = i;
        }
    }
    return best;
}