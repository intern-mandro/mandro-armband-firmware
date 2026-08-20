#ifndef MODEL_H
#define MODEL_H

// Topology/activation constants only — these define the network's fixed
// structure and don't change on retraining, so they stay compiled in.
// Actual weights/biases are no longer hardcoded here; they're loaded at
// boot time from LittleFS (/weights.bin) via NeuralNet::loadFromLittleFS().
// See docs/FIRMWARE_PROTOCOL.md 4-1 and LOCAL_MIGRATION.md for the file
// format (W0 b0 W1 b1 W2 b2 means stds, interleaved per layer).

const int MODEL_N_LAYERS = 4;

const int MODEL_TOPOLOGY[4] = {
    132, 64, 64, 6
};

// Activation codes: 0=RELU, 1=SOFTMAX, 2=SIGMOID, 3=TANH, 4=LINEAR
const int MODEL_ACTIVATIONS[3] = {
    0, 0, 1
};

#endif // MODEL_H
