#ifndef PREPROCESSOR_H
#define PREPROCESSOR_H

#pragma once
#include <Arduino.h>

#define N_CHANNEL       8           // ← was 4 for 4ch, 8 for 8ch
#define WINDOW_SIZE     128
#define FFT_SIZE        128       // == WINDOW_SIZE, must be power of 2
#define FFT_BINS        65        // FFT_SIZE/2 + 1

// ── Feature layout ─────────────────────────────────────────────────────
// Total features = classic (time + freq) + TSD (cov triu + energy)
// 4ch: 24+20 + 10+4 = 58
// 8ch: 48+40 + 36+8 = 132
#define N_TIME_FEATS    6           // mav, maxav, std, rms, wl, ssc
#define N_FREQ_FEATS    5           // mpow, tpow, mfreq, medfreq, peakfreq
#define N_COV_TRIU      ((N_CHANNEL * (N_CHANNEL + 1)) / 2)
#define N_FEATS_CLASSIC ((N_TIME_FEATS + N_FREQ_FEATS) * N_CHANNEL)
#define N_FEATS_TSD     (N_COV_TRIU + N_CHANNEL)
#define N_FEATURES      (N_FEATS_CLASSIC + N_FEATS_TSD)

// Offsets in features_raw[]
#define OFFSET_TIME     0                                      // 0..(6*N-1)
#define OFFSET_FREQ     (N_TIME_FEATS * N_CHANNEL)             // after time
#define OFFSET_TSD_COV  N_FEATS_CLASSIC                        // after classic
#define OFFSET_TSD_ENG  (N_FEATS_CLASSIC + N_COV_TRIU)         // after cov triu
#define SAMPLING_FREQ   1200.0f

#define BP_LOW_CUT      35.0f
#define BP_HIGH_CUT     300.0f
#define BP_ORDER        4
#define BP_TAPS         9         // 2*BP_ORDER + 1 for bandpass

#define ENVELOPE_KERNEL 12

#define TSD_WIN_MS      80
#define TSD_INC_MS      40
#define TSD_LAM         0.1f
#define TSD_WIN_SAMPLES 96        // int(1200 * 80/1000)
#define TSD_INC_SAMPLES 48        // int(1200 * 40/1000)


class Preprocessor {
public:
    Preprocessor();

    // Main pipeline: raw int samples -> 58 standardized features.
    // emg_window: WINDOW_SIZE rows x N_CHANNEL columns (int16 from ADC).
    // out_features: must point to at least N_FEATURES floats.
    void process(int16_t emg_window[WINDOW_SIZE][N_CHANNEL],
                 float out_features[N_FEATURES]);
    // Skip filter/rectify/envelope: inject an already-processed envelope
    // window and compute features + standardize. Used by Serial debug WIN_ENV.
    void processEnvelopeOnly(float env_in[WINDOW_SIZE][N_CHANNEL],
                             float out_features[N_FEATURES]);

    // ── Debug accessors (used by Serial debug mode) ────────────────────
    // After process(), these contain the intermediate steps.
    void getFiltered(float out[WINDOW_SIZE][N_CHANNEL]);
    void getRectified(float out[WINDOW_SIZE][N_CHANNEL]);
    void getEnvelope(float out[WINDOW_SIZE][N_CHANNEL]);
    void getFeaturesRaw(float out[N_FEATURES]);     // before scaler
    void getFeaturesStd(float out[N_FEATURES]);     // after scaler

    // Reset filter & envelope state (useful for debug, normally not used).
    void resetState();

    // Load StandardScaler mean/std (132 each) — called by
    // NeuralNet::loadFromLittleFS() after it parses the scaler section of
    // /weights.bin. Copies the values in (caller retains ownership of the
    // buffers passed in). Until this is called, standardize() uses a
    // no-op default (mean=0, std=1) so nothing divides by zero.
    void setStandardizer(const float* means, const float* stds);
    bool hasStandardizer() const { return _hasStandardizer; }

private:
    // ── Filter coefficients (must match Python butter(4, [35,300]/450)) ─
    static const float bp_b[BP_TAPS];
    static const float bp_a[BP_TAPS];
    static const float bp_zi[BP_TAPS - 1];   // lfilter_zi template

    // ── Filter state (Direct Form II Transposed, per channel) ──────────
    float filter_state[N_CHANNEL][BP_TAPS - 1];
    bool  filter_initialized[N_CHANNEL];     // for lfilter_zi behavior

    // ── Envelope: causal moving average via circular buffer ────────────
    float env_buffer[N_CHANNEL][ENVELOPE_KERNEL];
    float env_sum[N_CHANNEL];
    int   env_idx;
    int   env_count;                          // how many samples seen (saturates at ENVELOPE_KERNEL)

    // ── Intermediate storage for debug + computation ───────────────────
    float filtered_buf[WINDOW_SIZE][N_CHANNEL];
    float rectified_buf[WINDOW_SIZE][N_CHANNEL];
    float envelope_buf[WINDOW_SIZE][N_CHANNEL];
    float features_raw[N_FEATURES];
    float features_std[N_FEATURES];

    // ── Standard scaler, set at runtime via setStandardizer() ──────────
    float _standardizerMeans[N_FEATURES];
    float _standardizerStds[N_FEATURES];
    bool  _hasStandardizer = false;

    // ── FFT workspace ──────────────────────────────────────────────────
    float fft_real[FFT_SIZE];
    float fft_imag[FFT_SIZE];
    float fft_power[FFT_BINS];   // |FFT|^2 for one channel

    // ── Pipeline steps ─────────────────────────────────────────────────
    void applyBandpass(int16_t raw_in[WINDOW_SIZE][N_CHANNEL]);
    void applyRectify();
    void applyEnvelope();
    void extractClassicFeatures();
    void extractTSDFeatures();
    void standardize();

    // ── Building blocks ────────────────────────────────────────────────
    void computeFFT_radix2(float channel_data[WINDOW_SIZE]);
    void computeFFTPowerSpectrum(float channel_data[WINDOW_SIZE]);

    float mav(float* data, int n);
    float maxAbs(float* data, int n);
    float stddev(float* data, int n);
    float rms(float* data, int n);
    float waveformLength(float* data, int n);
    int   slopeSignChanges(float* data, int n);

    float meanPower(float* power, int n_bins);
    float totalPower(float* power, int n_bins);
    float meanFrequency(float* power, int n_bins, float total);
    float medianFrequency(float* power, int n_bins);
    float peakFrequency(float* power, int n_bins);

    // TSD on a sub-window of size win_samples
    void  computeTSDSubwindow(int start_idx, int win_samples,
                              float out_cov_triu[N_COV_TRIU], float out_energy[N_CHANNEL]);
};

#endif // PREPROCESSOR_H