#include "preprocessor.h"
#include <math.h>


// ═══════════════════════════════════════════════════════════════════════
// Filter coefficients
// Computed in Python with: butter(4, [35, 300] / (1200/2), btype='band')
// These MUST match exactly the coefficients in test_vectors.json metadata.
// ═══════════════════════════════════════════════════════════════════════
const float Preprocessor::bp_b[BP_TAPS] = {
     0.0636531173674723f,  0.0f,                -0.2546124694698890f,
     0.0f,                 0.3819187042048335f,  0.0f,
    -0.2546124694698890f,  0.0f,                 0.0636531173674723f
};

const float Preprocessor::bp_a[BP_TAPS] = {
     1.0f,                -3.7058545966788046f,  5.9335875406013256f,
    -5.8348440797927106f,  4.2487090364011282f, -2.2839036570242182f,
     0.7771775878379237f, -0.1549141739103137f,  0.0217504621436292f
};

// lfilter_zi(b, a) template — initial conditions scaled by signal[0] at runtime.
// Computed in Python: scipy.signal.lfilter_zi(b, a)
// >>> regenerate this manually if you ever change b, a <
// Length = BP_TAPS - 1 = 8
const float Preprocessor::bp_zi[BP_TAPS - 1] = {
    -0.0636531174f, -0.0636531174f,  0.1909593521f,  0.1909593521f,
    -0.1909593521f, -0.1909593521f,  0.0636531174f,  0.0636531174f
};


// ═══════════════════════════════════════════════════════════════════════
// Constructor
// ═══════════════════════════════════════════════════════════════════════
Preprocessor::Preprocessor() {
    resetState();
    // Safe defaults until setStandardizer() is called (avoids div-by-zero
    // if standardize() somehow runs before a scaler is loaded).
    for (int i = 0; i < N_FEATURES; i++) {
        _standardizerMeans[i] = 0.0f;
        _standardizerStds[i]  = 1.0f;
    }
}

void Preprocessor::resetState() {
    for (int ch = 0; ch < N_CHANNEL; ch++) {
        for (int i = 0; i < BP_TAPS - 1; i++) {
            filter_state[ch][i] = 0.0f;
        }
        filter_initialized[ch] = false;

        for (int i = 0; i < ENVELOPE_KERNEL; i++) {
            env_buffer[ch][i] = 0.0f;
        }
        env_sum[ch] = 0.0f;
    }
    env_idx = 0;
    env_count = 0;
}


// ═══════════════════════════════════════════════════════════════════════
// MAIN entry point
// ═══════════════════════════════════════════════════════════════════════
void Preprocessor::process(int16_t emg_window[WINDOW_SIZE][N_CHANNEL],
                           float out_features[N_FEATURES]) {
    applyBandpass(emg_window);
    applyRectify();
    applyEnvelope();
    extractClassicFeatures();
    extractTSDFeatures();
    standardize();

    for (int i = 0; i < N_FEATURES; i++) {
        out_features[i] = features_std[i];
    }
}

void Preprocessor::processEnvelopeOnly(float env_in[WINDOW_SIZE][N_CHANNEL],
                                       float out_features[N_FEATURES]) {
    for (int n = 0; n < WINDOW_SIZE; n++) {
        for (int ch = 0; ch < N_CHANNEL; ch++) {
            envelope_buf[n][ch] = env_in[n][ch];
        }
    }
    extractClassicFeatures();
    extractTSDFeatures();
    standardize();
    for (int i = 0; i < N_FEATURES; i++) {
        out_features[i] = features_std[i];
    }
}
// ═══════════════════════════════════════════════════════════════════════
// SECTION 1 — Causal Butterworth bandpass (Direct Form II Transposed)
// Matches scipy.signal.lfilter(b, a, x, zi=lfilter_zi(b,a) * x[0]) exactly.
// ═══════════════════════════════════════════════════════════════════════
void Preprocessor::applyBandpass(int16_t raw_in[WINDOW_SIZE][N_CHANNEL]) {
    for (int ch = 0; ch < N_CHANNEL; ch++) {
        // Initialize state with lfilter_zi * x[0] (only once per channel)
        if (!filter_initialized[ch]) {
            float x0 = (float)raw_in[0][ch];
            for (int i = 0; i < BP_TAPS - 1; i++) {
                filter_state[ch][i] = bp_zi[i] * x0;
            }
            filter_initialized[ch] = true;
        }

        // Direct Form II Transposed:
        //   y[n] = b[0]*x[n] + z[0]
        //   z[i] = b[i+1]*x[n] - a[i+1]*y[n] + z[i+1]   for i = 0..M-2
        //   z[M-1] = b[M]*x[n] - a[M]*y[n]
        for (int n = 0; n < WINDOW_SIZE; n++) {
            float x = (float)raw_in[n][ch];
            float y = bp_b[0] * x + filter_state[ch][0];

            for (int i = 0; i < BP_TAPS - 2; i++) {
                filter_state[ch][i] = bp_b[i + 1] * x
                                    - bp_a[i + 1] * y
                                    + filter_state[ch][i + 1];
            }
            filter_state[ch][BP_TAPS - 2] = bp_b[BP_TAPS - 1] * x
                                          - bp_a[BP_TAPS - 1] * y;

            filtered_buf[n][ch] = y;
        }
    }
}


// ═══════════════════════════════════════════════════════════════════════
// SECTION 2 — Rectification (abs)
// ═══════════════════════════════════════════════════════════════════════
void Preprocessor::applyRectify() {
    for (int n = 0; n < WINDOW_SIZE; n++) {
        for (int ch = 0; ch < N_CHANNEL; ch++) {
            rectified_buf[n][ch] = fabsf(filtered_buf[n][ch]);
        }
    }
}


// ═══════════════════════════════════════════════════════════════════════
// SECTION 3 — Causal moving-average envelope (circular buffer)
// Matches numpy.convolve(rectified, ones(K)/K, mode='full')[:N] exactly:
//   y[n] = (1/K) * sum(rectified[max(0, n-K+1) .. n])
// Equivalently: y[n] uses only past samples, with implicit zero-padding
// before the start (sample at index < 0 = 0).
// ═══════════════════════════════════════════════════════════════════════
void Preprocessor::applyEnvelope() {
    // Reset envelope state at start of each window — to match Python's
    // np.convolve which starts fresh each call.
    for (int ch = 0; ch < N_CHANNEL; ch++) {
        for (int i = 0; i < ENVELOPE_KERNEL; i++) {
            env_buffer[ch][i] = 0.0f;
        }
        env_sum[ch] = 0.0f;
    }
    env_idx = 0;

    const float kernel_inv = 1.0f / (float)ENVELOPE_KERNEL;

    for (int n = 0; n < WINDOW_SIZE; n++) {
        for (int ch = 0; ch < N_CHANNEL; ch++) {
            // Remove oldest sample from sum, add newest
            env_sum[ch] -= env_buffer[ch][env_idx];
            env_buffer[ch][env_idx] = rectified_buf[n][ch];
            env_sum[ch] += env_buffer[ch][env_idx];

            envelope_buf[n][ch] = env_sum[ch] * kernel_inv;
        }
        env_idx = (env_idx + 1) % ENVELOPE_KERNEL;
    }
}


// ═══════════════════════════════════════════════════════════════════════
// SECTION 4 — FFT radix-2 in-place
// Matches numpy.fft.rfft on FFT_SIZE-point real input.
// Output convention: fft_real[k] + j * fft_imag[k] for k = 0..FFT_SIZE/2
// (we only use indices 0..FFT_BINS-1).
// ═══════════════════════════════════════════════════════════════════════
void Preprocessor::computeFFT_radix2(float channel_data[WINDOW_SIZE]) {
    // Copy input to real, zero imag
    for (int i = 0; i < FFT_SIZE; i++) {
        fft_real[i] = channel_data[i];
        fft_imag[i] = 0.0f;
    }

    // Bit reversal
    int n = FFT_SIZE;
    int j = 0;
    for (int i = 1; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            float tr = fft_real[i]; fft_real[i] = fft_real[j]; fft_real[j] = tr;
            float ti = fft_imag[i]; fft_imag[i] = fft_imag[j]; fft_imag[j] = ti;
        }
    }

    // Cooley-Tukey butterflies
    for (int len = 2; len <= n; len <<= 1) {
        float ang = -2.0f * (float)M_PI / (float)len;
        float wlen_r = cosf(ang);
        float wlen_i = sinf(ang);

        for (int i = 0; i < n; i += len) {
            float w_r = 1.0f;
            float w_i = 0.0f;

            for (int k = 0; k < len / 2; k++) {
                float u_r = fft_real[i + k];
                float u_i = fft_imag[i + k];
                float v_r = fft_real[i + k + len / 2] * w_r
                          - fft_imag[i + k + len / 2] * w_i;
                float v_i = fft_real[i + k + len / 2] * w_i
                          + fft_imag[i + k + len / 2] * w_r;

                fft_real[i + k] = u_r + v_r;
                fft_imag[i + k] = u_i + v_i;
                fft_real[i + k + len / 2] = u_r - v_r;
                fft_imag[i + k + len / 2] = u_i - v_i;

                float nw_r = w_r * wlen_r - w_i * wlen_i;
                float nw_i = w_r * wlen_i + w_i * wlen_r;
                w_r = nw_r;
                w_i = nw_i;
            }
        }
    }
}


void Preprocessor::computeFFTPowerSpectrum(float channel_data[WINDOW_SIZE]) {
    computeFFT_radix2(channel_data);
    for (int k = 0; k < FFT_BINS; k++) {
        fft_power[k] = fft_real[k] * fft_real[k] + fft_imag[k] * fft_imag[k];
    }
}

// ═══════════════════════════════════════════════════════════════════════
// SECTION 5 — Classic features (time-domain primitives)
// ═══════════════════════════════════════════════════════════════════════

float Preprocessor::mav(float* data, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += fabsf(data[i]);
    return sum / (float)n;
}

float Preprocessor::maxAbs(float* data, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; i++) {
        float a = fabsf(data[i]);
        if (a > m) m = a;
    }
    return m;
}

float Preprocessor::stddev(float* data, int n) {
    // numpy.std default uses ddof=0 (population std)
    float mean = 0.0f;
    for (int i = 0; i < n; i++) mean += data[i];
    mean /= (float)n;

    float sumsq = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = data[i] - mean;
        sumsq += d * d;
    }
    return sqrtf(sumsq / (float)n);
}

float Preprocessor::rms(float* data, int n) {
    float sumsq = 0.0f;
    for (int i = 0; i < n; i++) sumsq += data[i] * data[i];
    return sqrtf(sumsq / (float)n);
}

float Preprocessor::waveformLength(float* data, int n) {
    float sum = 0.0f;
    for (int i = 1; i < n; i++) sum += fabsf(data[i] - data[i - 1]);
    return sum;
}

int Preprocessor::slopeSignChanges(float* data, int n) {
    // numpy diff/sign equivalent:
    //   d[i] = data[i+1] - data[i]
    //   sign(0) = 0
    //   count where sign changes between consecutive diffs
    int count = 0;
    for (int i = 1; i < n - 1; i++) {
        float d1 = data[i] - data[i - 1];
        float d2 = data[i + 1] - data[i];
        int s1 = (d1 > 0) - (d1 < 0);   // sign in {-1, 0, 1}
        int s2 = (d2 > 0) - (d2 < 0);
        if (s1 != s2) count++;
    }
    return count;
}


// ═══════════════════════════════════════════════════════════════════════
// SECTION 5b — Frequency-domain features
// ═══════════════════════════════════════════════════════════════════════

float Preprocessor::meanPower(float* power, int n_bins) {
    float sum = 0.0f;
    for (int k = 0; k < n_bins; k++) sum += power[k];
    return sum / (float)n_bins;
}

float Preprocessor::totalPower(float* power, int n_bins) {
    float sum = 0.0f;
    for (int k = 0; k < n_bins; k++) sum += power[k];
    return sum;
}

float Preprocessor::meanFrequency(float* power, int n_bins, float total) {
    // numpy: np.sum(freq * p) / total
    // freq[k] = k * (fs / FFT_SIZE)
    float bin_width = SAMPLING_FREQ / (float)FFT_SIZE;
    float weighted = 0.0f;
    for (int k = 0; k < n_bins; k++) {
        weighted += (float)k * bin_width * power[k];
    }
    return weighted / (total + 1e-12f);
}

float Preprocessor::medianFrequency(float* power, int n_bins) {
    // numpy:
    //   total = power.sum() + 1e-12
    //   cumsum = np.cumsum(power)
    //   idx = np.searchsorted(cumsum, total / 2)
    //   return freq[idx]
    //
    // searchsorted default 'left' returns first index i where cumsum[i] >= total/2
    float total = 0.0f;
    for (int k = 0; k < n_bins; k++) total += power[k];
    total += 1e-12f;
    float half = total * 0.5f;

    float cum = 0.0f;
    int idx = n_bins - 1;   // default if half never reached
    for (int k = 0; k < n_bins; k++) {
        cum += power[k];
        if (cum >= half) { idx = k; break; }
    }
    float bin_width = SAMPLING_FREQ / (float)FFT_SIZE;
    return (float)idx * bin_width;
}

float Preprocessor::peakFrequency(float* power, int n_bins) {
    int peak_idx = 0;
    float peak_val = power[0];
    for (int k = 1; k < n_bins; k++) {
        if (power[k] > peak_val) {
            peak_val = power[k];
            peak_idx = k;
        }
    }
    float bin_width = SAMPLING_FREQ / (float)FFT_SIZE;
    return (float)peak_idx * bin_width;
}


// ═══════════════════════════════════════════════════════════════════════
// SECTION 6 — Extract all classic features (44 values)
// Order MUST match Python exactly:
//   [mav, maxav, std, rms, wl, ssc, mpow, tpow, mfreq, medfreq, peakfreq]
//   × N_CHANNEL = 44 features
// ═══════════════════════════════════════════════════════════════════════
void Preprocessor::extractClassicFeatures() {
    float channel_data[WINDOW_SIZE];

    // ── Time-domain: grouped BY FEATURE (mav_ch0..3, maxav_ch0..3, ...) ────
    // Indices 0..23
    for (int ch = 0; ch < N_CHANNEL; ch++) {
        for (int n = 0; n < WINDOW_SIZE; n++) channel_data[n] = envelope_buf[n][ch];

        features_raw[OFFSET_TIME + 0*N_CHANNEL + ch] = mav(channel_data, WINDOW_SIZE);
        features_raw[OFFSET_TIME + 1*N_CHANNEL + ch] = maxAbs(channel_data, WINDOW_SIZE);
        features_raw[OFFSET_TIME + 2*N_CHANNEL + ch] = stddev(channel_data, WINDOW_SIZE);
        features_raw[OFFSET_TIME + 3*N_CHANNEL + ch] = rms(channel_data, WINDOW_SIZE);
        features_raw[OFFSET_TIME + 4*N_CHANNEL + ch] = waveformLength(channel_data, WINDOW_SIZE);
        features_raw[OFFSET_TIME + 5*N_CHANNEL + ch] = (float)slopeSignChanges(channel_data, WINDOW_SIZE);
    }

    // ── Frequency-domain: grouped BY CHANNEL ───────────────────────────────
    // Layout per channel: [mpow, tpow, mfreq, medfreq, peakfreq]
    // ch0 -> indices 24..28
    // ch1 -> indices 29..33
    // ch2 -> indices 34..38
    // ch3 -> indices 39..43
    for (int ch = 0; ch < N_CHANNEL; ch++) {
        for (int n = 0; n < WINDOW_SIZE; n++) channel_data[n] = envelope_buf[n][ch];

        computeFFTPowerSpectrum(channel_data);

        float total = totalPower(fft_power, FFT_BINS);
        float total_safe = total + 1e-12f;

        int base = OFFSET_FREQ + ch * N_FREQ_FEATS;
        features_raw[base + 0] = meanPower(fft_power, FFT_BINS);          // mpow
        features_raw[base + 1] = total;                                    // tpow
        features_raw[base + 2] = meanFrequency(fft_power, FFT_BINS, total_safe); // mfreq
        features_raw[base + 3] = medianFrequency(fft_power, FFT_BINS);    // medfreq
        features_raw[base + 4] = peakFrequency(fft_power, FFT_BINS);      // peakfreq
    }
}


// ═══════════════════════════════════════════════════════════════════════
// SECTION 7 — TSD features (14 values, indices 44..57)
// Order:
//   tsd_cov_00, tsd_cov_01, tsd_cov_02, tsd_cov_03,
//   tsd_cov_11, tsd_cov_12, tsd_cov_13,
//   tsd_cov_22, tsd_cov_23,
//   tsd_cov_33,
//   tsd_energy_ch0..3
//
// Method: slide a sub-window of TSD_WIN_SAMPLES through the envelope,
// compute regularized covariance (+ lam*I) and per-channel energy,
// then average over all sub-windows.
// ═══════════════════════════════════════════════════════════════════════
void Preprocessor::computeTSDSubwindow(int start_idx, int win_samples,
                                       float out_cov_triu[N_COV_TRIU],
                                       float out_energy[N_CHANNEL]) {
    // Compute means per channel on the sub-window
    float mean_ch[N_CHANNEL];
    for (int ch = 0; ch < N_CHANNEL; ch++) mean_ch[ch] = 0.0f;
    for (int n = 0; n < win_samples; n++) {
        for (int ch = 0; ch < N_CHANNEL; ch++) {
            mean_ch[ch] += envelope_buf[start_idx + n][ch];
        }
    }
    for (int ch = 0; ch < N_CHANNEL; ch++) {
        mean_ch[ch] /= (float)win_samples;
    }

    // Covariance matrix (upper triangle including diagonal: 10 entries)
    // np.cov uses ddof=1 by default: divide by (N-1)
    float cov[N_CHANNEL][N_CHANNEL];
    for (int i = 0; i < N_CHANNEL; i++) {
        for (int j = i; j < N_CHANNEL; j++) {
            float sum = 0.0f;
            for (int n = 0; n < win_samples; n++) {
                float di = envelope_buf[start_idx + n][i] - mean_ch[i];
                float dj = envelope_buf[start_idx + n][j] - mean_ch[j];
                sum += di * dj;
            }
            cov[i][j] = sum / (float)(win_samples - 1);
            // Regularization: + lam * I (only on diagonal)
            if (i == j) cov[i][j] += TSD_LAM;
        }
    }

    // Upper-triangle (i <= j) in row-major order: (0,0),(0,1),(0,2),(0,3),
    // (1,1),(1,2),(1,3),(2,2),(2,3),(3,3)
    int k = 0;
    for (int i = 0; i < N_CHANNEL; i++) {
        for (int j = i; j < N_CHANNEL; j++) {
            out_cov_triu[k++] = cov[i][j];
        }
    }

    // Energy per channel: mean of squared values
    for (int ch = 0; ch < N_CHANNEL; ch++) {
        float sumsq = 0.0f;
        for (int n = 0; n < win_samples; n++) {
            float v = envelope_buf[start_idx + n][ch];
            sumsq += v * v;
        }
        out_energy[ch] = sumsq / (float)win_samples;
    }
}


void Preprocessor::extractTSDFeatures() {
    float cov_acc[N_COV_TRIU];
    float energy_acc[N_CHANNEL];
    for (int i = 0; i < N_COV_TRIU; i++) cov_acc[i]    = 0.0f;
    for (int i = 0; i < N_CHANNEL; i++) energy_acc[i] = 0.0f;
    int   n_subwin       = 0;

    int start = 0;
    while (start + TSD_WIN_SAMPLES <= WINDOW_SIZE) {
        float cov_triu[N_COV_TRIU];
        float energy[N_CHANNEL];
        computeTSDSubwindow(start, TSD_WIN_SAMPLES, cov_triu, energy);

        for (int i = 0; i < N_COV_TRIU; i++) cov_acc[i]    += cov_triu[i];
        for (int i = 0; i < N_CHANNEL; i++) energy_acc[i] += energy[i];
        n_subwin++;

        start += TSD_INC_SAMPLES;
    }

    // Edge case: if no sub-window fits (shouldn't happen with WINDOW_SIZE=128
    // and TSD_WIN_SAMPLES=103, since one full sub-window fits), use whole window
    if (n_subwin == 0) {
        float cov_triu[N_COV_TRIU];
        float energy[N_CHANNEL];
        computeTSDSubwindow(0, WINDOW_SIZE, cov_triu, energy);
        for (int i = 0; i < N_COV_TRIU; i++) cov_acc[i]    = cov_triu[i];
        for (int i = 0; i < N_CHANNEL; i++) energy_acc[i] = energy[i];
        n_subwin = 1;
    }

    // Average
    float inv_n = 1.0f / (float)n_subwin;
    for (int i = 0; i < N_COV_TRIU; i++) features_raw[OFFSET_TSD_COV + i] = cov_acc[i]    * inv_n;
    for (int i = 0; i < N_CHANNEL; i++) features_raw[OFFSET_TSD_ENG + i] = energy_acc[i] * inv_n;
}


// ═══════════════════════════════════════════════════════════════════════
// SECTION 8 — Standardization
// Mean/std are no longer compiled in (means.h/stds.h removed) — they're
// loaded at boot from LittleFS via setStandardizer(), same as the NN
// weights in nn.cpp. See docs/FIRMWARE_PROTOCOL.md 4-1.
//
// Numerically robust form: (x/std) - (mean/std)
// to avoid catastrophic cancellation on features with large dynamics
// (e.g. tpow which can reach 2.9M).
// ═══════════════════════════════════════════════════════════════════════
void Preprocessor::setStandardizer(const float* means, const float* stds) {
    for (int i = 0; i < N_FEATURES; i++) {
        _standardizerMeans[i] = means[i];
        _standardizerStds[i]  = stds[i];
    }
    _hasStandardizer = true;
}

void Preprocessor::standardize() {
    for (int i = 0; i < N_FEATURES; i++) {
        float inv_std = 1.0f / _standardizerStds[i];
        features_std[i] = features_raw[i] * inv_std
                        - _standardizerMeans[i] * inv_std;
    }
}


// ═══════════════════════════════════════════════════════════════════════
// Debug accessors
// ═══════════════════════════════════════════════════════════════════════
void Preprocessor::getFiltered(float out[WINDOW_SIZE][N_CHANNEL]) {
    for (int n = 0; n < WINDOW_SIZE; n++)
        for (int ch = 0; ch < N_CHANNEL; ch++)
            out[n][ch] = filtered_buf[n][ch];
}

void Preprocessor::getRectified(float out[WINDOW_SIZE][N_CHANNEL]) {
    for (int n = 0; n < WINDOW_SIZE; n++)
        for (int ch = 0; ch < N_CHANNEL; ch++)
            out[n][ch] = rectified_buf[n][ch];
}

void Preprocessor::getEnvelope(float out[WINDOW_SIZE][N_CHANNEL]) {
    for (int n = 0; n < WINDOW_SIZE; n++)
        for (int ch = 0; ch < N_CHANNEL; ch++)
            out[n][ch] = envelope_buf[n][ch];
}

void Preprocessor::getFeaturesRaw(float out[N_FEATURES]) {
    for (int i = 0; i < N_FEATURES; i++) out[i] = features_raw[i];
}

void Preprocessor::getFeaturesStd(float out[N_FEATURES]) {
    for (int i = 0; i < N_FEATURES; i++) out[i] = features_std[i];
}