import serial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from collections import deque
from scipy.signal import butter, filtfilt
import time
import csv

# ----------------- Settings -----------------
PORT = "/dev/ttyACM0"
BAUD = 1000000
N = 1024
HEADER = b'\xCD\xAB'
VREF = 3.3
FS = 18860.0
HISTORY_LEN = 80
TRACK_LEN = 150
FRAME_SIZE = 2 + N * 2
MIN_FREQ = 5.0
PEAK_MIN_DB = -40
EXCLUDE_BINS = 5

# ----------------- Serial -----------------
ser = serial.Serial(PORT, BAUD, timeout=0.05)

# ----------------- Figures -----------------
plt.ion()
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
fig.canvas.manager.set_window_title("Real-Time FFT Analyzer with CSV Save & Filter Toggle")

ax_time, ax_fft, ax_spec, ax_track = axs.flatten()

# --- Time domain ---
time_line, = ax_time.plot([], [], color='black')
ax_time.set_title("Time Domain")
ax_time.set_xlabel("Time [s]")
ax_time.set_ylabel("Voltage [V]")
ax_time.grid(True)

# --- FFT Spectrum ---
fft_line, = ax_fft.plot([], [], color='purple')
peak_marker, = ax_fft.plot([], [], 'go', markersize=8, label="Main Peak")
sec_peak_marker, = ax_fft.plot([], [], 'ro', markersize=6, label="2nd Peak")
ax_fft.set_title("FFT Spectrum")
ax_fft.set_xlabel("Frequency [Hz]")
ax_fft.set_ylabel("Magnitude [dB]")
ax_fft.grid(True)
ax_fft.legend(loc="upper right")

# --- Spectrogram ---
# spectrogram image shape: (freq_bins, time_history_length)
spec_img = ax_spec.imshow(np.zeros((N//2, HISTORY_LEN)),
                          aspect='auto', origin='lower',
                          extent=[0, HISTORY_LEN, 0, FS/2],
                          cmap='inferno', vmin=-100, vmax=0)
ax_spec.set_title("Spectrogram")
ax_spec.set_xlabel("Frame Index")
ax_spec.set_ylabel("Frequency [Hz]")

# --- Frequency Tracking ---
track_line, = ax_track.plot([], [], color='teal')
ax_track.set_title("Main Frequency Tracking")
ax_track.set_xlabel("Frame")
ax_track.set_ylabel("Frequency [Hz]")
ax_track.grid(True)

# --- Buttons ---
button_ax_save = plt.axes([0.83, 0.02, 0.13, 0.05])
save_button = Button(button_ax_save, '💾 Save CSV', color='lightgray', hovercolor='lightgreen')

button_ax_filter = plt.axes([0.7, 0.02, 0.13, 0.05])
filter_button = Button(button_ax_filter, '🎚 Toggle Filter', color='lightgray', hovercolor='lightblue')

filter_enabled = True  # default ON

def toggle_filter(event):
    global filter_enabled
    filter_enabled = not filter_enabled
    state = "ON" if filter_enabled else "OFF"
    print(f"🔧 Band-pass filter: {state}")

filter_button.on_clicked(toggle_filter)

# ----------------- Buffers -----------------
# ensure spec_history elements length matches what's shown: N//2
spec_history = deque([np.zeros(N//2) for _ in range(HISTORY_LEN)], maxlen=HISTORY_LEN)
freq_history = deque([0]*TRACK_LEN, maxlen=TRACK_LEN)
time_history = deque(list(range(-TRACK_LEN+1, 1)), maxlen=TRACK_LEN)  # integer frame indices
prev_time = time.time()
frame_count = 0
fps_display = 0

# ----------------- Data Window -----------------
fig_info, ax_info = plt.subplots(figsize=(5, 3))
fig_info.canvas.manager.set_window_title("Live FFT Data")
ax_info.axis("off")
info_text = ax_info.text(0.05, 0.95, "", fontsize=12, va="top", family="monospace")

# ----------------- Helper Functions -----------------
def smooth(data, w=5):
    if len(data) < w:
        return np.array(data)
    return np.convolve(data, np.ones(w)/w, mode='same')

def parabolic_interpolation_linear(mag_lin, idx):
    """
    Parabolic interpolation using linear magnitude values (not dB).
    Returns fractional bin offset (can be negative/positive).
    """
    if idx <= 0 or idx >= len(mag_lin) - 1:
        return 0.0
    alpha = mag_lin[idx - 1]
    beta = mag_lin[idx]
    gamma = mag_lin[idx + 1]
    denom = (alpha - 2*beta + gamma)
    if denom == 0:
        return 0.0
    return 0.5 * (alpha - gamma) / denom

# --- Band-pass Filter ---
def bandpass_filter(data, fs, lowcut=10, highcut=4500, order=5):
    if len(data) < 16:  # filtfilt needs enough samples; 16 is a small safe-check
        return data
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = min(highcut / nyq, 0.99)
    if not (0 < low < high < 1):
        return data
    b, a = butter(order, [low, high], btype='band')
    # filtfilt can fail if array too short for the filter padlen; guard with try/except
    try:
        return filtfilt(b, a, data)
    except Exception:
        # fallback: return unfiltered data if filtfilt fails
        return data

# --- FFT & Peaks ---
def find_peaks(volt):
    """
    Returns:
      freq_axis_full, mag_db_full,
      main_freq_hz, main_amp_lin,
      second_freq_hz, second_amp_lin,
      rms_volt, snr_db
    """
    volt = volt - np.mean(volt)
    if filter_enabled:
        volt = bandpass_filter(volt, FS)  # apply filter dynamically (safe-guarded)

    if len(volt) != N:
        # if we don't have N samples, pad or trim to N
        volt = np.resize(volt, N)

    window = np.hanning(N)
    fft_vals = np.fft.rfft(volt * window)
    # convert to linear amplitude spectrum with window correction
    # normalization: factor = 2/N divided by mean of window (to correct amplitude)
    window_correction = np.sum(window)/N
    mag = (2.0 / N) * np.abs(fft_vals) / (window_correction + 1e-16)
    mag_db = 20 * np.log10(mag + 1e-12)
    freq_axis = np.fft.rfftfreq(N, 1/FS)

    # mask out very low freqs below MIN_FREQ
    mask = freq_axis > MIN_FREQ
    if np.count_nonzero(mask) == 0:
        # no valid bins, return safe defaults
        main_freq = 0.0
        main_amp = 0.0
        second_freq = 0.0
        second_amp = 0.0
        rms = np.sqrt(np.mean(volt**2))
        snr_db = np.nan
        return freq_axis, mag_db, main_freq, main_amp, second_freq, second_amp, rms, snr_db

    mag_valid = mag[mask]
    mag_db_valid = mag_db[mask]
    freq_axis_valid = freq_axis[mask]

    # find main peak on linear magnitude (more appropriate for interpolation)
    main_idx = int(np.argmax(mag_valid))
    main_amp = mag_valid[main_idx]
    # parabolic refine using linear values:
    delta = parabolic_interpolation_linear(mag_valid, main_idx)
    df = FS / N
    main_freq = freq_axis_valid[main_idx] + delta * df

    # exclude neighborhood around main peak (in index space) to find second peak
    mag_copy = mag_db_valid.copy()
    start = max(main_idx - EXCLUDE_BINS, 0)
    end = min(main_idx + EXCLUDE_BINS + 1, len(mag_copy))
    mag_copy[start:end] = -np.inf
    second_idx = int(np.argmax(mag_copy))
    second_amp = mag[second_idx] if len(mag_valid) > 0 else 0.0
    second_freq = freq_axis_valid[second_idx] if len(freq_axis_valid) > 0 else 0.0

    rms = np.sqrt(np.mean(volt**2))
    snr_db = 20 * np.log10((main_amp + 1e-12) / (second_amp + 1e-12)) if second_amp > 0 else np.nan

    return freq_axis, mag_db, main_freq, main_amp, second_freq, second_amp, rms, snr_db

# ----------------- CSV Save -----------------
current_freq_axis, current_mag_db = np.zeros(N//2+1), np.zeros(N//2+1)

def save_to_csv(event):
    filename = f"fft_snapshot_{int(time.time())}.csv"
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Frequency (Hz)", "Magnitude (dB)"])
        for f_hz, mag_db_val in zip(current_freq_axis, current_mag_db):
            writer.writerow([f_hz, mag_db_val])
    print(f"✅ Saved FFT data to {filename}")

save_button.on_clicked(save_to_csv)

# ----------------- Main Loop -----------------
while True:
    try:
        frame = ser.read(FRAME_SIZE)
        if len(frame) != FRAME_SIZE or frame[0:2] != HEADER:
            continue

        adc_vals = np.frombuffer(frame[2:], dtype=np.uint16)
        if adc_vals.size != N:
            # guard: if wrong count, skip this frame
            print("⚠️ Wrong ADC sample count, skipping frame")
            continue

        # Convert ADC to voltage (12-bit assumed)
        volt = (adc_vals * VREF / 4095.0) * 2.0
        # calibration polynom (as you had)
        volt = (-0.0269 * np.square(volt)) + (2.2672 * volt) + 0.0571
        volt = volt * 1.075

        freq_axis, mag_db, main_freq, main_amp, sec_freq, sec_amp, rms, snr_db = find_peaks(volt)
        current_freq_axis, current_mag_db = freq_axis, mag_db

        # --- Update plots ---
        time_line.set_data(np.arange(N) / FS, volt)
        ax_time.relim(); ax_time.autoscale_view()

        fft_line.set_data(freq_axis, mag_db)
        peak_marker.set_data([main_freq], [20*np.log10(main_amp + 1e-12)])
        sec_peak_marker.set_data([sec_freq], [20*np.log10(sec_amp + 1e-12)])
        ax_fft.relim(); ax_fft.autoscale_view()

        # update spectrogram (we take first N//2 bins for visual)
        spec_history.append(mag_db[:N//2])
        spec_array = np.array(spec_history).T  # shape (freq_bins, time_history)
        spec_img.set_data(spec_array)
        # robust dynamic range: ignore extreme outliers
        try:
            spec_img.set_clim(np.percentile(spec_array, [5, 95]))
        except Exception:
            pass

        freq_history.append(main_freq)
        # use integer frame indices for track x axis
        time_history.append(time_history[-1] + 1)
        track_line.set_data(list(time_history), smooth(list(freq_history), 5))
        ax_track.relim(); ax_track.autoscale_view()

        # --- Info window update ---
        info_text.set_text(
            f"Main Frequency : {main_freq:8.2f} Hz\n"
            f"Main Amplitude : {main_amp:8.4f} V\n"
            f"2nd Frequency  : {sec_freq:8.2f} Hz\n"
            f"2nd Amplitude  : {sec_amp:8.4f} V\n"
            f"SNR (est)      : {snr_db:8.2f} dB\n"
            f"RMS Voltage    : {rms:8.4f} V\n"
            f"Filter Enabled : {filter_enabled}"
        )
        fig_info.canvas.draw_idle()

        # --- FPS update ---
        frame_count += 1
        now = time.time()
        if now - prev_time >= 1.0:
            fps_display = frame_count / (now - prev_time)
            frame_count = 0
            prev_time = now
            fig.suptitle(f"Main: {main_freq:.1f} Hz | 2nd: {sec_freq:.1f} Hz | SNR: {snr_db:.1f} dB | FPS: {fps_display:.1f}")
        plt.pause(0.001)

    except KeyboardInterrupt:
        print("🛑 Exiting...")
        break
    except Exception as e:
        print("⚠️ Frame skipped:", e)
        continue
