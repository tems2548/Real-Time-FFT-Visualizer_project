"""
fft_analyzer_s3_full_with_autocal.py

Expect frames from ESP32-S3 Arduino sketch that send:
  [2 bytes HEADER] + [N * uint16_t (millivolts)] + [4-byte float FS_actual]

How to use:
 - Set PORT to your serial port
 - Upload the Arduino sketch that sends calibrated millivolts (esp_adc_cal_raw_to_voltage)
 - Run this script. Click:
     - "💾 Save CSV" to save a snapshot of current FFT (freq,dB)
     - "🎚 Toggle Filter" to enable/disable band-pass
     - "🔧 Auto-Cal" to auto-calibrate FFT amplitude to match time-domain RMS
     - "↺ Reset Cal" to reset calibration factor to 1.0
"""

import serial
import struct
import time
import csv
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.signal import butter, filtfilt

# ----------------- Settings -----------------
PORT =  "/dev/ttyACM0"   # <-- change this to your port (e.g. "/dev/ttyACM0")
BAUD = 1000000
N = 1024
HEADER = b'\xCD\xAB'
HISTORY_LEN = 80
TRACK_LEN = 150
FRAME_SIZE = 2 + N * 2 + 4  # header + N*uint16 + float FS
MIN_FREQ = 5.0
EXCLUDE_BINS = 5
ADC_MV_MAX = 3300.0  # millivolt full-scale reference (approx)
# initial calibration factor (applied to volts after conversion)
CAL_FACTOR = 1.0

# ----------------- Serial -----------------
ser = serial.Serial(PORT, BAUD, timeout=0.1)

# ----------------- Plot setup -----------------
plt.ion()
fig, axs = plt.subplots(2, 2, figsize=(13, 8))
fig.canvas.manager.set_window_title("ESP32-S3 FFT Analyzer (mV frames)")

ax_time, ax_fft, ax_spec, ax_track = axs.flatten()

# time domain
time_line, = ax_time.plot([], [], lw=0.9)
ax_time.set_title("Time Domain")
ax_time.set_xlabel("Time [s]")
ax_time.set_ylabel("Voltage [V]")
ax_time.grid(True)

# FFT
fft_line, = ax_fft.plot([], [], lw=1.0)
peak_marker, = ax_fft.plot([], [], 'go', markersize=8, label="Main Peak")
sec_marker, = ax_fft.plot([], [], 'ro', markersize=6, label="2nd Peak")
ax_fft.set_title("FFT Spectrum")
ax_fft.set_xlabel("Frequency [Hz]")
ax_fft.set_ylabel("Magnitude [dB]")
ax_fft.legend(loc="upper right")
ax_fft.grid(True)

# spectrogram
spec_img = ax_spec.imshow(np.zeros((N//2, HISTORY_LEN)), aspect='auto', origin='lower',
                          extent=[0, HISTORY_LEN, 0, 8000], cmap='inferno', vmin=-100, vmax=0)
ax_spec.set_title("Spectrogram")
ax_spec.set_xlabel("Frame Index")
ax_spec.set_ylabel("Frequency [Hz]")

# tracking
track_line, = ax_track.plot([], [], lw=1.2)
ax_track.set_title("Main Frequency Tracking")
ax_track.set_xlabel("Frame")
ax_track.set_ylabel("Frequency [Hz]")
ax_track.grid(True)

# Buttons (save, toggle filter, auto-cal, reset cal)
button_ax_save = plt.axes([0.83, 0.02, 0.13, 0.05])
save_button = Button(button_ax_save, '💾 Save CSV', color='lightgray', hovercolor='lightgreen')

button_ax_filter = plt.axes([0.68, 0.02, 0.13, 0.05])
filter_button = Button(button_ax_filter, '🎚 Toggle Filter', color='lightgray', hovercolor='lightblue')

button_ax_autocal = plt.axes([0.53, 0.02, 0.13, 0.05])
autocal_button = Button(button_ax_autocal, '🔧 Auto-Cal', color='lightgray', hovercolor='lightyellow')

button_ax_resetcal = plt.axes([0.38, 0.02, 0.13, 0.05])
resetcal_button = Button(button_ax_resetcal, '↺ Reset Cal', color='lightgray', hovercolor='lightsalmon')

filter_enabled = True

def toggle_filter(event):
    global filter_enabled
    filter_enabled = not filter_enabled
    print(f"🔧 Band-pass filter: {'ON' if filter_enabled else 'OFF'}")
filter_button.on_clicked(toggle_filter)

# ----------------- Buffers -----------------
spec_history = deque([np.zeros(N//2) for _ in range(HISTORY_LEN)], maxlen=HISTORY_LEN)
freq_history = deque([0]*TRACK_LEN, maxlen=TRACK_LEN)
time_history = deque(list(range(-TRACK_LEN+1, 1)), maxlen=TRACK_LEN)

prev_time = time.time()
frame_count = 0

# Info window
fig_info, ax_info = plt.subplots(figsize=(5, 3))
fig_info.canvas.manager.set_window_title("Live FFT Data")
ax_info.axis("off")
info_text = ax_info.text(0.01, 0.99, "", fontsize=11, va="top", family="monospace")

# ----------------- Helper functions -----------------
def safe_bandpass(data, fs, lowcut=10.0, highcut=4500.0, order=4):
    """Apply bandpass filter with safety guards."""
    data = np.asarray(data)
    if len(data) < 16:
        return data
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = min(highcut / nyq, 0.99)
    if not (0 < low < high < 1):
        return data
    try:
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)
    except Exception as e:
        print("⚠️ Filter failed:", e)
        return data

def parabolic_interpolation_linear(mag_lin, idx):
    """Return fractional bin offset for peak interpolation on linear-mag."""
    if idx <= 0 or idx >= len(mag_lin) - 1:
        return 0.0
    a = mag_lin[idx-1]
    b = mag_lin[idx]
    c = mag_lin[idx+1]
    denom = (a - 2*b + c)
    if denom == 0:
        return 0.0
    return 0.5 * (a - c) / denom

def compute_fft_and_peaks(volt, fs):
    """
    Returns:
      freq_axis_full, mag_db_full,
      main_freq_hz, main_amp_lin,
      second_freq_hz, second_amp_lin,
      rms_time, snr_db
    """
    volt = np.asarray(volt)
    volt = volt - np.mean(volt)
    if filter_enabled:
        volt = safe_bandpass(volt, fs)

    # ensure length N
    if len(volt) != N:
        volt = np.resize(volt, N)

    window = np.hanning(N)
    fft_vals = np.fft.rfft(volt * window)
    # amplitude normalization: correct for window energy
    window_corr = np.sum(window)/N
    mag = (2.0 / N) * np.abs(fft_vals) / (window_corr + 1e-16)   # linear peak amplitude approx
    mag_db = 20.0 * np.log10(mag + 1e-12)
    freq_axis = np.fft.rfftfreq(N, 1.0/fs)

    mask = freq_axis > MIN_FREQ
    if not np.any(mask):
        # safe defaults
        return freq_axis, mag_db, 0.0, 0.0, 0.0, 0.0, np.sqrt(np.mean(volt**2)), np.nan

    mag_valid = mag[mask]
    mag_db_valid = mag_db[mask]
    freq_valid = freq_axis[mask]

    main_idx = int(np.argmax(mag_valid))
    # parabolic interpolation on linear magnitudes
    delta = parabolic_interpolation_linear(mag_valid, main_idx)
    df = fs / N
    main_freq = freq_valid[main_idx] + delta * df
    main_amp = mag_valid[main_idx]

    # exclude neighborhood for second peak
    mag_db_copy = mag_db_valid.copy()
    start = max(main_idx - EXCLUDE_BINS, 0)
    end = min(main_idx + EXCLUDE_BINS + 1, len(mag_db_copy))
    mag_db_copy[start:end] = -np.inf
    second_idx = int(np.argmax(mag_db_copy))
    second_freq = freq_valid[second_idx] if len(freq_valid)>0 else 0.0
    second_amp = mag_valid[second_idx] if len(mag_valid)>0 else 0.0

    rms_time = np.sqrt(np.mean(volt**2))
    snr_db = 20.0 * np.log10((main_amp + 1e-12)/(second_amp + 1e-12)) if second_amp>0 else np.nan

    return freq_axis, mag_db, main_freq, main_amp, second_freq, second_amp, rms_time, snr_db

# ----------------- CSV save -----------------
current_freq_axis = np.zeros(N//2+1)
current_mag_db = np.zeros(N//2+1)

def save_to_csv(event):
    filename = f"fft_snapshot_{int(time.time())}.csv"
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Frequency (Hz)", "Magnitude (dB)"])
        for f_hz, mag_db_val in zip(current_freq_axis, current_mag_db):
            writer.writerow([f_hz, mag_db_val])
    print("✅ Saved FFT data to", filename)

save_button.on_clicked(save_to_csv)

# ----------------- Calibration handlers -----------------
def autocal_callback(event):
    """
    Auto-calibration:
      Compute Vrms_from_fft (main_amp/sqrt(2)) and Vrms_time (time-domain).
      Calculate cal = Vrms_time / Vrms_from_fft and apply to global CAL_FACTOR.
      This makes FFT peak-based Vrms match time-domain Vrms (helps normalization).
    """
    global CAL_FACTOR
    if last_volt is None or last_main_amp is None:
        print("⚠️ No valid frame yet to calibrate.")
        return
    # Vrms from fft: convert main_amp (peak) to Vrms
    vrms_fft = last_main_amp / np.sqrt(2.0)
    vrms_time = last_rms_time
    if vrms_fft <= 0:
        print("⚠️ Invalid FFT Vrms for calibration.")
        return
    new_cal = (vrms_time / (vrms_fft + 1e-12))
    # apply smoothing to avoid huge jumps
    CAL_FACTOR = float(CAL_FACTOR * 0.3 + new_cal * 0.7)
    print(f"🔧 Auto-cal applied: new CAL_FACTOR={CAL_FACTOR:.6f} (raw factor {new_cal:.6f})")

autocal_button.on_clicked(autocal_callback)

def resetcal_callback(event):
    global CAL_FACTOR
    CAL_FACTOR = 1.0
    print("↺ Calibration factor reset to 1.0")

resetcal_button.on_clicked(resetcal_callback)

# ----------------- Main loop -----------------
last_volt = None
last_main_amp = None
last_rms_time = None

print("Waiting for data on", PORT, " — press Ctrl+C to exit.")
while True:
    try:
        frame = ser.read(FRAME_SIZE)
        if len(frame) != FRAME_SIZE:
            # sometimes partial reads; keep reading
            continue
        if frame[0:2] != HEADER:
            # alignment lost — try to re-sync: find header inside buffer
            idx = frame.find(HEADER)
            if idx >= 0:
                # seek the serial stream forward to align next frame
                # move serial cursor: read and discard bytes up to idx
                discard = ser.read(idx)
                print("🔁 Resync: dropped", idx, "bytes to find header.")
            continue

        # parse ADC mV values and FS float
        adc_mvs = np.frombuffer(frame[2:2+N*2], dtype=np.uint16)
        FS = struct.unpack('<f', frame[-4:])[0]  # little-endian float from Arduino

        # convert millivolts -> volts and apply global calibration
        volt = (adc_mvs.astype(np.float64) / 1000.0) * CAL_FACTOR

        # compute FFT & peaks
        freq_axis, mag_db, main_freq, main_amp, sec_freq, sec_amp, rms_time, snr_db = compute_fft_and_peaks(volt, FS)

        # Save last frame metrics for autocal
        last_volt = volt.copy()
        last_main_amp = main_amp
        last_rms_time = rms_time

        # store for CSV export
        current_freq_axis = freq_axis
        current_mag_db = mag_db

        # update time plot
        time_line.set_data(np.arange(N) / FS, volt)
        ax_time.relim(); ax_time.autoscale_view()

        # update FFT plot
        fft_line.set_data(freq_axis, mag_db)
        # show markers in dB space; guard values for log
        peak_db = 20.0 * np.log10(main_amp + 1e-12)
        sec_db = 20.0 * np.log10(sec_amp + 1e-12)
        peak_marker.set_data([main_freq], [peak_db])
        sec_marker.set_data([sec_freq], [sec_db])
        ax_fft.relim(); ax_fft.autoscale_view()

        # update spectrogram
        spec_history.append(mag_db[:N//2])
        spec_array = np.array(spec_history).T
        spec_img.set_data(spec_array)
        try:
            spec_img.set_clim(np.percentile(spec_array, [5, 95]))
        except Exception:
            pass

        # update tracking plot
        freq_history.append(main_freq)
        time_history.append(time_history[-1] + 1)
        track_line.set_data(list(time_history), smooth := (np.convolve(list(freq_history), np.ones(5)/5, mode='same')))
        ax_track.relim(); ax_track.autoscale_view()

        # update info window
        info_text.set_text(
            f"FS (measured)   : {FS:8.1f} Hz\n"
            f"Main Frequency   : {main_freq:8.2f} Hz\n"
            f"Main Amplitude   : {main_amp:8.4f} V (peak)\n"
            f"Main Vrms (FFT)  : {main_amp/np.sqrt(2):8.4f} V\n"
            f"Time-domain RMS  : {rms_time:8.4f} V\n"
            f"SNR (est)        : {snr_db:8.2f} dB\n"
            f"Calibration (x)  : {CAL_FACTOR:8.5f}\n"
            f"Filter Enabled   : {filter_enabled}"
        )
        fig_info.canvas.draw_idle()

        # FPS / title update
        frame_count += 1
        now = time.time()
        if now - prev_time >= 1.0:
            fps = frame_count / (now - prev_time)
            frame_count = 0
            prev_time = now
            fig.suptitle(f"Main: {main_freq:.1f} Hz | 2nd: {sec_freq:.1f} Hz | RMS: {rms_time:.3f} V | FS={FS:.1f} Hz | FPS: {fps:.1f}")

        plt.pause(0.001)

    except KeyboardInterrupt:
        print("🛑 Exiting...")
        break
    except Exception as e:
        print("⚠️ Frame skipped:", e)
        # small pause to avoid busy-loop spamming on persistent errors
        time.sleep(0.01)
        continue
