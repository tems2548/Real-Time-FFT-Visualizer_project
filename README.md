# 🎛️ Real-Time FFT Visualizer

A Python-based **real-time frequency analyzer** for ESP32 (or any microcontroller) sending sampled ADC data.  
It performs **FFT, RMS, THD, and SNR** analysis, displaying multiple live plots — time waveform, spectrum, spectrogram, and frequency tracking.

---

## 🧠 Overview

This project converts **ADC samples → frequency domain** in real time.  
It helps visualize and measure signal characteristics such as:
- Fundamental frequency  
- Signal-to-noise ratio (SNR)  
- RMS voltage  
- Time–frequency evolution (spectrogram)

---

## ⚙️ System Flow

```
Analog Signal → ESP32 ADC → Serial → Python (NumPy + Matplotlib)
          ↓                         ↓
     Sampling                FFT Analysis
          ↓                         ↓
     Voltage Data      →    Spectrum + Metrics
```

---

## 🧩 Specifications

| Parameter | Symbol | Typical Value | Description |
|:-----------|:--------|:--------------|:-------------|
| Sampling frequency | F_s | 18.86 kHz | Sampling rate of ESP32 |
| Samples per frame | N | 1024 | FFT window size |
| Frequency resolution | Δf = F_s/N | 18.4 Hz | Bin width |
| ADC resolution | – | 12-bit (0–4095) | |
| Voltage reference | V_ref | 3.3 V | ADC reference |
| Max measurable frequency | f_Nyquist = F_s/2 | 9.43 kHz | |

---

## 🧮 Mathematical Foundations

![App Screenshot](https://cdn.discordapp.com/attachments/816657587986104331/1432323294455136337/image.png?ex=6900a284&is=68ff5104&hm=83bfff234932b5a111a879b5f4fe4393852601254813da17029482d4b1dbe8e4&)

---

## 🧰 Implementation Outline

```python
import serial
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

PORT = "/dev/ttyACM0"
BAUD = 1000000
N = 1024
FS = 18860.0
HEADER = b'\xCD\xAB'
VREF = 3.3
```

---

## 📊 Visualization

| Plot | X-axis | Y-axis | Description |
|------|---------|---------|--------------|
| Time Domain | Time (s) | Voltage (V) | Raw signal waveform |
| FFT Spectrum | Frequency (Hz) | Magnitude (dB) | Amplitude vs frequency |
| Spectrogram | Frame index | Frequency (Hz) | Time–frequency heatmap |
| Frequency Tracking | Frame index | Frequency (Hz) | Fundamental over time |
| Text Panel | – | – | Shows RMS, THD, SNR |

---

## 📏 Example Results

| Metric | Symbol | Example Value |
|:--|:--|:--|
| RMS Voltage | V_RMS | 0.707 V |
| Fundamental Frequency | f₁ | 1000 Hz |
| 2nd Harmonic | f₂ | 2000 Hz |
| 3rd Harmonic | f₃ | 3000 Hz |
| THD | – | 0.45% |
| SNR | – | 54.2 dB |

---

## 🔧 Dependencies

```bash
pip install numpy matplotlib pyserial psutil
```

---

## 🚀 How to Run

1. Connect ESP32 (or any board sending 1024 ADC samples with header 0xCDAB)  
2. Adjust PORT and BAUD in the script  
3. Run:
```bash
python fft_visualizer.py
```

---

## 🧰 Future Improvements

- Add amplitude calibration for Hanning window  
- Add noise floor averaging for more accurate SNR  
- Support multiple input channels (stereo FFT)  
- Add waterfall 3D plot (matplotlib or pyqtgraph)

---

## 🧠 Arduino Data Acquisition

### Overview
The ESP32 continuously samples analog input from **ADC1 Channel 3** (`GPIO 39`) at a variable sampling rate determined by the loop execution time. Each acquisition cycle collects **N = 1024 samples** and sends them as a binary frame to the Python visualizer over USB.

### ADC Configuration
- **Resolution:** 12 bits → values from 0–4095  
- **Voltage Reference (VREF):** 3.3 V  
- **Input Attenuation:** `ADC_ATTEN_DB_11` (≈ 0–3.6 V range)  

**Voltage conversion equation:**
\[
V_{in} = \frac{ADC_{raw}}{4095} \times V_{REF}
\]

### Sampling Rate Measurement
Sampling frequency \( F_s \) is computed in real time based on the time it takes to sample 1024 points:

\[
F_s = \frac{N}{T_{elapsed}} = \frac{N}{(end - start) / 10^6} = \frac{N \times 10^6}{elapsed_{us}}
\]

This ensures exact FFT frequency scaling, compensating for CPU timing variations.

### Frame Structure (Serial Transmission)
| Field | Size (bytes) | Description |
|-------|---------------|-------------|
| Header | 2 | 0xABCD — used for synchronization |
| ADC samples | 2048 | 1024 samples × 2 bytes |
| Sampling frequency | 4 | `float` value of actual Fs |

**Total = 2054 bytes per frame**

### Example Binary Stream Layout
```
[CD][AB] [s0_lo][s0_hi] [s1_lo][s1_hi] ... [s1023_lo][s1023_hi] [Fs_byte0]...[Fs_byte3]
```

### Timing Notes
- `micros()` used to measure total sampling duration.
- Wrap-around handled with `0xFFFFFFFFUL` correction.
- Minimum `elapsed_us` clamped to avoid division by zero.
- Typical Fs ≈ 18.8 kHz on ESP32-S3/C6 at 240 MHz CPU.

### FFT Scaling Impact
Accurate \( F_s \) ensures the FFT frequency bins are correct:
\[
f_k = \frac{k \cdot F_s}{N}, \quad k = 0, 1, 2, \ldots, \frac{N}{2}
\]

This value is used by the Python visualizer to label spectrum peaks and compute THD correctly.
