# **IMPORTANT NOTE**
Do not clone this repository, simply follow the instructions below

# GRS Feet Sensor

Github Repository for GRS Feet Sensor

## Installation

Create new folder in raspberry pi
```bash
mkdir sensor
cd sensor
```

Clone [grove.py](https://github.com/Seeed-Studio/grove.py) from GitHub and go into directory

```bash
git clone https://github.com/Seeed-Studio/grove.py.git
```
```
cd grove.py
```

Now clone the grove sensor into pytorch

```bash
wget -nc https://raw.githubusercontent.com/GiudaJude/Pi-GRS-Sensor/refs/heads/main/sensor.py
```

## Usage

Use the number of the port you are connected to.

```python
python sensor.py 0

```

## Running Device Main

To test the device's main functionality, use the following command:

```bash
python device_main.py 0 --model "ML Testing/lda_full_pipeline.joblib" --fs 4.0 --window 100 --step 30 
```

- Replace `<channel_number>` with the ADC channel number your GRS sensor is connected to.
- The `--model` flag specifies the path to the complete joblib or compact LDA model file.
- `--fs` sets the sampling rate in Hz (default: 4 Hz).
- `--window` specifies the window length in seconds (default: 8 seconds).
- `--step` sets the step length in seconds (default: 4 seconds).

For example, if your sensor is connected to channel 0, you can run:

```bash
python device_main.py 0 --model "ML Testing/lda_full_pipeline.joblib" --fs 4.0 --window 100 --step 30 
```




