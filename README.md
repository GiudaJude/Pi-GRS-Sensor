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




