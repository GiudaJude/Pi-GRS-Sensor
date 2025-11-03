import sys
import time
import matplotlib.pyplot as plt
from grove.adc import ADC

# Load GRS Sensor Class
class GroveGSRSensor:
    def __init__(self, channel):
        self.channel = channel
        self.adc = ADC()

    @property
    def GSR(self):
        return self.adc.read(self.channel)


# Adjust hardware setup
ADC_MAX = 1023       # 10-bit ADC
V_REF = 3.3          # Reference voltage (V)
R_FIXED = 100000.0   # Fixed resistor in ohms (100 kΩ typical)
MAX_GSR = 40.0       # Maximum allowed GSR value (μS)

def adc_to_us(adc_value):
    """Convert ADC reading to micro-Siemens (μS)."""
    v_out = (adc_value / ADC_MAX) * V_REF
    if v_out >= V_REF:
        return None 
    try:
        r_skin = (R_FIXED * v_out) / (V_REF - v_out)
        g_siemens = 1.0 / r_skin
        g_microsiemens = g_siemens * 1e6

        # Cap the value at MAX_GSR
        if g_microsiemens > MAX_GSR:
            g_microsiemens = MAX_GSR

        return g_microsiemens
    except ZeroDivisionError:
        return None


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} adc_channel")
        sys.exit(1)

    channel = int(sys.argv[1])
    sensor = GroveGSRSensor(channel)

    print("Monitoring GSR continuously (4 Hz)...")
    print("Press Ctrl+C to stop.\n")
    print("Time(s)\tGSR (μS)")

    interval = 1 / 4.0  # 4 Hz = 0.25 s between samples
    start_time = time.time()
    times = []
    gsr_us_values = []

    try:
        while True:
            adc_value = sensor.GSR
            gsr_us = adc_to_us(adc_value)
            now = time.time() - start_time

            if gsr_us is not None:
                times.append(now)
                gsr_us_values.append(gsr_us)
                print(f"{now:8.2f}\t{gsr_us:8.3f}")

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nStopping data collection...")
        print(f"Collected {len(gsr_us_values)} samples.")

        # --- Plot the entire session ---
        plt.figure(figsize=(10, 5))
        plt.plot(times, gsr_us_values, color='blue', linewidth=1)
        plt.title("Continuous GSR Readings (μS) 4 Hz Sampling")
        plt.xlabel("Time (seconds)")
        plt.ylabel("GSR (μS)")
        plt.grid(True)
        plt.show()

        # --- Optional: Save to CSV ---
        save = input("Save raw data to CSV file? (y/n): ").strip().lower()
        if save == "y":
            with open("gsr_wesad_4hz.csv", "w") as f:
                f.write("time_s,gsr_us\n")
                for t, v in zip(times, gsr_us_values):
                    f.write(f"{t:.3f},{v:.6f}\n")
            print("Data saved to gsr_wesad_4hz.csv")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()