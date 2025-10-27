import sys
import time
import statistics
import matplotlib.pyplot as plt
from grove.adc import ADC


class GroveGSRSensor:
    def __init__(self, channel):
        self.channel = channel
        self.adc = ADC()

    @property
    def GSR(self):
        return self.adc.read(self.channel)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} adc_channel")
        sys.exit(1)

    sensor = GroveGSRSensor(int(sys.argv[1]))

    print("Monitoring GSR... (press Ctrl+C to stop)")
    interval = 0.1       # sample every 0.1 seconds
    window_duration = 150 # compute average every 60 seconds

    start_time = time.time()
    window_start = start_time
    window_values = []
    averages = []
    minute_marks = []

    try:
        while True:
            value = sensor.GSR
            now = time.time()
            window_values.append(value)
            time.sleep(interval)

            # Check if 60 seconds have passed
            if (now - window_start) >= window_duration:
                avg = statistics.mean(window_values)
                elapsed_minutes = (now - start_time) / 60
                averages.append(avg)
                minute_marks.append(elapsed_minutes)
                print(f"[{elapsed_minutes:5.2f} min] Average GSR: {avg:.2f}")

                # reset window
                window_start = now
                window_values = []

    except KeyboardInterrupt:
        print("\nStopping data collection...")
        print(f"Collected {len(averages)} averaged data points.")

        # --- Show final graph ---
        plt.figure()
        plt.plot(minute_marks, averages, marker='o', color='blue')
        plt.title("Average GSR per 60 Seconds")
        plt.xlabel("Time (minutes)")
        plt.ylabel("Average GSR Value")
        plt.grid(True)
        plt.show()

        # --- Optional: Save to CSV ---
        save = input("Save data to CSV file? (y/n): ").strip().lower()
        if save == "y":
            with open("gsr_averages.csv", "w") as f:
                f.write("minute,avg_gsr\n")
                for t, v in zip(minute_marks, averages):
                    f.write(f"{t:.2f},{v:.2f}\n")
            print("Data saved to gsr_averages.csv")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
