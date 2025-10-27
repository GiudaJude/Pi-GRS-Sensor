import math
import sys
import time
import matplotlib.pyplot as plt
from grove.adc import ADC


class GroveGSRSensor:
    def __init__(self, channel):
        self.channel = channel
        self.adc = ADC()

    @property
    def GSR(self):
        value = self.adc.read(self.channel)
        return value


def main():
    if len(sys.argv) < 2:
        print('Usage: {} adc_channel'.format(sys.argv[0]))
        sys.exit(1)

    sensor = GroveGSRSensor(int(sys.argv[1]))

    print('Detecting GSR...')
    plt.ion()  # turn on interactive mode
    fig, ax = plt.subplots()
    x_data, y_data = [], []
    line, = ax.plot([], [], color='blue')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('GSR Value')
    ax.set_title('Real-Time GSR Sensor Readings')
    ax.set_ylim(0, 1024)  # Adjust based on your ADC range
    start_time = time.time()

    while True:
        value = sensor.GSR
        elapsed = time.time() - start_time
        print(f'GSR value: {value}')

        x_data.append(elapsed)
        y_data.append(value)

        # Keep only the last 100 readings for smoother display
        if len(x_data) > 100:
            x_data = x_data[-100:]
            y_data = y_data[-100:]

        line.set_xdata(x_data)
        line.set_ydata(y_data)
        ax.set_xlim(x_data[0], x_data[-1])
        plt.draw()
        plt.pause(0.1)  # update every 100 ms

        time.sleep(0.5)  # adjust to match your sampling speed


if __name__ == '__main__':
    main()
