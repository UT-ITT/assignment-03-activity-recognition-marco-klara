# this program gathers sensor data

import socket
import time
import DIPPID
import pandas
import sys

# Use UDP
PORT = 5700
sensor = DIPPID.SensorUDP(PORT)

# timestamp variable
timestamp = 0

# button press variable
button_pressed = False

# variables for sensor data
acc_x, acc_y, acc_z = 0, 0, 0
gyro_x, gyro_y, gyro_z = 0, 0, 0

# define how to handle acceleration data
def handle_acceleration(data):
    global acc_x, acc_y, acc_z

    if (sensor.has_capability('accelerometer')):

        # get acceleration values of every axis
        acc_x = sensor.get_value('accelerometer')['x']
        acc_y = sensor.get_value('accelerometer')['y']
        acc_z = sensor.get_value('accelerometer')['z']

sensor.register_callback('accelerometer', handle_acceleration)

# define how to handle gyro data
def handle_gyro(data):
    global gyro_x, gyro_y, gyro_z

    if (sensor.has_capability('gyroscope')):

        # get gyro values of every axis
        gyro_x = sensor.get_value('gyroscope')['x']
        gyro_y = sensor.get_value('gyroscope')['y']
        gyro_z = sensor.get_value('gyroscope')['z']

sensor.register_callback('gyroscope', handle_gyro)


# define how to handle button_1 press
def handle_button_1(data):
    global button_pressed

    if (sensor.has_capability('button_1')):
        if int(data) == 1:
            button_pressed = True

sensor.register_callback('button_1', handle_button_1)

# Main loop to collect data at 100 Hz for 10 seconds (1000 data points)
def main():

    print("Press Button_1 to start recording")
    while (button_pressed != True):
        time.sleep(0.1)

    id = 0 # id of every sample
    data = []  # List to store data 
    duration = 10  # 10 seconds
    interval = 0.01  # 100 Hz -> 0.01 seconds per sample
    samples = int(duration / interval)  # in this case should be 1000

    print("Collecting Data for 10 seconds:")

    for i in range(samples):
        # Get time of the sample
        timestamp = time.time()

        # Collect sensor values of this step
        data_point = {
            'id' : id,
            'timestamp': timestamp,
            'acc_x': acc_x,
            'acc_y': acc_y,
            'acc_z': acc_z,
            'gyro_x': gyro_x,
            'gyro_y': gyro_y,
            'gyro_z': gyro_z
        }
        data.append(data_point)
        id += 1

        # Wait for next sample
        time.sleep(interval)

    print(f"Collected {len(data)} data points.")

    # Save as CSV file
    filename = 'data/marcoschneider/marcoschneider-rowing-4.csv' 
    df = pandas.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

    # disconnect sensor and stop program
    sensor.disconnect()
    sys.exit(0)

if __name__ == "__main__":
    main()