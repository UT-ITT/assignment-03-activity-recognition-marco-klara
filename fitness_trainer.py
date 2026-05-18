# this program visualizes activities with pyglet

import pyglet
from pathlib import Path
from pyglet import window
import random
import math
import time
import sys
import pandas as pd
import activity_recognizer as activity

import threading
from collections import deque

from sklearn.metrics import accuracy_score

from gather_data import interval, sensor, handle_acceleration, handle_gyro
import gather_data

# variables for sensor data collection thread
buffer = deque(maxlen = 100)
lock = threading.Lock()

# variables for sensor data
acc_x, acc_y, acc_z = 0, 0, 0
gyro_x, gyro_y, gyro_z = 0, 0, 0

# handle gyroscope and accelerometer data from the DIPPID device
sensor.register_callback('gyroscope', handle_gyro)
sensor.register_callback('accelerometer', handle_acceleration)

# Path variables for the csv data folder
THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR / "data"

# variables for prediction model
trained_model = None
model_ready = False
current_prediction = None

# Pyglet variables
config = pyglet.gl.Config(double_buffer=True)

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
TOP_MARGIN = 160
BOTTOM_MARGIN = 150
MAX_SCALE = 0.7

win = window.Window(WINDOW_WIDTH, WINDOW_HEIGHT, config=config)
pyglet.gl.glClearColor(0.83, 0.83, 0.83, 1.0)

# list of activities and the path to their visualization
activities = {
    "running": ["img/running_1.png", "img/running_2.png"],
    "rowing": ["img/rowing_1.png", "img/rowing_2.png"],
    "lifting": ["img/lifting_1.png", "img/lifting_2.png"],
    "jumpingjacks": ["img/jumpingjack_1.png", "img/jumpingjack_2.png"],
}

# variables for changing activity functionality 
current_activity = None
frames = []
frame_index = 0
sprite = None
timer_running = True
remaining_time = 10.0

# text labels
waiting_1 = pyglet.text.Label("Prepare yourself,",
                          font_name='Calibri',
                          font_size=70,
                          color=(0, 0, 0, 255),
                          weight = 'ultrabold',
                          x=WINDOW_WIDTH//2, y=WINDOW_HEIGHT//2+150,
                          anchor_x='center', anchor_y='center')

waiting_2 = pyglet.text.Label("training starts every moment!",
                          font_name='Calibri',
                          font_size=70,
                          color=(0, 0, 0, 255),
                          weight = 'ultrabold',
                          x=WINDOW_WIDTH//2, y=WINDOW_HEIGHT//2-150,
                          anchor_x='center', anchor_y='center')

motivation = pyglet.text.Label("Keep going!",
                          font_name='Calibri',
                          font_size=70,
                          color=(0, 0, 0, 255),
                          weight = 'ultrabold',
                          x=WINDOW_WIDTH//2, y=100,
                          anchor_x='center', anchor_y='center')

# label for timer
timer = pyglet.text.Label(str(math.ceil(remaining_time)),
                          font_name='Calibri',
                          font_size=100,
                          color=(255, 0, 0, 255),
                          weight = 'ultrabold',
                          x=WINDOW_WIDTH//2, y=WINDOW_HEIGHT-100,
                          anchor_x='center', anchor_y='center')

# function to scale the image to fit into the frame
def fit_sprite_to_window(activity_name, top_margin=TOP_MARGIN, bottom_margin=BOTTOM_MARGIN, max_scale=MAX_SCALE):
    activity_frames = [pyglet.image.load(path) for path in activities[activity_name]]
    
    max_width = max(img.width for img in activity_frames)
    max_height = max(img.height for img in activity_frames)
    
    available_width = WINDOW_WIDTH - 40
    available_height = WINDOW_HEIGHT - top_margin - bottom_margin
    scale_w = available_width / max_width
    scale_h = available_height / max_height
    return min(scale_w, scale_h, max_scale)

# functio to load the activity specified by activity_name
def load_activity(activity_name):
    global current_activity, frames, frame_index, sprite

    current_activity = activity_name
    frames = [pyglet.image.load(path) for path in activities[activity_name]]
    frame_index = 0

    scale = fit_sprite_to_window(activity_name)
    x = (WINDOW_WIDTH - frames[0].width * scale) / 2
    available_height = WINDOW_HEIGHT - TOP_MARGIN - BOTTOM_MARGIN
    y = BOTTOM_MARGIN + (available_height - frames[0].height * scale) / 2

    if sprite is None:
        sprite = pyglet.sprite.Sprite(frames[0], x=x, y=y)
    else:
        sprite.image = frames[0]
        sprite.x = x
        sprite.y = y

    sprite.scale = scale

# function to switch to the next frame for the activity animation
def next_frame(dt):
    global frame_index, sprite
    if not frames or sprite is None:
        return 
    
    frame_index = 1 - frame_index
    new_img = frames[frame_index]
    sprite.image = new_img

    img_w = new_img.width * sprite.scale
    img_h = new_img.height * sprite.scale
    sprite.x = (WINDOW_WIDTH - img_w) / 2
    available_height = WINDOW_HEIGHT - TOP_MARGIN - BOTTOM_MARGIN
    sprite.y = BOTTOM_MARGIN + (available_height - img_h) / 2

# function to switch to the next activity
def switch_activity(dt=None): 
    activity_name = random.choice(list(activities.keys()))
    load_activity(activity_name)

# function to collect sensor data
def sensor_loop():
    global interval

    while True:
        # Collect sensor values
        data_point = {
            'acc_x': gather_data.acc_x,
            'acc_y': gather_data.acc_y,
            'acc_z': gather_data.acc_z,
            'gyro_x': gather_data.gyro_x,
            'gyro_y': gather_data.gyro_y,
            'gyro_z': gather_data.gyro_z
        }
        with lock:
            buffer.append(data_point)

        time.sleep(interval)

# function to predict the executed activity based on the phone data
def predict_activity(model):
    if model is None:
        return None
    with lock:
        if len(buffer) < 100: # 100 -> 1s window
            return None
        
        # take 100 last samples (1 second of samples)
        window = list(buffer)[-100:]

        # calculate features for the samples
        df = pd.DataFrame(window)
        df_features = activity.calc_features(df)

        # get prediction for the sample
        prediction = model.predict(pd.DataFrame([df_features]))[0]
        return prediction
    
# function to return model accuracy on an independent test set
def test_model(TEST_PATH):
    global trained_model

    df = activity.csv_feature_extraction(TEST_PATH)

    # select features and targets for prediction
    features = df.drop(columns=["activity", "name"])
    targets = df["activity"]

    predictions = trained_model.predict(features)

    print("Independent test data Accuracy: ", accuracy_score(targets, predictions))

TEST_PATH = THIS_DIR / "test"

# load csv data and train model
def load_model():
    global model_ready
    global trained_model
    global TEST_PATH

    data = activity.csv_feature_extraction(DATA_DIR)
    trained_model = activity.train_model(data)
    model_ready = True
    print("Model trained")
    #test_model(TEST_PATH)

def update(dt):
    global remaining_time, timer_running, current_prediction, current_activity

    current_prediction = predict_activity(trained_model)
  
    # timer runs if prediction == activity
    if current_prediction == current_activity:
        timer_running = True
        remaining_time -= dt
    else:
        timer_running = False

    if remaining_time <= 0:
        switch_activity()
        remaining_time = 10.0

# draw the different components 
@win.event
def on_draw():
    win.clear()

    # wait for model to finish training
    if not model_ready:
        waiting_1.draw()
        waiting_2.draw()

    else:
        timer.text = str(math.ceil(remaining_time))

        if sprite is not None:
            sprite.draw()

        timer.draw()

        if not timer_running:
            motivation.draw()

# function to close the window when esc is pressed
@win.event
def on_key_press(symbol, modifiers):
    if symbol == pyglet.window.key.ESCAPE:
        pyglet.app.exit()
        gather_data.sensor.disconnect()

# start pyglet
switch_activity(0)
pyglet.clock.schedule_interval(next_frame, 0.5)
pyglet.clock.schedule_interval(update, 1)

# start thread to run model training without stalling main loop
threading.Thread(target = load_model, daemon=True).start()

# start thread to run pyglet and sensor loop simultaneously
threading.Thread(target = sensor_loop, daemon=True).start()

pyglet.app.run()


