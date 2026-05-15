# this program visualizes activities with pyglet

#import activity_recognizer as activity
import pyglet
from pyglet import window, shapes
import sys
import random
import math

config = pyglet.gl.Config(double_buffer=True)

WINDOW_WIDTH = 1500
WINDOW_HEIGHT = 1000
TOP_MARGIN = 160
BOTTOM_MARGIN = 150
MAX_SCALE = 0.7

win = window.Window(WINDOW_WIDTH, WINDOW_HEIGHT, config=config)
background = pyglet.sprite.Sprite(pyglet.image.load("img/background.jpg"), x=0, y=0)
background.scale = WINDOW_WIDTH / background.image.width  


activities = {
    "running": ["img/running_1.png", "img/running_2.png"],
    "rowing": ["img/rowing_1.png", "img/rowing_2.png"],
    "lifting": ["img/lifting_1.png", "img/lifting_2.png"],
    "jumpingjacks": ["img/jumpingjack_1.png", "img/jumpingjack_2.png"],
}

current_activity = None
frames = []
frame_index = 0
sprite = None
timer_running = True
remaining_time = 10.0

timer = pyglet.text.Label(str(math.ceil(remaining_time)),
                          font_name='Calibri',
                          font_size=100,
                          color=(255, 0, 0, 255),
                          weight = 'ultrabold',
                          x=WINDOW_WIDTH//2, y=WINDOW_HEIGHT-100,
                          anchor_x='center', anchor_y='center')

motivation = pyglet.text.Label("Keep going!",
                          font_name='Calibri',
                          font_size=70,
                          color=(0, 0, 0, 255),
                          weight = 'ultrabold',
                          x=WINDOW_WIDTH//2, y=100,
                          anchor_x='center', anchor_y='center')

current_prediction = "jumpingjacks"


def fit_sprite_to_window(activity_name, top_margin=TOP_MARGIN, bottom_margin=BOTTOM_MARGIN, max_scale=MAX_SCALE):
    activity_frames = [pyglet.image.load(path) for path in activities[activity_name]]
    
    max_width = max(img.width for img in activity_frames)
    max_height = max(img.height for img in activity_frames)
    
    available_width = WINDOW_WIDTH - 40
    available_height = WINDOW_HEIGHT - top_margin - bottom_margin
    scale_w = available_width / max_width
    scale_h = available_height / max_height
    return min(scale_w, scale_h, max_scale)

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

def switch_activity(dt=None): 
    activity_name = random.choice(list(activities.keys()))
    load_activity(activity_name)

def update(dt):
    global remaining_time, timer_running
    if not timer_running:
        return 
    remaining_time -= dt
    if remaining_time <= 0:
        switch_activity()
        remaining_time = 10.0
    
    if current_prediction != current_activity:
        timer_running = False
    else:
        timer_running = True
        

@win.event
def on_draw():
    win.clear() 
    background.draw()

    timer.text = str(math.ceil(remaining_time))

    if sprite is not None:
        sprite.draw()
    timer.draw()

    if not timer_running:
        motivation.draw()

switch_activity(0)
pyglet.clock.schedule_interval(next_frame, 0.5)
pyglet.clock.schedule_interval(update, 0.1)

pyglet.app.run()


