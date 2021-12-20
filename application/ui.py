import os
import sys
from tkinter import Button, Event, Frame, Label, StringVar, Tk, ttk
from typing import Callable

import numpy
from camera import set_selected_camera
from PIL import Image, ImageTk

from camera import get_cameras
from pipe import calibrate

video_label_image = numpy.asarray(Image.new('RGB', (100,100), color=(255,1,0)))

def init_tkinter_app() -> Tk:
    global root
    Logo = resource_path("favicon.ico")
    root = Tk()
    root.title('Airpose')
    root.iconbitmap(Logo)
    # Create a frame
    app = Frame(root, bg="white")
    app.grid()

    init_calibrate_button(app, calibrate)
    init_video_output(app)
    init_camera_combobox(app)

    return app

def init_calibrate_button(app: Tk, command: Callable):
    # Create calibration button
    calibration_button = Button(app, text="Calibrate", command=command, width = 50, height = 5, bg = 'green')
    calibration_button.grid(row=2,column=0)

def init_video_output(app: Tk) -> Label:
    # Create a label for video stream
    global video_label
    video_label = Label(app)
    video_label.grid(row=1,column=0,columnspan=1)
    video_label.after(1, update_video_output)
    return video_label

def update_video(image: Image):
    global video_label_image
    video_label_image = image

def update_video_output():
    global video_label_image
    global video_label
    img = Image.fromarray(video_label_image)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    video_label.after(30, update_video_output)

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def init_camera_combobox(app: Tk):
    cameras = get_cameras()
    stringvar = StringVar()
    cb = ttk.Combobox(app, textvariable=stringvar, state= "readonly")
    cb['values'] = list(cameras.keys())
    cb.current(0)
    
    cb.grid(row=0, column=0, columnspan=1, rowspan=1)
    cb.bind("<<ComboboxSelected>>", lambda _: (set_selected_camera(cb.get())))
    