import tkinter as tk
import os.path
from threading import Timer

import h5py
import numpy as np
from PIL import Image, ImageTk

class App(tk.Frame):

    def __init__(self, master, player_detection_in, line_annotation_in, video):
        tk.Frame.__init__(self, master)
        self.master.title("Rugby Tracking")
        self.master.resizable(False, False)

        self.lines = False
        self.footage_detections = False
        self.homographies = False
        self.filters = True
        self.should_play = False

        # Define Footage
        self.footage_img = Image.new('RGB', (1280, 720))
        self.footage_img_tk = ImageTk.PhotoImage(self.footage_img)
        self.footage_lbl = tk.Label(self, image=self.footage_img_tk)
        self.footage_lbl.grid(column=0, row=0, rowspan=4)

        # Define Pitch
        self.pitch_img = Image.new('RGB', (600, 350))
        self.pitch_img_tk = ImageTk.PhotoImage(self.pitch_img)
        self.pitch_lbl = tk.Label(self, image=self.pitch_img_tk)
        self.pitch_lbl.grid(column=1, row=0, sticky='N', columnspan=2)

        # Define Options
        self.options = tk.Frame(self)
        self.options.grid(column=1, row=1, sticky='nesw', padx=10, rowspan=2)
        tk.Label(self.options, text='Toggles:', font='Helvetica 18 bold').pack(anchor='n')

        self.lines_btn = tk.Button(self.options, text='Show Footage Lines', font='Helvetica 14', command=self.toggle_lines)
        self.lines_btn.pack(fill='both', anchor='w')

        self.detect_btn = tk.Button(self.options, text='Show Footage Detections', font='Helvetica 14', command=self.toggle_detect)
        self.detect_btn.pack(fill='both', anchor='w')

        self.homog_btn = tk.Button(self.options, text='Show Homography Translations', font='Helvetica 14', command=self.toggle_homog)
        self.homog_btn.pack(fill='both', anchor='w')

        self.filter_btn = tk.Button(self.options, text='Show Particle Filter', font='Helvetica 14', command=self.toggle_part_filter)
        self.filter_btn.pack(fill='both', anchor='w')

        self.changedLabel = tk.Label(self, text='SCENE CHANGED', fg='red', font=('Courier', 38))

        # Homography Display
        self.homog_frame = tk.Label(self)
        self.homog_frame.grid(column=2, row=1, sticky='new')
        tk.Label(self.homog_frame, text='Homography Matrix:', font='Helvetica 18 bold').pack(anchor='n')
        self.homog_text = tk.Label(self.homog_frame, text='')
        self.homog_text.pack()

        self.players_and_frames = tk.Label(self, text='Number of active filters: \nFrame:', font='Helvetica 14', justify='left')
        self.players_and_frames.grid(column=2, row=2, sticky='new')

        # Pause/Play Button
        self.play_btn = tk.Button(self, text=' > Play ', font='Helvetica 18 bold', command=self.toggle_play)
        self.play_btn.grid(column=1, row=3, sticky='ew', padx=10, columnspan=2)

        np.set_printoptions(precision=4)

    def toggle_part_filter(self):
        self.filters = not self.filters
        self.filter_btn.configure(text='Hide Particle Filter' if self.filters else 'Show Particle Filter')

    def toggle_lines(self):
        self.lines = not self.lines
        self.lines_btn.configure(text='Hide Footage Lines' if self.lines else 'Show Footage Lines')

    def toggle_detect(self):
        self.footage_detections = not self.footage_detections
        self.detect_btn.configure(text='Hide Footage Detections' if self.footage_detections else 'Show Footage Detections')

    def toggle_homog(self):
        self.homographies = not self.homographies
        self.homog_btn.configure(text='Hide Homography Translations' if self.homographies else 'Show Homography Translations')

    def toggle_play(self):
        self.should_play = not self.should_play
        self.play_btn.configure(text=' || Pause ' if self.should_play else ' > Play ')
        #self.master.after(1, self.update_frames)

    def hide_change_lbl(self):
        self.changedLabel.grid_remove()


if __name__ == '__main__':
    root = tk.Tk()

    dirname = os.path.dirname(__file__)
