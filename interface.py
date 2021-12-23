from prediction import *
import tkinter as tk
from tkinter import IntVar
from PIL import ImageGrab, ImageOps


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Handwritten digits recognition")
        self.clear_canvas_butt = tk.Button(self, text="Clear", width=8, height=1, command=self.clear_canvas)
        self.start_rec_butt = tk.Button(self, text="Recognize", width=8, height=1, command=self.predict)
        self.draw_field = tk.Canvas(self, width=400, height=400, bg="white", cursor="cross")
        self.prediction_out = tk.Label(self, width=10, font="Arial 40", justify="left")
        self.pen_size_scale = tk.Scale(self, from_=1, to=10, command=self.pen_size_change)
        self.pen_size = IntVar()
        self.size_label = tk.Label(self, text="Pen size")
        self.draw_field.bind("<B1-Motion>", self.draw)

        self.clear_canvas_butt.grid(column=2, row=2)
        self.start_rec_butt.grid(column=3, row=2)
        self.draw_field.grid(column=2, row=1)
        self.prediction_out.grid(column=3, row=1)
        self.size_label.grid(column=1, row=2)
        self.pen_size_scale.grid(column=1, row=1)

    def pen_size_change(self, val):
        v = int(float(val))
        self.pen_size.set(v)

    def draw(self, event):
        x, y = event.x, event.y
        r = 6+self.pen_size.get()
        self.draw_field.create_oval(x+r, y+r, x-r, y-r, fill="black")

    def clear_canvas(self):
        self.draw_field.delete("all")

    def predict(self):
        x = self.winfo_rootx() + self.draw_field.winfo_x()
        y = self.winfo_rooty() + self.draw_field.winfo_y()
        x1 = x + self.draw_field.winfo_width()
        y1 = y + self.draw_field.winfo_height()
        image = ImageGrab.grab().crop((x, y, x1, y1))
        image = ImageOps.invert(image)
        image = image.resize((28, 28))
        image = image.convert('L')
        # image.show()
        digit = predict_digit(image)
        self.prediction_out.configure(text=f'Prediction: {digit}')
