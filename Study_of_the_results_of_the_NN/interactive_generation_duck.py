from matplotlib.figure import Figure
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib import cm

import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from time import time

from scipy.stats import multivariate_normal

import imageio

mean_var_duck = np.load('mean_var_duck.npz')

mean_joint = mean_var_duck["mean_joint"]
var_joint = mean_var_duck["var_joint"]

vae = tf.keras.models.load_model('../Trained_VAE/VAE_full_duck/')

print("Files open successfully")

M = 200
X, Y = np.meshgrid(np.linspace(-25, 25, M), np.linspace(-25, 25, M))
d = np.dstack([X, Y])

gauss = []
Z = []

for i in range(len(mean_joint)):
    gauss.append(multivariate_normal(mean=mean_joint[i], cov=var_joint[i]))

    Z.append(gauss[i].pdf(d).reshape(M, M))
print("Multivariate Normals computed successfully")


# Implement the default Matplotlib key bindings.


root = tk.Tk()
root.wm_title("Generate duck")

fig = Figure(figsize=(5, 4), dpi=100)
subplot = fig.add_subplot(111)

for i in range(len(mean_joint)):
    subplot.contour(X, Y, Z[i], extend='min')
subplot.plot(mean_joint[:, 0], mean_joint[:, 1], color='red')

subplot.axis("equal")

canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

toolbar = NavigationToolbar2Tk(canvas, root)
toolbar.update()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)


def onclick(event):
    subplot.scatter(event.xdata, event.ydata,
                    marker='X', s=200, c="red", zorder=2)
    canvas.draw()
    show_generated_image(event.xdata, event.ydata)


def show_generated_image(x, y):
    z_PI, z_image = tf.split(vae.shared_decoder([[x, y]]),
                             num_or_size_splits=2,
                             axis=1)
    reconstructed_PI = vae.decoder_PI(z_PI)
    reconstructed_image = vae.decoder_image(z_image)
    newWindow = tk.Toplevel(root)
    fig2 = Figure(figsize=(5, 4), dpi=100)
    subplot1 = fig2.add_subplot(121)
    subplot1.imshow(np.reshape(reconstructed_image[0], [128, 128]),
                    vmin=0,
                    vmax=1,
                    cmap='gist_gray')
    subplot1.axis('off')
    subplot1.set_title("New image")

    subplot2 = fig2.add_subplot(122)
    subplot2.imshow(np.flip(np.reshape(reconstructed_PI[0], [
                    50, 50]), 0), cmap='jet', vmin=0, vmax=1)
    subplot2.axis('off')
    subplot2.set_title("New persistence image")
    canvas2 = FigureCanvasTkAgg(fig2, master=newWindow)  # A tk.DrawingArea.
    canvas2.draw()
    canvas2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)


cid = fig.canvas.mpl_connect('button_press_event', onclick)


def _quit():
    root.quit()     # stops mainloop
    root.destroy()  # this is necessary on Windows to prevent
    # Fatal Python Error: PyEval_RestoreThread: NULL tstate


button = tk.Button(master=root, text="Quit", command=_quit)
button.pack(side=tk.BOTTOM)


tk.mainloop()
