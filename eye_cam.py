#!/usr/bin/python
from __future__ import absolute_import, division, print_function, unicode_literals

""" Example showing what can be left out. ESC to quit"""
import demo
import pi3d
import numpy as np
import picamera
import picamera.array
import threading
import time
import io
from math import cos, sin, radians

SIZE = 64
NBYTES = SIZE * SIZE * 3
threshold = 60 # HSV value below this will be tracked
POS = np.arange(SIZE, dtype=np.float) # list of numbers for finding av. position

npa = np.zeros((SIZE, SIZE, 4), dtype=np.uint8) # array for loading image
npa[:,:,3] = 255 # set alpha 1.0 (effectively)
new_pic = False

# Create a pool of image processors
done = False
lock = threading.Lock()
pool = []

class ImageProcessor(threading.Thread):
    def __init__(self):
        super(ImageProcessor, self).__init__()
        self.stream = io.BytesIO()
        self.event = threading.Event()
        self.terminated = False
        self.start()

    def run(self):
        # This method runs in a separate thread
        global done, npa, new_pic, SIZE, NBYTES
        while not self.terminated:
            # Wait for an image to be written to the stream
            if self.event.wait(1):
                try:
                    if self.stream.tell() >= NBYTES:
                      self.stream.seek(0)
                      # python2 doesn't have the getbuffer() method
                      #bnp = np.fromstring(self.stream.read(NBYTES), 
                      #              dtype=np.uint8).reshape(SIZE, SIZE, 3)
                      bnp = np.array(self.stream.getbuffer(), 
                                    dtype=np.uint8).reshape(SIZE, SIZE, 3)
                      npa[:,:,0:3] = bnp
                      new_pic = True
                except Exception as e:
                  print(e)
                finally:
                    # Reset the stream and event
                    self.stream.seek(0)
                    self.stream.truncate()
                    self.event.clear()
                    # Return ourselves to the pool
                    with lock:
                        pool.append(self)

def streams():
    while not done:
        with lock:
            if pool:
                processor = pool.pop()
            else:
                processor = None
        if processor:
            yield processor.stream
            processor.event.set()
        else:
            # When the pool is starved, wait a while for it to refill
            time.sleep(0.1)


def start_capture(): # has to be in yet another thread as blocking
  global SIZE, pool
  with picamera.PiCamera() as camera:
    pool = [ImageProcessor() for i in range(3)]
    camera.resolution = (SIZE, SIZE)
    camera.framerate = 30
    #camera.led = False
    time.sleep(2)
    camera.capture_sequence(streams(), format='rgb', use_video_port=True)

t = threading.Thread(target=start_capture)
t.start()


while not new_pic:
    time.sleep(0.1)

########################################################################
DISPLAY = pi3d.Display.create(x=100, y=100, w=960, h=720)
DW, DH = DISPLAY.width, DISPLAY.height
CAMERA = pi3d.Camera(is_3d=False)
shader = pi3d.Shader("uv_flat")

tex = pi3d.Texture(npa)
screen = pi3d.Sprite(w=SIZE * 4, h=SIZE * 4, z=1.0)
screen.set_draw_details(shader, [tex])

# Fetch key presses ----------------------
mykeys = pi3d.Keyboard()

xmin, xmax, ymin, ymax = 1000.0, -1000.0, 1000.0, -1000.0

while DISPLAY.loop_running():

  k = mykeys.read()
  if k >-1:
    if k==27:
      mykeys.close()
      DISPLAY.destroy()
      break
    elif k==ord('r'):
      xmin, xmax, ymin, ymax = 1000.0, -1000.0, 1000.0, -1000.0

  if new_pic:
    drk = np.zeros((SIZE, SIZE)) # 2D grid fill with 0.0
    drk[np.where(npa[:,:,:3].max(axis=2) < threshold)] = 1.0 # change to 1.0 where img is dark
    tot = drk.sum() # total sum for grid
    if tot > 0:
      xav = (drk.sum(axis=0) * POS).sum() / tot # mean of dark pixels
      yav = (drk.sum(axis=1) * POS).sum() / tot
      if xav > xmax: xmax = xav
      if xav < xmin: xmin = xav
      if yav > ymax: ymax = yav
      if yav < ymin: ymin = yav
      screen.positionX(DW / 2 - (xav - xmin) / (xmax - xmin) * DW)
      screen.positionY(DH / 2 - (yav - ymin) / (ymax - ymin) * DH)
      #print("{} {} x={:2.1f},y={:2.1f}".format(DW, DH, xav, yav))
    tex.update_ndarray(npa)
    new_pic = False

    
  screen.draw()


# Shut down the processors in an orderly fashion
while pool:
  done = True
  with lock:
    processor = pool.pop()
  processor.terminated = True
  processor.join()
