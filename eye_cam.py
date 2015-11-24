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
threshold = 40 # HSV value below this will be tracked
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
    camera.framerate = 60
    #camera.led = False
    time.sleep(2)
    camera.shutter_speed = camera.exposure_speed
    camera.exposure_mode = 'off'
    g = camera.awb_gains
    print('g is {}'.format(g))
    camera.awb_mode = 'off'
    camera.awb_gains = g
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
matsh = pi3d.Shader("mat_flat")

tex = pi3d.Texture(npa)
screen = pi3d.Sprite(w=SIZE * 4, h=SIZE * 4, z=1.0)
screen.set_draw_details(shader, [tex])
target = pi3d.Sprite(w=20, h=20, z=0.9)
target.set_material([1.0, 0.7, 0.0])
target.set_shader(matsh)

# Fetch key presses ----------------------
mykeys = pi3d.Keyboard()

ax, ay, bx, by, cx, cy, dx, dy = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
BA, DA, CD, CB = 0.0, 0.0, 0.0, 0.0
SM = 0.98
nf = 0
tm = time.time()
mode = 0
while DISPLAY.loop_running():
  k = mykeys.read()
  if k >-1:
    if k==27:
      mykeys.close()
      DISPLAY.destroy()
      break
    elif k==ord(' '): # space bar
      mode = (mode + 1) % 6
    elif k==ord('l'):
      threshold *= 0.9
    elif k==ord('o'):
      threshold *= 1.1

  if new_pic:
    drk = np.zeros((SIZE, SIZE)) # 2D grid fill with 0.0
    drk[np.where(npa[:,:,:3].max(axis=2) < threshold)] = 1.0 # change to 1.0 where img is dark
    npa[:,:,0] = drk * 255
    tot = drk.sum() # total sum for grid
    if tot > 0:
      x = (drk.sum(axis=0) * POS).sum() / tot # mean of dark pixels
      y = (drk.sum(axis=1) * POS).sum() / tot
      if mode == 0:
        target.position(-DW / 2, -DH / 2, 0.9)
        ax = ax * SM + x * (1.0 - SM)
        ay = ay * SM + y * (1.0 - SM)
      elif mode == 1:
        target.position(-DW / 2, DH / 2, 0.9)
        bx = bx * SM + x * (1.0 - SM)
        by = by * SM + y * (1.0 - SM)
      elif mode == 2:
        target.position(DW / 2, DH / 2, 0.9)
        cx = cx * SM + x * (1.0 - SM)
        cy = cy * SM + y * (1.0 - SM)
      elif mode == 3:
        target.position(DW / 2, -DH / 2, 0.9)
        dx = dx * SM + x * (1.0 - SM)
        dy = dy * SM + y * (1.0 - SM)
      elif mode == 4:
        BA = (bx - ax) / (by - ay)
        CD = (cx - dx) / (cy - dy)
        DA = (dy - ay) / (dx - ax)
        CB = (cy - by) / (cx - bx)
      else:
        target.position(-DW / 2 + DW * (x - ax - (y - ay) * BA) /
                          (dx + (y - dy) * CD - ax - (y - ay) * BA),
                        -DH / 2 + DH * (y - ay - (x - ax) * DA) /
                          (by + (x - bx) * CB - ay - (x - ax) * DA), 1.0)
      if tot > 60.0:
        threshold *= 0.99
    if tot < 50.0:
      threshold *= 1.01
    tex.update_ndarray(npa)
    new_pic = False
    nf += 1

    
  screen.draw()
  target.draw()

print(nf / (time.time() - tm))
print(tot, threshold)

# Shut down the processors in an orderly fashion
while pool:
  done = True
  with lock:
    processor = pool.pop()
  processor.terminated = True
  processor.join()
