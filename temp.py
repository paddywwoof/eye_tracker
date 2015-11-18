#!/usr/bin/python
# place holder for numpy code to be used with picam for watching pupil

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
THRESHOLD = 25
POS = np.arange(64, dtype=np.float)

img = np.random.randint(0,255,(64,64,3)).astype(np.uint8) # actually get from camera
drk = np.zeros((64,64)) # 2D grid fill with 0.0
drk[np.where(img.max(axis=2) < THRESHOLD)] = 1.0 # change to 1.0 where img is dark
tot = drk.sum() # total sum for grid
xav = (drk.sum(axis=0) * POS).sum() / tot # mean of dark pixels
yav = (drk.sum(axis=1) * POS).sum() / tot

print(xav, yav)
