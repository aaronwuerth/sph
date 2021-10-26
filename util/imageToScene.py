#!/usr/bin/env python3

# To the extent possible under law, Aaron Würth has waived all
# copyright and related or neighboring rights to this work.
#
# You should have received a copy of the CC0 legalcode along with this work.
# If not, see <https://creativecommons.org/publicdomain/zero/1.0>.

import matplotlib.pyplot as plt
import numpy as np
import sys

img = plt.imread(sys.argv[1])
img = img / img.max()
center = np.array(img.shape[:2]) / 2.0
particleSize = 0.1
particleTransform = center

if img.shape[:2][0] % 2 == 0:
	particleTransform[0] += 0.5
else:
	particleTransform[0] -= 0.5

if img.shape[:2][1] % 2 == 0:
	particleTransform[1] += 0.5
else:
	particleTransform[1] -= 0.5

for x in range(img.shape[0]):
	for y in range(img.shape[1]):
		if img[y, x][3] > 0.0:
			print(int(img[y, x, 3] <= 0.6), particleSize * (x - particleTransform[0]), -particleSize * (y - particleTransform[1]), img[y, x, 0], img[y, x, 1], img[y, x, 2])
