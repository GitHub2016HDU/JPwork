#!/usr/bin/evn python
# -*- coding: utf-8 -*-
import numpy as np
from scipy import ndimage

maskdata = np.zeros(((5, 5)))

maskdata[1:4, 1] = 1

maskdata[1, 1:4] = 1

maskdata[4, 4] = 1

# maskdata[2:]

label_image, num_future = ndimage.label(maskdata)
print(maskdata, label_image, num_future)

print(ndimage.sum(maskdata))
a = 108
b = 34


def sum(n1, n2):
    sum1 = n1
    sum1 = n1+n2
    return sum1


c = sum(n1=a, n2=b)

print(a, b, c)

try:
    print('all ok')
except Exception as err:
    print(err)





