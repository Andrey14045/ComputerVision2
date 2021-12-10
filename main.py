import cv2.cv2 as cv
import numpy as np

img = cv.imread('kitten.jpg')
convolution = [5, 3, 3, 3]

width, height, channels = img.shape

filters = np.random.uniform(size=(convolution[0], convolution[1], convolution[2], convolution[3]))
AfterConv = np.zeros(shape=(convolution[0], height, width))

for m in range(convolution[0]):
    for w in range(width):
        for h in range(height):
            for i in range(convolution[2]):
                for j in range(convolution[3]):
                    for k in range(convolution[1]):
                        x = min(w + i, height - 1)
                        y = min(h + j, channels - 1)
                        AfterConv[m][w][h] += img[k, x, y] * filters[m, k, i, j]

gamma = 1
beta = 0
epsilon = 1e-9

for ind in range(convolution[0]):
    avg_1 = AfterConv[ind].mean(axis=0)
    avg_2 = ((AfterConv[ind] - avg_1) ** 2).mean(axis=0)
    AfterConv[ind] = ((AfterConv[ind] - avg_1) / np.sqrt(avg_2 + epsilon)) * gamma + beta

for ind in range(convolution[0]):
    for w in range(width):
        for h in range(height):
            AfterConv[ind, w, h] = np.maximum(0, AfterConv[ind, w, h])


pooling_size = 2
channels, width, height = AfterConv.shape
pooling_width = width // pooling_size
pooling_height = height // pooling_size

maxpooling = np.empty(shape=(channels, pooling_width, pooling_height))

for c in range(channels):
    for w in range(pooling_width):
        for h in range(pooling_height):
            stnd = -1
            for x in range(w * pooling_size, (w + 1) * pooling_size):
                for y in range(h * pooling_size, (h + 1) * pooling_size):
                    stnd = np.maximum(stnd, AfterConv[c, x, y])
            maxpooling[c, w, h] = stnd

softmax = maxpooling
for ind in range(convolution[0]):
    softmax = np.exp(maxpooling[ind]) / sum(np.exp(maxpooling[ind]))

print(softmax)