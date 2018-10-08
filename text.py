import numpy as np
import scipy.misc
import rasterizer
import matplotlib.pyplot as plt

r = rasterizer.Rasterizer(224, 224, 128)

data = [[[38, 25, 13, 4, 0, 4, 12, 25, 32, 30], [29, 23, 25, 31, 43, 58, 63, 53, 34, 28]], [[72, 60, 52, 49, 54, 61, 66, 66, 61], [72, 82, 96, 106, 117, 110, 88, 75, 63]], [[55, 39, 25, 26, 33, 38, 48, 57, 56, 51], [8, 20, 39, 47, 60, 65, 67, 52, 39, 32]], [[102, 91, 79, 76, 82, 88, 103, 103, 99], [34, 42, 65, 86, 89, 84, 60, 48, 43]], [[115, 109, 105, 108, 113, 120, 131, 137, 135, 124], [90, 96, 116, 136, 146, 146, 139, 124, 111, 96]], [[196, 191, 172, 157, 166, 174, 187, 190, 185], [65, 61, 68, 89, 91, 87, 74, 63, 55]], [[255, 239, 226, 220, 214, 220, 228, 219], [2, 2, 10, 19, 39, 37, 22, 0]]]
data2 = [[[167, 151], [0, 12]], [[99, 74, 32, 21, 15, 16, 22, 37, 60, 75], [33, 30, 31, 41, 52, 67, 78, 91, 99, 97]], [[155, 146, 125, 113, 106, 110, 130, 165, 180, 188, 188, 181], [73, 69, 69, 73, 81, 96, 101, 101, 93, 75, 56, 44]], [[225, 202, 192, 183, 182, 189, 202, 219, 240, 252, 255], [59, 58, 60, 76, 92, 107, 116, 123, 123, 101, 74]], [[60, 50, 38, 5, 0, 6, 17, 57, 82, 97, 101], [106, 106, 113, 139, 165, 177, 185, 188, 173, 154, 141]], [[167, 157, 130, 119, 118, 128, 143, 166, 177], [125, 125, 151, 166, 181, 186, 185, 175, 163]], [[232, 230, 213, 199, 196, 206, 236, 246, 255, 254, 244, 225], [154, 147, 147, 164, 182, 189, 189, 182, 165, 147, 143, 143]]]

for x, y in data:
    plt.plot(x, y, marker='.')
    plt.axis('off')

plt.gca().invert_yaxis()

plt.show()  

for x, y in data2:
    plt.plot(x, y, marker='.')
    plt.axis('off')

plt.gca().invert_yaxis()

plt.show()  

batch = [data, data2]

draw_batch = []

for draw in batch:
    drawing = []
    for x, y in draw:
        xy = np.array([x, y], dtype=np.float32)
        print(xy)
        print(xy.shape)
        drawing.append(xy)
    draw_batch.append(drawing)

print(draw_batch)

im = r.Render(draw_batch)
scipy.misc.imsave('sample0.png', im[0,:,:])
scipy.misc.imsave('sample1.png', im[1,:,:])