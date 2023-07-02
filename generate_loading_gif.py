import matplotlib.pyplot as plt
import numpy as np
import imageio

# Function to generate points
def sierpinski(data, iterations, n):
    if iterations == 0:
        return data

    sierpinski_data = []

    for points in data:
        s = np.zeros((3,n,2))
        s[0,:,:] = points/2
        s[1,:,:] = points/2 + np.array([0, 0.5])
        s[2,:,:] = points/2 + np.array([0.5, 0])
        sierpinski_data.extend(s)

    return sierpinski(sierpinski_data, iterations-1, n)

# Initial triangle
data = [np.array([[0,0], [1,0], [0.5, np.sqrt(3)/2], [0,0]])]

fig = plt.figure()
filenames = []

# Generate GIF
for i in range(4):  # Adjust range for more iterations
    plt.clf()
    data = sierpinski(data, i, 4)
    for points in data:
        plt.plot(points[:,0], points[:,1], 'k')
        plt.gca().set_aspect('equal')
    filename = f'frame{i}.png'
    filenames.append(filename)
    plt.savefig(filename)

images = []
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('sierpinski.gif', images, duration=0.5)

print("GIF saved as 'sierpinski.gif'")

