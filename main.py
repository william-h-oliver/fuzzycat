from FuzzyCat import *
import matplotlib.pyplot as plt

dirName = ['/home/william/Research/Reclusterings/HalfMoons',
           '/home/william/Research/Reclusterings/Circles',
           '/home/william/Research/Reclusterings/Blobs'][2]
fc = FuzzyCat(dirName)
fc.run()

plt.plot(np.arange(fc.ordering.size), fc.jaccardIndices[fc.ordering], lw = 1)
plt.show()