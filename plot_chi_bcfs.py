from __future__ import print_function, division
import sys
import matplotlib.pyplot as plt
import utils

dirnames = sys.argv[1:]

fig = plt.figure()
ax = fig.add_subplot(111)

chis, bcfs = utils.chi_bcfs(dirnames)
plt.scatter(chis, bcfs)
plt.show()
