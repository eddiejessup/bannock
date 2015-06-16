from __future__ import print_function, division
import sys
import matplotlib.pyplot as plt
import utils

dirname = sys.argv[1]

fig = plt.figure()
ax = fig.add_subplot(111)

ts, p_means, p_mins, p_maxs = utils.get_pmeans(dirname)
plt.plot(ts, p_means)
plt.plot(ts, p_mins)
plt.plot(ts, p_maxs)
plt.show()
