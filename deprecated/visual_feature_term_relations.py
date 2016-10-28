import pandas
import operator

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
feature_weight_list = pandas.read_csv("output/visual_160000_320000.csv", header=None, dtype=float)

#print feature_weight_list
# plt.plot(airports)
# plt.show()

fig = plt.figure()

df = pandas.DataFrame(feature_weight_list)
df.sort_values([300],inplace=True)
for i in range(0,10):
    df[i].plot(style=['.'])
fig.savefig("output/threshold_visualize/temp_160000_320000.png",dpi=300)
