from packaging import version

import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
import tensorboard as tb


experiment_id = "aHqvxPXzTGWesKPKLX8srQ"
experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
df = experiment.get_scalars()

csv_path = './tb_experiment_2.csv'
dfw = experiment.get_scalars(pivot=False)

dfw.to_csv(csv_path, index=False)

blue = [19 / 255., 147 / 255., 202 / 255.]
red = [255 / 255., 2 / 255., 0 / 255.]
green = [40 / 255., 189 / 255., 79 / 255.]
color = [red, blue, green]

x = [0.05, 0.1, 0.15, 0.2, 0.3]
fedavg = [0.90929, 0.91207, 0.914514, 0.90973, 0.83]
our = [0.9502933673469389, 0.9504948979591836, 0.9499056122448979, 0.9460229591836734, 0.94078]

plt.plot(x, fedavg, color=color[0], linewidth='3')
plt.plot(x, our, color=color[1], linewidth='3')

plt.xlabel('Communication Frequency (Percentage)')
plt.ylabel('Overall Accuracy')


plt.show()
