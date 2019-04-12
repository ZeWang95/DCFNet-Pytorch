import numpy as np 
from matplotlib import pyplot as plt 
import torch
import sys, os
import pdb

log_dir = sys.argv[1]

net = torch.load(os.path.join(log_dir, 'ckpt.t7'))['net']

ll = []

for n in net.keys():
	if n.endswith('mask'):
		ll.append(torch.relu(net[n]).cpu().data.numpy())

rr = list(range(len(ll[0])))

plot, ax = plt.subplots(len(ll))

accl = []
print('Num of activated bases for each conv layers:')
for i in range(len(ll)):
	ax[i].bar(rr, ll[i])
	acc = np.sum((ll[i] > 0.001)*1.0)
	print(acc)
	accl.append(acc)
	# pdb.set_trace()
print(np.mean(accl))

plt.show()