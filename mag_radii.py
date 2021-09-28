import h5py
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.signal import argrelextrema
#read in file(s)
begin = 1
end = 2

r = None
u = []
for i in range(begin,end):
    print("reading file {}".format(i))
    f = h5py.File('test_outputs/slices/slices_s{}.h5'.format(i),'r')
    utemp = np.array(f['tasks/u_mer(phi=0)'])[:,2,0,47,:]
    u.append(utemp)
    if r is None:
        print(f['tasks/u_mer(phi=0)'].dims[3][0][:].ravel())
        r = f['tasks/u_mer(phi=0)'].dims[3][0][:].ravel()


u = np.concatenate(u)

#find rms average of B and u
usqrd = np.abs(u)**2
uavg = np.sqrt(np.average(usqrd, axis = 0))
#find indices of local maxima
uidx = argrelextrema(uavg, np.greater)

#find radii of local maxima
urad = r[uidx]

#plot
plt.clf()
plt.plot(r,uavg)
plt.xlabel('r')
plt.ylabel('u')
plt.plot([urad,urad],[np.min(uavg),np.max(uavg)],label="r={}".format(urad))
plt.legend()
plt.savefig("uavg_r.pdf")
