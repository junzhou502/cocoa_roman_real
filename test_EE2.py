# IMPOER EE2
import sys
sys.path.insert(0, '/project/chihway/junzhou/cocoa_approx/Cocoa/external_modules/code/euclidemu2/build/lib.linux-x86_64-cpython-310')
if "euclidemu2" in sys.modules:
    del sys.modules["euclidemu2"]
import euclidemu2 as ee2
import numpy as np
import matplotlib.pyplot as plt
import scipy, scipy.interpolate


font = {'size'   : 16, 'family':'STIXGeneral'}
axislabelfontsize='x-large'
plt.rc('font', **font)
plt.rcParams['text.usetex'] = True

np.set_printoptions(precision=3,linewidth=200,suppress=False)

# Input parameter dictionary 
#
# Accepts multiple formats such as CLASS par dictionary
# Accepts both Omega_x or omega_x = Omega_x*h^2 depending on capitalization of O
# Requires at least 5 parameters (As, ns, Omb, Omm, h)
# Other parameters (mnu,w,wa) will be set to LCDM values if not specified
# Additional parameters will be ignored.

cosmo_par={'As':2.1e-09,
           'ns':0.966,
           'Omb':0.04,
           'Omm':0.3,
           'h':0.68,
           'mnu':0.15,
           'w':-1.0,
           'wa':0.0}
N=100
for idx in range(N):
    print(f'test random cosmology {idx}/{N}')
    cosmo_par={'As':np.random.uniform(1.7,2.5)*10**(-9),
            'ns':np.random.uniform(0.92, 1.00),
            'Omb':np.random.uniform(0.04, 0.06),
            'Omm':np.random.uniform(0.24, 0.40),
            'h':np.random.uniform(0.61, 0.73),
            'mnu':np.random.uniform(0.0, 0.15),
            'w':np.random.uniform(-1.3, -0.7),
            'wa':np.random.uniform(-0.7, 0.5)}

    print([f'{key}:{cosmo_par[key]} ' for key in cosmo_par.keys()])
    tmp=int(1000 + 250*1)
    z_interp_2D = np.concatenate((np.linspace(0,3.0,max(50,int(0.75*tmp))), 
                                        np.linspace(3.01,50.1,max(30,int(0.25*tmp)))),axis=0)
    len_z_interp_2D = len(z_interp_2D)
    log10k_interp_2D = np.linspace(-4.99,2.0,int(1250+250*1))
    len_log10k_interp_2D = len(log10k_interp_2D)

    try:
        k_val, bvals = ee2.get_boost2(cosmo_par, z_interp_2D[z_interp_2D < 10.0], ee2.PyEuclidEmulator(), 10**np.linspace(-2.0589,0.973,len_log10k_interp_2D))
        #k_val, bvals = ee2.get_boost2(cosmo_par, z_interp_2D[z_interp_2D < 1.0], ee2.PyEuclidEmulator(), 10**np.linspace(0,0.1,len_log10k_interp_2D))
    except:
        for key in cosmo_par.keys():
            print(f'{key}: {cosmo_par[key]}')