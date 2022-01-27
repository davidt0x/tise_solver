import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from tise_solver.numerov import numerov
from tise_solver.marsiglio import marsiglio
from tise_solver.one_well_analytic import one_well_energies


# Setting up the parameters
rng = np.random.RandomState(seed=42)

n_wells_values = list(range(1, 11))
# 50 and 100 only if feasible.

params = {}
for n_wells in n_wells_values:

    # Periodic landscape
    depths = 5.0*np.ones((n_wells,))
    widths = 3.0*depths
    seps = 2.0*np.ones((n_wells-1,))
    params[f'periodic n_wells={n_wells}'] = dict(depths=depths, widths=widths, separations=seps)

    # Aperiodic landscape
    depths_ap = depths + 10.0*rng.uniform(size=(n_wells,))
    widths_ap = 3.0*depths + 10.0*rng.uniform(size=(n_wells,))
    seps_ap = seps + 10.0*rng.uniform(size=(n_wells-1,))
    params[f'aperiodic n_wells={n_wells}'] = dict(depths=depths_ap, widths=widths_ap, separations=seps_ap)



#%%
import time

p = params[f'periodic n_wells=1']

marsiglio_E = []
marsiglio_times = []
numerov_E = []
numerov_times = []
n_steps_vals = []

dxs = [0.5, 0.2, 0.1, 0.02, 0.01, 0.009, 0.008, 0.007, 0.006, 0.005]

for dx in dxs:
    t0 = time.time()
    nr = numerov(**p, dx=dx, method='dense')
    numerov_times.append(time.time() - t0)
    numerov_E.append(nr['E'])
    n_steps = nr['n_steps']
    n_steps_vals.append(n_steps)
    t0 = time.time()
    mr = marsiglio(**p, nt=n_steps)
    marsiglio_times.append(time.time() - t0)
    marsiglio_E.append(mr['E'])
    print(f"dx = {dx}, numerov_time = {numerov_times[-1]}, marsiglio_time = {marsiglio_times[-1]}")

#%%

E1_marsiglio = np.array([E[0] for E in marsiglio_E])
E1_numerov = np.array([E[0] for E in numerov_E])

# Do a convergence plot for the one-well case
s = one_well_energies(depth=p['depths'][0], width=p['widths'][0])
df_marsiglio = pd.DataFrame({'n_steps': n_steps_vals,
                             'log10(n_steps)': np.log10(n_steps_vals),
                             'dx': dxs,
                             'E1_err': np.abs(E1_marsiglio - s[0])/s[0],
                             'time': marsiglio_times,
                             'method': 'marsiglio'})
df_numerov = pd.DataFrame({'n_steps': n_steps_vals,
                           'log10(n_steps)': np.log10(n_steps_vals),
                           'dx': dxs,
                           'E1_err': np.abs(E1_numerov - s[0])/s[0],
                           'time': numerov_times,
                           'method': 'numerov'})
df = pd.concat([df_marsiglio, df_numerov])

df.to_csv('results/numerov_vs_marsiglio.csv', index=False)

#%%
plt.figure()
ax = sns.lineplot(x='log10(n_steps)', y='E1_err', hue='method', data=df)
ax.set(ylabel='Relative Error with analytic solution one well first energy')
plt.savefig('results/numerov_vs_marsiglio_accuracy.png')

#%%
plt.figure()
ax = sns.lineplot(x='n_steps', y='time', hue='method', data=df)
ax.set(ylabel='Execution Time (s)')
plt.savefig('results/numerov_vs_marsiglio_time.png')
