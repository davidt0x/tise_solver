import timeit
import pandas as pd
import seaborn as sns

from tise_solver.numerov import numerov

if __name__ == "__main__":
    results = []
    for method in ['sparse', 'dense']:
        for dx in [0.02, 0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003]:

            # Run it to get the total number of steps for this dx
            n_steps = numerov(widths=[3.0], depths=[10.0], separations=[], dx=dx, method=method)['n_steps']

            # Run to time it
            e = timeit.timeit(stmt=f"numerov(widths=[3.0], depths=[10.0], separations=[], dx={dx}, method='{method}')",
                          setup="from tise_solver.numerov import numerov", number=1)
            print(f'method={method}, dx={dx}, n_steps={n_steps}, time={e}')
            results.append((method, dx, n_steps, e))

    results = pd.DataFrame(results, columns=['method', 'dx', 'n_steps', 'execution_time'])

    sns.lineplot(data=results, x='n_steps', y='execution_time', hue='method')