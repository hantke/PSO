import numpy as np
import time
import PSO

def f(param, a = 0.05, b = 0.05):
    time.sleep(0.01) # To test MPI performance only
    x,y = param
    return (x-a)**2+(b-y)**2

bounds = np.array([[0,2],[0,2]])
params = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
pso = PSO.global_PSO(bounds = bounds, params = params, npoints=6, niter = 100)
#pso = PSO.global_PSO(backup_name = 'backup.h5')
pso.run(f, func_argv = (0.05,0.05))
pso.save('backup.h5')
#pso.plot_animation(best = (0.05, 0.05))
