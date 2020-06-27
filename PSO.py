import numpy as np

class global_PSO():
    def __init__(self,
                 bounds = None,
                 params = None,
                 npoints = 10,
                 niter = 100,
                 initial_pos = None,
                 initial_vel = None,
                 max_speed   = 0.5):
        
        self.ndim    = len(bounds)
        self.npoints = npoints
        self.niter   = niter
        self.swarm   = self.init_swarm()
        self.bounds  = bounds
        
        if max_speed is None:
            #The max speed should not be more than 1 box length 
            self.max_speed = bounds[:,1]-bounds[:,0]
        else:
            #The max speed is on unites of the boundaries
            self.max_speed = (bounds[:,1]-bounds[:,0])*max_speed
            
        self.params = params if params is not None else {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        
        self.init_particles(initial_pos = None, initial_vel = None)
             
    def init_swarm(self):
        swarm = {}
        swarm['pos'] = np.zeros((self.npoints, self.ndim))
        swarm['vel'] = np.zeros((self.npoints, self.ndim))
        swarm['val'] = np.zeros((self.npoints))
        
        swarm['pos_histo'] = np.zeros((self.niter,self.npoints, self.ndim))
        swarm['val_histo'] = np.zeros((self.niter,self.npoints))
        
        swarm['pos_blocal']  = np.zeros((self.npoints, self.ndim))
        swarm['pos_bglobal'] = np.zeros((self.ndim))
        swarm['val_blocal']  = np.zeros((self.npoints))
        swarm['val_bglobal'] = 0
        return swarm
        
    def update_velocity(self):
        #Based on the function from pyswarms
        #https://github.com/ljvmiranda921/pyswarms/blob/master/pyswarms/backend/operators.py
        def _apply_clamp(self):
            for i in range(self.ndim):
                self.swarm['vel'][:,i][self.swarm['vel'][:,i] >  self.max_speed[i]] =  self.max_speed[i]
                self.swarm['vel'][:,i][self.swarm['vel'][:,i] < -self.max_speed[i]] = -self.max_speed[i]
        c1 = self.params["c1"]
        c2 = self.params["c2"]
        w  = self.params["w"]
        
        for i in range(self.ndim):
            cognitive = c1 * np.random.uniform(0, 1, self.npoints) * (self.swarm['pos_blocal'][:,i] - self.swarm['pos'][:,i])
            social    = c2 * np.random.uniform(0, 1, self.npoints) * (self.swarm['pos_bglobal'][i]  - self.swarm['pos'][:,i])
            self.swarm['vel'][:,i] = (w * self.swarm['vel'][:,i]) + cognitive + social
            self.swarm['vel'][:,i][self.swarm['vel'][:,i] >  self.max_speed[i]] =  self.max_speed[i]
            self.swarm['vel'][:,i][self.swarm['vel'][:,i] < -self.max_speed[i]] = -self.max_speed[i]
            
    def update_position(self):
            self.swarm['pos'] += self.swarm['vel']
            
    def correct_bound(self, method, reflect_param = 0.5):
        evaluate_mask = np.ones(self.npoints, dtype=bool)
        if method is None:
            #No method is apply, in that case we simple do not evaluate the function on these points,
            #It velocity is not affected, so it may go even further.
            for i in range(self.ndim):
                evaluate_mask[self.swarm['pos'][:,i] < self.bounds[i][0]] = False
                evaluate_mask[self.swarm['pos'][:,i] > self.bounds[i][1]] = False
            return evaluate_mask
        
        if method == 'reflect' or method == 'Reflect':
            # You see a wall, you go against it, you 'bounce', reflect_param slower 
            for i in range(self.ndim):
                count = 0
                any = True
                #Maybe a particle is too far away from the bounds, 
                #the correction must be aplied again,ut with some precautions
                while any:
                    _mask1 = self.swarm['pos'][:,i] < self.bounds[i][0]
                    _mask2 = self.swarm['pos'][:,i] > self.bounds[i][1]
                    if np.any(_mask1):
                        self.swarm['pos'][:,i][_mask1] += 2*(self.bounds[i][0]-self.swarm['pos'][:,i][_mask1])
                        self.swarm['vel'][:,i][_mask1] *= -reflect_param
                    if np.any(_mask2):
                        self.swarm['pos'][:,i][_mask2] -= 2*(self.swarm['pos'][:,i][_mask2]-self.bounds[i][1])
                        self.swarm['vel'][:,i][_mask2] *= -reflect_param
                    any = np.any(_mask1) | np.any(_mask2)
                    count +=1
                    if any and (count > 10):
                        print('WARNING! Particle corrected more than 10 times, probably a bug. seting it in a random position with no velocity')
                        _mask1 = self.swarm['pos'][:,i] < self.bounds[i][0]
                        _mask2 = self.swarm['pos'][:,i] > self.bounds[i][1]
                        self.swarm['pos'][:,i][_mask1] = self.bounds[i][0] + np.random.random()*(self.bounds[i][1]-self.bounds[i][0])
                        self.swarm['vel'][:,i][_mask1] = 0
                        self.swarm['pos'][:,i][_mask2] = self.bounds[i][0] + np.random.random()*(self.bounds[i][1]-self.bounds[i][0])
                        self.swarm['vel'][:,i][_mask2] = 0
                        any=False
                        
        if method == 'border' or method == 'Border':
            # Stay in the edge you cross, no velocity
            for i in range(self.ndim):
                _mask1 = self.swarm['pos'][:,i] < self.bounds[i][0]
                _mask2 = self.swarm['pos'][:,i] > self.bounds[i][1]
                self.swarm['pos'][:,i][_mask1] = self.bounds[i][0]
                self.swarm['vel'][:,i][_mask1] = 0
                self.swarm['pos'][:,i][_mask2] = self.bounds[i][1]
                self.swarm['vel'][:,i][_mask2] = 0
                
        #if methos is 'periodic' or method is 'Periodic' #TODO
        
        return evaluate_mask
            
    def init_particles(self, initial_pos = None, initial_vel = None):
        self.swarm['pos'] = np.random.random((self.npoints, self.ndim)) if initial_pos is None else initial_pos
        self.swarm['vel'] = np.random.random((self.npoints, self.ndim))*self.max_speed  if initial_vel is None else initial_vel
        
    def update_best_pos(self,index):
        self.swarm['pos_histo'][index]   = self.swarm['pos']
        self.swarm['val_histo'][index]   = self.swarm['val']
        for i in range(self.npoints):
            _ival_local               = np.argmin(self.swarm['val_histo'][:index+1,i],axis=0)
            self.swarm['pos_blocal'][i]  = self.swarm['pos_histo'][_ival_local][i]
            self.swarm['val_blocal'][i]  = self.swarm['val_histo'][_ival_local][i]
        _ival_global              = np.argmin(self.swarm['val_blocal'])
        self.swarm['pos_bglobal'] = self.swarm['pos_blocal'][_ival_global]
        self.swarm['val_bglobal'] = self.swarm['val_blocal'][_ival_global]
    
    def print_info(self,i):
        print('%d / %d    lower cost: %.03f    best pos: '%(i,self.niter, self.swarm['val_bglobal']),self.swarm['pos_bglobal'])
        
    def run(self, f, func_argv = (), bound_correct_method = 'reflect', reflect_param = 0.5, verbose = True):
        for i,p in enumerate(self.swarm['pos']):
            self.swarm['val'][i] = f(p,*func_argv)
        self.update_best_pos(0)
        for i in range(1,self.niter):
            self.update_velocity()
            self.update_position()
            evaluate_mask = self.correct_bound(bound_correct_method, reflect_param = reflect_param)
            for j in range(self.npoints):
                p = self.swarm['pos'][j]
                if evaluate_mask[j]:
                    self.swarm['val'][j] = f(p,*func_argv)
                else:
                    self.swarm['val'][j] = np.inf
            if verbose: self.print_info(i)
            self.update_best_pos(i)

    def plot_animation(self, filename = 'plot.gif', xdim = 0, ydim = 1, best = None, MPI = False):
        if MPI and self.rank != 0: return 0
        print('Warning! This plotting code is not efficient, fast or elegant... yet..')
        import bacco
        bacco.plotting.set_alternative1() #Sergio's Style, Really Important!
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from celluloid import Camera
            
        fig,ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        ax.set_xlabel(r'$\rm X$', fontsize=40)
        ax.set_ylabel(r'$\rm Y$', fontsize=40)
        ax.set_xlim([self.bounds[xdim][xdim],self.bounds[xdim][ydim]])
        ax.set_ylim([self.bounds[ydim][xdim],self.bounds[ydim][ydim]])
        camera = Camera(fig)
        for step in range(self.niter):
            ax.text(0.05, 0.9, r'$\rm step=%d$'%(step), transform=ax.transAxes, fontsize=30)
            ax.scatter(self.swarm['pos_histo'][step,:,xdim], self.swarm['pos_histo'][step,:,ydim],s=70,c=['r','g','b','c','m','y'])
            if best is not None: ax.scatter([best[0],], [best[1],],s=100,c='k')
            camera.snap()
        animation = camera.animate()
        animation.save(filename, writer = 'imagemagick')

####### From here MPI ########
    def mpi_scatter_swarm(self):
        _pos = self.swarm['pos'] if self.rank == 0 else None
        _pos = self.comm.bcast(_pos, root=0)
        return _pos
        
    def mpi_run(self, f, func_argv = (), bound_correct_method = 'reflect', reflect_param = 0.5, verbose = True):
        
        from mpi4py import MPI
        self.mpi  = MPI
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
        _pos = self.mpi_scatter_swarm()
        _val = np.zeros(np.size(self.swarm['val']))
        for i,p in enumerate(_pos):
            if i%self.size == self.rank:
                _val[i] = f(p,*func_argv)
        self.comm.Reduce(_val,self.swarm['val'], self.mpi.SUM, 0)
        self.update_best_pos(0)
        for i in range(1,self.niter):
            self.update_velocity()
            self.update_position()
            evaluate_mask = self.correct_bound(bound_correct_method, reflect_param = reflect_param)
            _pos = self.mpi_scatter_swarm()
            _val = np.zeros(np.size(self.swarm['val']))
            for j in range(self.npoints):
                p = _pos[j]
                if j%self.size == self.rank:
                    _val[j] = f(p,*func_argv) if evaluate_mask[j] else np.inf
            self.comm.Reduce(_val,self.swarm['val'], self.mpi.SUM, 0)
            if verbose and self.rank == 0: self.print_info(i)
            self.update_best_pos(i)
