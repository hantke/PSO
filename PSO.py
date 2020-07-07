#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Implementation of the 'PSO' class and helper functions

"""

import numpy as np

class global_PSO():
    def __init__(self,
                 bounds = None,
                 params = None,
                 npoints = 10,
                 niter = 100,
                 initial_pos = None,
                 initial_vel = None,
                 max_speed   = 0.5,
                 backup_name = None):
        """
        Global PSO methid

        Args:
            bounds (array, optional): Boundaries where the PSO is run. Defaults to None.
            params (dictionary, optional): main parameter of the PSO:
                                            c1: cognitive parameter. How atracted the particle is to it best personal value.
                                            c2: social parameter. How atracted the particle is to the best global value.
                                            w:  Inertia parameter. 
                                            Defaults to None.
            npoints (int, optional): Number of points of the PSO. Defaults to 10.
            niter (int, optional): Number of steps of the PSO. Defaults to 100.
            initial_pos (array, optional): Initial position of the particles. Defaults to None.
            initial_vel (array, optional): Initial velocity of the particles. Defaults to None.
            max_speed (float, optional): maximum speed per axis: 1 means 1 box lenght in that axis. Defaults to 0.5.
            backup_name (str optional): Name of the file to initialisate the PSO. Defaults to None.
        """
        
        from mpi4py import MPI
        self.mpi  = MPI
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
        if backup_name is not None:
            self.load(backup_name)
            return                                                      
        
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
        
       
    def save(self, filename):
        """
        Save main information of the PSO

        Args:
            filename (str): Name of the saving file
        """
        if self.rank != 0: return
        import deepdish as dd
        d                        = {}
        d['swarm']               = self.swarm
        d['header']              = {}
        d['header']['ndim']      = self.ndim
        d['header']['npoints']   = self.npoints
        d['header']['niter']     = self.niter
        d['header']['bounds']    = self.bounds
        d['header']['max_speed'] = self.max_speed
        d['header']['params']    = self.params
        dd.io.save(filename, d)
        
    def load(self, filename):
        """
        Load information of the PSO

        Args:
            filename (str): Name of the loading file
        """
        import deepdish as dd
        d = dd.io.load(filename)
        self.swarm     = d['swarm']
        self.ndim      = d['header']['ndim']
        self.npoints   = d['header']['npoints']
        self.niter     = d['header']['niter']
        self.bounds    = d['header']['bounds']
        self.max_speed = d['header']['max_speed']
        self.params    = d['header']['params']

    def init_swarm(self):
        """
        Init. the swarm information

        Returns:
            dictionary: Swarm information
        """
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
        """
        Update velocity of the particles
        """
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
        """
        Update position of the particles
        """
        self.swarm['pos'] += self.swarm['vel']
            
    def correct_bound(self, method, reflect_param = 0.5):
        """
        Correct boundaries

        Args:
            method (str): Type of correction:
                          Reflect: reflect the particle and inverse it velocity
                          Border:  locate the particle in the closest border with velocity 0 in that axis
            reflect_param (float, optional): reduce the velocity when reflecting by this factor. Defaults to 0.5.
        """
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
        
        #These method can be implemented if needed
        #if methos is 'periodic' or method is 'Periodic' #TODO
        #if method is None:
            ##No method is apply, in that case we simple do not evaluate the function on these points,
            ##It velocity is not affected, so it may go even further.
            #for i in range(self.ndim):
                #evaluate_mask[self.swarm['pos'][:,i] < self.bounds[i][0]] = False
                #evaluate_mask[self.swarm['pos'][:,i] > self.bounds[i][1]] = False
            #return evaluate_mask
            
    def init_particles(self, initial_pos = None, initial_vel = None):
        """
        Initial location and velocities of the particles

        Args:
            initial_pos (array or None, optional): Initial position of the particles. If None they are located randomly inside the bondaries. Defaults to None.
            initial_vel (array or None, optional): Initial velocities of the particles. If None they are located randomly between 0 and the maximum velocity. Defaults to None.
        """
        self.swarm['pos'] = self.bounds[:,0]+(self.bounds[:,1]-self.bounds[:,0])*np.random.random((self.npoints, self.ndim)) if initial_pos is None else initial_pos
        self.swarm['vel'] = (1-2*np.random.random((self.npoints, self.ndim)))*self.max_speed  if initial_vel is None else initial_vel
        
    def update_best_pos(self,index):
        """
        Update the best local and global position of the particles

        Args:
            index (int): ID of the particle
        """
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
        """
        Print summary information of the status of the PSO

        Args:
            i (int): Step of the PSO cicle
        """
        print('%d / %d    lower cost: %.03f    best pos: '%(i,self.niter, self.swarm['val_bglobal']),self.swarm['pos_bglobal'])

    def plot_animation(self, filename = 'plot.gif', xdim = 0, ydim = 1, best = None):
        """
        Make an animation of the position and history of the particles

        Args:
            filename (str, optional): Name of the gif file. Defaults to 'plot.gif'.
            xdim (int, optional): Dimention of the x-axis. Defaults to 0.
            ydim (int, optional): Dimention of the y-axis. Defaults to 1.
            best (array, optional): Location of the real value, to compare in the plot. Defaults to None.
        """
        if self.rank != 0: return 
        print('Warning! This plotting code is not efficient, fast or elegant... yet..')
        import plotting
        plotting.set_style() #Sergio's Style, Really Important!
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

    def mpi_scatter_swarm(self):
        """
        Share the position of the particles from CPU 0 to the rest of the CPUs

        Returns:
            array: position of the particles for each CPU
        """
        _pos = self.swarm['pos'] if self.rank == 0 else None
        _pos = self.comm.bcast(_pos, root=0)
        return _pos
        
    def run(self, f, func_argv = (), bound_correct_method = 'reflect', reflect_param = 0.5, verbose = True):
        """
        Run the PSO. 

        Args:
            f (function): Function to evaluate the PSO
            func_argv (tuple, optional): aditional argument of function f. Defaults to ().
            bound_correct_method (str, optional): Method to correct boundaries. Defaults to 'reflect'.
            reflect_param (float, optional): Speed decrease factor when particles are reflected when crossing the boundaries. Defaults to 0.5.
            verbose (bool, optional): Old Classic Verbose. Defaults to True.
        """
        
        _pos = self.mpi_scatter_swarm()
        _val = np.zeros(np.size(self.swarm['val']))
        for i,p in enumerate(_pos):
            if i%self.size == self.rank:
                _val[i] = f(p,*func_argv)
        self.comm.Reduce(_val,self.swarm['val'], self.mpi.SUM, 0)
        self.update_best_pos(0)
        if verbose and self.rank == 0: self.print_info(0)
        for i in range(1,self.niter):
            if self.rank == 0:
                self.update_velocity()
                self.update_position()
                self.correct_bound(bound_correct_method, reflect_param = reflect_param)
            _pos = self.mpi_scatter_swarm()
            _val = np.zeros(np.size(self.swarm['val']))
            for j in range(self.npoints):
                p = _pos[j]
                if j%self.size == self.rank: _val[j] = f(p,*func_argv)
            self.comm.Reduce(_val,self.swarm['val'], self.mpi.SUM, 0)
            self.update_best_pos(i)
            if verbose and self.rank == 0: self.print_info(i)
