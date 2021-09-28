import os
import sys
import time

import numpy as np
from mpi4py import MPI
import dedalus.public as d3
from dedalus.extras.flow_tools import GlobalArrayReducer

import logging
logger = logging.getLogger(__name__)

args = sys.argv
# Parameters
radius = 1
Lmax = 63 
L_dealias = 3/2
N_phi = 4
Nmax = 63 
N_dealias = 3/2
dealias_tuple = (1, L_dealias, N_dealias)
# I decreased the timestep size for stability
timestep = 0.005
t_end = 1000
ts = d3.SBDF2
dtype = np.float64
nu = 1e-4
eta = 1e-4
kappa = 1e-4
nprocs = MPI.COMM_WORLD.size


if nprocs >= 2: 


    mesh = [2, int(nprocs/2)]
else:
    mesh = None
omega = float(args[1])
r_0 = 0.875
delta_r = 0.02
tau = 3.
# location of top damping layer
r_top = 0.95

# Bases
c = d3.SphericalCoordinates('phi', 'theta', 'r')
d = d3.Distributor(c, dtype=dtype, mesh=mesh)
b = d3.BallBasis(c, shape=(N_phi, Lmax+1, Nmax+1), radius=radius, dealias=dealias_tuple, dtype=dtype)
b_S2 = b.S2_basis()
phi, theta, r = b.local_grids((1, L_dealias, N_dealias))
Dt = 2*np.pi*0.1
t0 = 4*Dt
F_func = lambda t: np.cos(theta)*np.sin(omega*t)*np.exp(-(r-r_0)**2/delta_r**2)*(np.tanh((t-t0)/Dt)+1)/2

# Fields
u = d.VectorField(c, bases=b, name='u')
rho = d.Field(bases=b, name='rho')
#A   = d.VectorField(c, bases=b, name='A')
p   = d.Field(bases=b, name='p') 
#Phi_field = d.Field(bases=b, name='Phi')

#tau_A = d.VectorField(c, bases=b_S2, name='tau_A')
tau_u = d.VectorField(c, bases=b_S2, name='tau_u')
tau_rho = d.Field(bases=b_S2, name='tau_rho')

#B_0   = d.VectorField(c, bases=b, name='B_0')
rho_0 = d.Field(bases=b.radial_basis, name='rho_0')
g     = d.VectorField(c, bases=b.radial_basis, name='g')
D_N   = d.Field(bases=b.radial_basis, name='D_N')
F     = d.Field(bases=b, name='F')

for fd in [u, F, rho_0, g, D_N]:
    fd.set_scales(dealias_tuple)

rho_0['g'] = -r**2 #change this to change N^2, must be function of r^2
g['g'][2] = -9*r
D_N['g'] = (1+np.tanh((r-r_top)/delta_r))/(2.*tau)
F['g'] = F_func(0)
#bpre = 0.0005


#Initial magneic field
#B_0['g'][0] = 0
#B_0['g'][1] = -bpre*np.sin(theta)
#B_0['g'][2] = bpre*np.cos(theta)

# Parameters and operators
ez = d.VectorField(c, bases=b, name='ez')
ez.set_scales(dealias_tuple)
ez['g'][1] = -np.sin(theta)
ez['g'][2] =  np.cos(theta)

LiftTau = lambda A: d3.LiftTau(A, b, -1)
r_out = 1
ell_func = lambda ell: ell+1
#A_potential_bc = d3.radial(d3.grad(A)(r=1)) + d3.SphericalEllProduct(A, c, ell_func)(r=1)/r_out
stress = d3.grad(u) + d3.TransposeComponents(d3.grad(u))

#grid_B0 = d3.Grid(B_0).evaluate()
#grid_J0 = d3.Grid(d3.curl(B_0)).evaluate()
# Problem
problem = d3.IVP([rho, p, u, tau_u, tau_rho], namespace=locals())

problem.add_equation("dt(rho) + dot(u,grad(rho_0)) - kappa*lap(rho) + LiftTau(tau_rho) = F", condition="ntheta != 0")#density eqn
problem.add_equation("rho = 0", condition="ntheta == 0")

problem.add_equation("div(u) = 0", condition="ntheta != 0") #incompressibility 
problem.add_equation("p = 0", condition="ntheta == 0")

problem.add_equation("dt(u) + grad(p) - g*rho - nu*lap(u) + LiftTau(tau_u) = 0", condition="ntheta != 0") #momentum
problem.add_equation("u = 0", condition="ntheta == 0")

problem.add_equation("radial(u(r=1)) = 0", condition="ntheta != 0")
problem.add_equation("angular(radial(stress(r=1))) = 0", condition="ntheta !=0")

problem.add_equation("tau_u = 0", condition="ntheta == 0")
problem.add_equation("rho(r=1) = 0", condition="ntheta != 0")

problem.add_equation("tau_rho = 0", condition="ntheta == 0")


# new BC's
#problem.add_equation("Phi_field(r=1) = 0", condition="ntheta != 0")

print("Problem built")

# Solver
solver = problem.build_solver(ts)

solver.stop_sim_time = t_end

integ = lambda A: d3.Integrate(A, c)

# Analysis
output_dir = './test_outputs/'
if MPI.COMM_WORLD.rank == 0:
    if not os.path.exists('{:s}/'.format(output_dir)):
        os.makedirs('{:s}/'.format(output_dir))

#B_vec = d3.curl(A)
scalars = solver.evaluator.add_file_handler(output_dir+'scalar', max_writes=np.inf, iter=100)
scalars.add_task(integ(0.5*d3.dot(u, u)),  name='KE')
#scalars.add_task(integ(0.5*d3.dot(B_vec, B_vec)),  name='ME')
KE_op = scalars.tasks[0]['operator']
#ME_op = scalars.tasks[1]['operator']

slices = solver.evaluator.add_file_handler(output_dir+'slices', max_writes=100, sim_dt=1.0)
slices.add_task(u(phi=np.pi), name='u_mer(phi=pi)')
slices.add_task(u(phi=0), name='u_mer(phi=0)')
#slices.add_task(B_vec(phi=np.pi), name='B_mer(phi=pi)')
#slices.add_task(B_vec(phi=0), name='B_mer(phi=0)')
slices.add_task(rho(phi=np.pi), name='rho(phi=pi)')
slices.add_task(rho(phi=0), name='rho(phi=0)')

reducer = GlobalArrayReducer(d.comm_cart)

output_cadence = 10000
file_num = 0

#hermitian cadence so it doesn't blow up??
hermitian_cadence = 100


# Main loop
start_time = time.time()
while solver.proceed:

    if solver.iteration % 10 == 0:
        op_output = KE_op.evaluate()['g']
        if d.comm_cart.rank == 0:
            KE0 = op_output.min()
            #ME0 = ME_op.evaluate()['g'].min()
        else:
            KE0 = 0 #ME0 = 0
        #ME0 = reducer.reduce_scalar(ME0, MPI.SUM)
        logger.info("t = %f, KE = %e" %(solver.sim_time, KE0))

    if solver.iteration % hermitian_cadence in [0,1]:
        for f in solver.state:
            f.require_grid_space()
    F['g'] = F_func(solver.sim_time)
    solver.step(timestep)
end_time = time.time()

