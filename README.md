Python codes for numerically exact dynamics for a 2 level problem coupled to one coordinate x.

For time independent Hamiltonian, 
            
            psi(x,t)=\sum_i c(i) e^(-i.t.eig_en(i)/hbar) eig_fn(x,i) 

where eig_fn(x,i) and eig_en(i) are the ith eigenfunctions of the total Hamiltonian: H.eig_fn(x,i) = eig_en(i).eig_fn(x,i), and c(i) are the quantum coefficients at t=0: c(i)=<eig_fn(i)|psi(t=0)>

eig_fn and eig_en computed by constructing the Hamiltonian in a DVR basis and thereafter diagonalizing the Hamiltonian matrix using inbuilt Python libraries. This code is specifically written for only a 2 level problem coupled to 1 coordinate x.

Running the codes:
  - Download both evolve_QD.py and user_input.py in the same location.
  - Requires Python3, with numpy, and matplotlib libraries. All are freely available.
  - Running on terminal: cd to the directory where the above 2 files are stored. Issue on the terminal: python3 evolve_QD.py
  - Can use Python editors (such as Pycharm) to run the code if not comfortable with terminal.
  - The codes will reproduce the results for the potential in Fig. 1(a) (non-parallel surfaces) of the review.
  - On running, the code will plot the adiabatic potential energy surfaces as a function x. Close the plot, and a movie showing the dynamics of the wavepackets on the 2 surfaces will show up. On closing this movie, another plot with the electronic adiabatic density matrix will show up, after which a plot of the nuclear wavepackets at different time steps will show.
  - The code takes about 1-2 minutes on a regular desktop.
  
 Structure of the code:
  - evolve_QD.py - contains the main code for the dynamics. Detailed algorithm in the review. Comments are added in the code with some details.
  - user_input.py - contains all required input from the user. This includes DVR grid (total number of points and limits), mass and potential parameters, time step and total simulation time.
    - pot(x) in user_input.py takes the classical x (position) as input and returns the potential V[nquant,nquant].
    - init_cond() - returns the initial normalized wavefunction at t=0.
  - The code plots the electronic density matrix after every nprint steps.
  - The code plots the nuclear wavefunctions after every nsave steps.
 
Things to try (excercies):
  - Change DVR grid and limits and re-run to check if the plot of density matrix changes.
  - Current code is provided for the potential V[0,0]=A tanh(Bx), V[1,1]=-V[0,0], V[0,1]=V[1,0]=C.exp(-D.x^2) (x is the classical position).
  - This can be found in the pot(x) function in user_input.py
  - Change the potential to V[1,1]=V[0,0]/2. Re-run the code to reproduce Fig. 4.
  - Change the initial energy. Currently the code starts with kinetic energy of 0.03 a.u. Find the variable k in init_cond() in user_input.py and change initial KE to 0.02 a.u and see the differences
  
  - Change potential to the 3 Tully models given in Tully, JCP 93, 1061, 1990 to reproduce the original results.

  - Run dynamics on just the ground state by setting V[0,1]=V[1,0]=0. Investigate scattering, tunneling, simple harmonic motion, free particle motion, etc.
