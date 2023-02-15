import numpy as np
from ship_params import *


#Text File
beta_TC35 = np.loadtxt("C:/Users/benjg/Documents/3A_Cours/Manoeuvrabilité/BE_MAN_3A/mmg_3dof/data_exp/beta_TC35.dat")
beta_ZZ10 = np.loadtxt("C:/Users/benjg/Documents/3A_Cours/Manoeuvrabilité/BE_MAN_3A/mmg_3dof/data_exp/beta_ZZ10.dat")
delta_ZZ10 = np.loadtxt("C:/Users/benjg/Documents/3A_Cours/Manoeuvrabilité/BE_MAN_3A/mmg_3dof/data_exp/delta_ZZ10.dat")
psi_ZZ10= np.loadtxt("C:/Users/benjg/Documents/3A_Cours/Manoeuvrabilité/BE_MAN_3A/mmg_3dof/data_exp/psi_ZZ10.dat")
r_TC35 = np.loadtxt("C:/Users/benjg/Documents/3A_Cours/Manoeuvrabilité/BE_MAN_3A/mmg_3dof/data_exp/r_TC35.dat")
r_ZZ10 = np.loadtxt("C:/Users/benjg/Documents/3A_Cours/Manoeuvrabilité/BE_MAN_3A/mmg_3dof/data_exp/r_ZZ10.dat")
traj_TC35 = np.loadtxt("C:/Users/benjg/Documents/3A_Cours/Manoeuvrabilité/BE_MAN_3A/mmg_3dof/data_exp/traj_TC35.dat")
traj_ZZ10 = np.loadtxt("C:/Users/benjg/Documents/3A_Cours/Manoeuvrabilité/BE_MAN_3A/mmg_3dof/data_exp/traj_ZZ10.dat")
U_TC35 = np.loadtxt("C:/Users/benjg/Documents/3A_Cours/Manoeuvrabilité/BE_MAN_3A/mmg_3dof/data_exp/U_TC35.dat")
U_ZZ10 = np.loadtxt("C:/Users/benjg/Documents/3A_Cours/Manoeuvrabilité/BE_MAN_3A/mmg_3dof/data_exp/U_ZZ10.dat")


#Data
traj_TC35_c1 = np.array(traj_TC35[:,0])*L_pp
traj_TC35_c2 = np.array(traj_TC35[:,1])*L_pp
traj_ZZ10_c1 = np.array(traj_ZZ10[:,0])*L_pp
traj_ZZ10_c2 = np.array(traj_ZZ10[:,1])*L_pp

U_TC35_c1 = np.array(U_TC35[:,0])*L_pp/u_0
U_TC35_c2 = np.array(U_TC35[:,1])*u_0
U_ZZ10_c1 = np.array(U_ZZ10[:,0])*L_pp/u_0
U_ZZ10_c2 = np.array(U_ZZ10[:,1])*u_0

r_TC35_c1 = np.array(r_TC35[:,0])*L_pp/u_0
r_TC35_c2 = np.array(r_TC35[:,1])*u_0/L_pp
r_ZZ10_c1 = np.array(r_ZZ10[:,0])*L_pp/u_0
r_ZZ10_c2 = np.array(r_ZZ10[:,1])*u_0/L_pp

beta_TC35_c1 = np.array(beta_TC35[:,0])*L_pp/u_0
beta_TC35_c2 = np.array(beta_TC35[:,1])
beta_ZZ10_c1 = np.array(beta_ZZ10[:,0])*L_pp/u_0
beta_ZZ10_c2 = np.array(beta_ZZ10[:,1])

delta_ZZ10_c1 = np.array(delta_ZZ10[:,0])*L_pp/u_0
delta_ZZ10_c2 = np.array(delta_ZZ10[:,1])

psi_ZZ10_c1 = np.array(psi_ZZ10[:,0])*L_pp/u_0
psi_ZZ10_c2 = np.array(psi_ZZ10[:,1])

ship_params = kvlcc2
mu_0 = ship_params['mu_0']
L_pp = ship_params['geo']['L_pp']
u_0 = mu_0[0]


