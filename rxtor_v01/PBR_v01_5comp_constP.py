# %%
# Importing packages

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# %%
# Key parameters
# %%
NN = 201       # Number of points in the reactor
L_bed = 0.2     # Catalyst Bed length
rho_cat = 1000  # (kg/m^3)
epsi = 0.4      # void fraction


# %%
# Function to calculate the rate of reaction
# %%
# Comp 1 ~ 5 =
# A, B, C, D, E, 
# Reactions:
# r1: A <-> B
# r2: B <-> C
# r3: C <-> D
# r4: D <-> E
# r_ov = -2*r1+2*r1_rev -2*r3+2*r3_rev
R_gas = 8.3145  # J/mol/K
def Arrh(Ea,T, T_ref):
    within_exp = -Ea/R_gas*(1/T-1/T_ref)
    exp_term = np.exp(within_exp)
    return exp_term
def rate_cal(k_ref_list,
             K_eq_ref_list,
             T_ref_list,
             Ea_list,
             dH_list,
             T, P_list):
    #k1,k2,k3,k4 = k_ref_list
    P1,P2,P3,P4,P5 = P_list
    Ea_rev_list = np.array(Ea_list) - np.array(dH_list)
    ex_f_list = []
    ex_b_list = []
    for T_r, Ea, Ea_rev in zip(T_ref_list, Ea_list, Ea_rev_list):
        ex_f_tmp = Arrh(Ea, T, T_r)
        ex_b_tmp = Arrh(Ea_rev, T, T_r)
        ex_f_list.append(ex_f_tmp)
        ex_b_list.append(ex_b_tmp)
    ex_f_arr = np.array(ex_f_list)
    ex_b_arr = np.array(ex_b_list)
    kf_ar = np.array(k_ref_list)*ex_f_arr
    kb_ar = np.array(k_ref_list)/np.array(K_eq_ref_list)*ex_b_arr

    #for TT in T_ref_list:
    r1 = kf_ar[0]*P1
    r2 = kf_ar[1]*P2
    r3 = kf_ar[2]*P3
    r4 = kf_ar[3]*P4
    r1_rev = kb_ar[0]*P2
    r2_rev = kb_ar[1]*P3
    r3_rev = kb_ar[2]*P4
    r4_rev = kb_ar[3]*P5
    return r1, r2, r3,r4, r1_rev, r2_rev, r3_rev, r4_rev

kf_list_test = [0.0001, 0.03, 0.01, 0.0002]
K_eq_list_test=[0.2, 0.1, 0.3, 0.4]
T_ref_list_test = [300, ]*4
Ea_list_test = [40000, 3000, 5000, 20000] # J/mol
dH_list_test = [30000,-2000,-3000, 10000] # J/mol

# %% 
# Testing the reaction function
# %% 
P_list_test = [0.4,0.2,0.2,0.2, 0 ] # in bar
res_rate_cal = rate_cal(kf_list_test, K_eq_list_test,         
                        T_ref_list_test, 
                        Ea_list_test, dH_list_test,
                        400, P_list_test)
print(res_rate_cal)
         
# %%
# Function for the ODEs

# %%
######## TEMPERATURE DEPENDENT EQUILIBRIUM CONSTANT ########
#T = 773 # K # Temperature (400 oC)
#K_eq = 10**(-2.4198 + 0.0003855*T + 2180.6/T) # Equilibrium constant
#############################################################

######## REACTION RATE CONSTANTS ########

# Comp 1 ~ 5 =
# A, B, C, D, E, 
# Reactions:
# r1: A <-> B
# r2: B <-> C
# r3: C <-> D
# r4: D <-> E
kf_list_test = [0.0001, 0.03, 0.01, 0.0002]
K_eq_list_test=[0.2, 0.1, 0.3, 0.4]
T_ref_list_test = [300, ]*4
Ea_list_test = [40000, 3000, 5000, 20000] # J/mol
dH_list_test = [30000,-2000,-3000, 10000] # J/mol
T_test = 773 # K # Temperature (400 oC)
#########################################

def PBR(y, z, k_list, K_eq_list,
        T_ref_list,Ea_list, dH_list, T):
    C1 = y[0]   # A
    C2 = y[1]   # B
    C3 = y[2]   # C
    C4 = y[3]   # D
    C5 = y[4]   # E
    u = y[5]    # velocity
    C_ov = C1+C2+C3+C4+C5
    y1,y2,y3,y4,y5 = np.array(y[:5])/C_ov
    #k1, k2, k3, k1_rev, k2_rev, k3_rev = k_list
    # Concentration to Pressure (bar)
    P1 = C1*R_gas*T/1E5
    P2 = C2*R_gas*T/1E5
    P3 = C3*R_gas*T/1E5
    P4 = C4*R_gas*T/1E5
    P5 = C5*R_gas*T/1E5
    P_list = [P1,P2,P3,P4,P5]
    # Reaction rate calculation
    #r1,r2 = rate_cal(k1_wgs, k2_wgs, P1,P2,P3,P4)
    res_rate = rate_cal(k_list, K_eq_list,
                        T_ref_list, 
                        Ea_list, dH_list, 
                        T, P_list)
    r1,r2,r3,r4 = res_rate[:4]
    r1_rev,r2_rev,r3_rev,r4_rev = res_rate[-4:]
    # Reaction terms
    #Sig_CO2= -(r2-r2_rev)
    #Sig_H2 =-2*(r1-r1_rev)-(r2-r2_rev)-3*(r3-r3_rev)
    #Sig_CO = -(r1-r1_rev) + (r2-r2_rev) - (r3-r3_rev)
    #Sig_CH3OH = (r1-r1_rev)+(r3-r3_rev)
    #Sig_H2O= (r2-r2_rev)+(r3-r3_rev)
    Sig_A = -(r1 - r1_rev)
    Sig_B = (r1-r1_rev) - (r2 - r2_rev)
    Sig_C = (r2 - r2_rev) - (r3 - r3_rev)
    Sig_D = (r3 - r3_rev) - (r4 - r4_rev)
    Sig_E = (r4 - r4_rev)
    #r_ov = Sig_CO2 + Sig_H2 + Sig_CO + Sig_CH3OH + Sig_H2O
    r_ov = Sig_A + Sig_B + Sig_C + Sig_D + Sig_E

    # ODEs
    term_r_ov = 1/u*(1-epsi)/epsi*rho_cat*r_ov
    dC1dz = -y1*term_r_ov+ 1/u*(1-epsi)/epsi*rho_cat*Sig_A  # A
    dC2dz = -y2*term_r_ov+ 1/u*(1-epsi)/epsi*rho_cat*Sig_B  # B
    dC3dz = -y3*term_r_ov+ 1/u*(1-epsi)/epsi*rho_cat*Sig_C  # C
    dC4dz = -y4*term_r_ov+ 1/u*(1-epsi)/epsi*rho_cat*Sig_D  # D
    dC5dz = -y5*term_r_ov+ 1/u*(1-epsi)/epsi*rho_cat*Sig_E  # E
    dudz = 1/C_ov*(1-epsi)/epsi*rho_cat*r_ov 
    
    return np.array([dC1dz, dC2dz, dC3dz, dC4dz, dC5dz, dudz])

# %%
# Initial conditions (Feed condition)
# %%
u_feed = 0.15        # (m/s) Advective velocity
y_feed = np.array([0.9, 0.0, 0.0, 0, 0.1]) # A~E
T_feed = 773 # K (500)
#T_feed = 733 # K 

P_feed = 30                         # (bar)
C_feed = y_feed*P_feed*1E5/R_gas/T_feed  # (mol/m^3)
C_u_feed = np.concatenate([C_feed, [u_feed]])
print(C_u_feed)

# %%
# Solve the ODEs
# %%
kf_list1 = [0.000001, 0.003, 0.005, 0.0001]
K_eq_list1=[4.0, 3.5, 3.0, 4.0]
T_ref_list1 = [300, ]*4
Ea_list1 = [40000, 3000, 5000, 20000] # J/mol
dH_list1 = [30000,-2000,-3000, 10000] # J/mol
T_test1 = T_feed # K # Temperature (400 oC)

arg_ode_list = (kf_list1, K_eq_list1,
                T_ref_list1, 
                Ea_list1, dH_list1,
                T_test1)

z = np.linspace(0,L_bed,NN)
C_res = odeint(PBR, C_u_feed, z, args = arg_ode_list)


# %%
# %%
# Plot the results
# %%
plt.figure(figsize = [5,3.7], dpi = 200)
plt.plot(z, C_res[:,0], label='A', ls = ':')
plt.plot(z, C_res[:,1], label='B', ls = '-.')
plt.plot(z, C_res[:,2], label='C', ls = '--')
plt.plot(z, C_res[:,3], label='D', ls = '-')
plt.plot(z, C_res[:,4], label='E', ls = '-')

plt.xlabel('Position (m)', fontsize = 12)
plt.ylabel('Concentration (mol/m$^{3}$)', fontsize = 12)
plt.legend(fontsize=13)
plt.show()

# %%
# Mole fraction graph
# %%
plt.figure()
C_ov = np.sum(C_res[:,:5], axis=1)
plt.plot(z, C_res[:,0]/C_ov, label='CO$_{2}$', ls = ':')
plt.plot(z, C_res[:,1]/C_ov, label='H$_{2}$', ls = '-.')
plt.plot(z, C_res[:,2]/C_ov, label='CO', ls = '--')
plt.plot(z, C_res[:,3]/C_ov, label='CH$_{3}$OH', ls = '-')
plt.plot(z, C_res[:,4]/C_ov, label='H$_{2}$O', ls = '-')

plt.xlabel('Position (m)', fontsize = 12)
plt.ylabel('Mole fraction (mol/mol)', fontsize = 12)
plt.legend(fontsize=13)
print('mole fraction of H2 at the exit:', C_res[-1,0]/C_ov[-1])
print()
print('Conversion of CO:', (C_feed[1] - C_res[-1,1])/C_feed[1]*100)
#print(C_res[-1,0]/C_ov[-1])

# %%
plt.figure()
plt.plot(z, C_res[:,-1], 'k-',
         label = 'velocity (m/s)')
plt.xlabel('Position (m)', fontsize = 12)
plt.ylabel('Advective velocity (m/s)', fontsize = 12)
plt.legend(fontsize=13)

# %%
# Overall Conversion calculation

# %%
# Based on the feed CO concentration
X_CO = (C_feed[2] - C_res[-1,2])/C_feed[2]
print('The overall conversion of CO is:', X_CO*100, '%')

# %%
# Dummy data for testing parameter estimation

# %%
T_feed = T
u_feed = u_feed
NN = NN
L_bed =L_bed
k1 = 0.006
k2 = 0.002
k3 = 0.004
k1_rev = 0.003
k2_rev = 0.004
k3_rev = 0.001
k_rxn_list = [k1,k2,k3,
              k1_rev,k2_rev,k3_rev]
def P_2_X(P_list,k_list):
    P1,P2,P3,P4,P5 = P_list
    # CO2, H2, CO, CH3OH, H2O
    #C1,C2,C3,C4,C5 = np.array(P_list)/R_gas/T_feed*1E5
    C_feed = np.array(P_list)/R_gas/T_feed*1E5
    C_u_feed = np.concatenate([C_feed, [u_feed,]])
    z_dom = np.linspace(0,L_bed, NN)
    C_res = odeint(PBR, C_u_feed, z_dom, args=(k_list,))
    u_end = C_res[-1,-1]
    X_CO2 = (u_feed*C_feed[0] - u_end*C_res[-1,0] )/u_feed/C_feed[0]
    X_H2 = (u_feed*C_feed[1] - u_end*C_res[-1,1] )/u_feed/C_feed[1]
    X_CO = (u_feed*C_feed[2] - u_end*C_res[-1,2] )/u_feed/C_feed[2]
    
    return X_CO2,X_H2, X_CO
    
P_f_list_test = [2,4,2,0,0]
X_test = P_2_X(P_f_list_test, k_rxn_list) 
print(X_test)

# %%
P_ov = 10
P_CO2_arr = np.arange(0.5,5+0.5, 0.5)
P_H2_arr = np.arange(0.5,5+0.5, 0.5)

P_CO2_input = []
P_H2_input = []
P_CO_input = []
X_H2_output = []
X_CO_output = []

for pco2 in P_CO2_arr:
    for ph2 in P_H2_arr:
        pco = P_ov-pco2-ph2
        if pco <= 0:
            break
        else:
            P_feed_list = [pco2,ph2,pco, 0, 0]
            xco2,xh2,xco = P_2_X(P_feed_list, k_rxn_list)
            P_CO2_input.append(pco2)
            P_H2_input.append(ph2)
            P_CO_input.append(pco)
            X_H2_output.append(xh2)
            X_CO_output.append(xco)
di = {'P_CO2': P_CO2_input,
      'P_H2': P_H2_input,
      'P_CO': P_CO_input,
      'X_H2': X_H2_output, 
      'X_CO': X_CO_output,}
import pandas as pd
df = pd.DataFrame(di)
print(df)
# %%
# If k1 is missing...
# %%
k2 = 0.002
k3 = 0.004
k1_rev = 0.003
k2_rev = 0.004
k3_rev = 0.001
def obj(k1_guess,):
    k_list_tmp = [k1_guess[0], k2,k3,k1_rev,k2_rev,k3_rev]
    X_H2_list = []
    X_CO_list = []
    for pco2, ph2, pco in zip(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2]):
        P_list_tmp = [pco2,ph2,pco,0,0]
        
        _, xh2,xco = P_2_X(P_list_tmp, k_list_tmp)
        X_H2_list.append(xh2)
        X_CO_list.append(xco)
    X_H2_arr = np.array(X_H2_list)
    X_CO_arr = np.array(X_CO_list)
    diff_sq = (X_H2_arr - X_H2_output)**2 + (X_CO_arr - X_CO_output)**2
    diff_sq_sum = np.sum(diff_sq)
    return diff_sq_sum

# %%
k_guess_list = np.linspace(0.001, 0.015, 30)

k1_sol = [0.006,]
diff_sq_sol = obj(k1_sol)

diff_sq_list = []
for kk in k_guess_list:
    diff_sq_tmp = obj([kk,])
    diff_sq_list.append(diff_sq_tmp)

# %%
plt.plot(k_guess_list, diff_sq_list,
         'k-', linewidth = 1.8 )
plt.plot([k1_sol],[diff_sq_sol], 'o',
         ms = 9, mfc = 'r', mec = 'k', mew = 1.5)
plt.xlabel('k guess')
plt.ylabel('Mean squared error (MSE)')

# %%

# %%
from scipy.optimize import minimize

# %%
# Parameter estimation with optim. solver
# %%
k1_guess0 = [0.009,]
#opt_res = minimize(obj, k1_guess0, method = 'Nelder-mead')
#opt_res = minimize(obj, k1_guess0, method = 'BFGS')
#opt_res = minimize(obj, k1_guess0,)
opt_res = minimize(obj, k1_guess0, method = 'Nelder-mead')

# %%
# Solution
# %%
print(opt_res)
print()
print('[SOLUTION of k fitting]')
print('k1 = ', opt_res.x[0])
# %%
MSE_opt = obj(opt_res.x)
plt.plot(k_guess_list, diff_sq_list,
         'k-', linewidth = 1.8 )
plt.title('k1 from Parameter estimation', fontsize = 13.5)
plt.plot([opt_res.x[0]],[MSE_opt], 'o',
         ms = 9, mfc = 'r', mec = 'k', mew = 1.5)
plt.xlabel('k guess')
plt.ylabel('Mean squared error (MSE)')
