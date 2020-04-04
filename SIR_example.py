import numpy as np
import scipy as sp
from scipy.integrate import odeint
import matplotlib.pyplot as plt

I = 1e-6
S = 1
R = 0
D = 0
Icum = 0
gamma = 0.0488   # recovery rate
mui = gamma/8.5
beta = 2.91*(gamma+mui) # infection rate

# differential equatinons
def diff(sir, t):

    dsdt = -(beta * sir[0] * sir[1])
    didt = (beta * sir[0] * sir[1]) - gamma * sir[1] - mui*sir[1]
    drdt = gamma * sir[1]
    dddt = mui*sir[1]
    dIdt = beta * sir[0] * sir[1]
    
    dsirdt = [dsdt, didt, drdt, dddt, dIdt]
    
    return dsirdt


# initial conditions
sir0 = (S, I, R, D, Icum)

# time points
t = np.linspace(0, 400, 500)

# solve ODE
# the parameters are, the equations, initial conditions, 
# and time steps (between 0 and 100)
sir = odeint(diff, sir0, t)

fig, ax = plt.subplots()
plt.plot(t, sir[:, 1], label=r'$I(t)$', color = 'darkorange', alpha = 0.6)
plt.plot(t, sir[:, 2], label=r'$R(t)$', color = 'forestgreen', alpha = 0.6)
plt.plot(t, sir[:, 3], label=r'$D(t)$', color = 'darkred', alpha = 0.6)
plt.xlabel(r'time [arb.~unit]')
plt.ylabel(r'fraction')
plt.yscale('log')
#plt.ylim(0,0.8)
plt.legend(frameon = False)
plt.tight_layout()
plt.margins(0,0)
plt.savefig('SIR.png', dpi=300, bbox_inches = 'tight',
    pad_inches = 0)
plt.show()

fig, ax = plt.subplots()
plt.plot(t, sir[:, 3]/(sir[:, 2]+sir[:, 3]), label=r'$M_{\mathrm{p}}^0(t)$')
plt.plot(t, sir[:, 3]/sir[:, 4], label=r'$\mathrm{CFR}_{\mathrm{d}}(t,\tau=0)$')
plt.plot(t[17:], sir[:, 3][17:]/[sir[:, 4][i] for i in range(len(sir[:, 4])-17)], label=r'$\mathrm{CFR}_{\mathrm{d}}(t,\tau=17)$')
plt.legend(frameon = True, loc = 1, fontsize = 10)
plt.ylim([10**-2,1])
ax.yaxis.grid(which="both", alpha = 0.4, ls = '-')
plt.xlabel(r'time [arb. unit]')
plt.yscale('log')
plt.ylabel(r'mortality rate')
plt.tight_layout()
plt.margins(0,0)
plt.savefig('mortality.png', dpi=300, bbox_inches = 'tight',
    pad_inches = 0)
plt.show()
