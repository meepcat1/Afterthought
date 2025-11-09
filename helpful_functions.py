# rocketCEA run. Assumes propellent of Jet-A, and pressure in kPa.

from rocketcea.cea_obj_w_units import CEA_Obj
import numpy as np

def CEA_create(oxidizer, fuel, CR):
    return CEA_Obj(
        oxName=oxidizer,
        fuelName=fuel,
        useFastLookup=0,
        makeOutput=0,
        isp_units="sec",
        cstar_units="m/s",
        pressure_units="kPa",
        temperature_units="K",
        sonic_velocity_units="m/s",
        enthalpy_units="J/kg",
        density_units="kg/m^3",
        specific_heat_units="J/kg-K",
        viscosity_units="millipoise",
        thermal_cond_units="mcal/cm-K-s",
        fac_CR=CR,
        make_debug_prints=False,
    )


# function definition
def area_eqn(mach_e, gamma, area_ratio):
    return (1/mach_e) * (((gamma + 1)/2) ** (-(gamma + 1)/(2*(gamma - 1)))) * \
        ((1 + ((gamma - 1)/2) * (mach_e ** 2)) ** ((gamma + 1)/(2 * (gamma - 1)))) - area_ratio

def area_mach(mach_e, gamma):
    return (1/mach_e) * (((gamma + 1)/2) ** (-(gamma + 1)/(2*(gamma - 1)))) * \
        ((1 + ((gamma - 1)/2) * (mach_e ** 2)) ** ((gamma + 1)/(2 * (gamma - 1))))

# bisection solver
def bisection_solver(f, a, b, tol, n_max, *func_consts):

    n = 1

    while n <= n_max:

        midpoint = (a+b)/2

        if f(midpoint, *func_consts) == 0 or (b-a)/2 < tol:
            return midpoint, n

        n += 1

        if np.sign(f(midpoint, *func_consts)) == np.sign(f(a, *func_consts)):
            a = midpoint

        else:
            b = midpoint


# p/p0
def pres_ratio(mach, gamma):
    return (1 + ((gamma - 1)/2)*(mach**2)) ** (- gamma / (gamma - 1))


# T/T0
def static_temp(mach, gamma):
    return (1 + ((gamma - 1)/2)*(mach**2)) ** -1


# rho/rho0
def static_density(mach, gamma):
    return (1 + ((gamma - 1)/2)*(mach**2)) ** (-1 / (gamma - 1))


# local speed of sound
def sound_speed(gamma, R, Temp):
    return np.sqrt(gamma * R * Temp)


# needs separate things
def thrust_coefficient(area_rat, pep0, pap0, gamma):
    return np.sqrt( ((2*gamma**2)/(gamma - 1)) * ((2/(gamma + 1)) ** ((gamma + 1)/(gamma - 1))) *
                    (1 - pep0**((gamma - 1)/gamma)) ) + (pep0 - pap0)*area_rat



def mach_solver(gamma, area_ratio, tol=0.001, n_max=100, supersonic=True):

    if supersonic:
        left_bound = 1
        right_bound = 20
    else:
        left_bound = 0.000001
        right_bound = 1

    val, iterations = bisection_solver(area_eqn, left_bound, right_bound, tol, n_max, gamma, area_ratio)

    return val, iterations

def volume_equation(A_t, L_c, cont, theta, V_c):
    return A_t * (L_c*cont + (1/3)* np.sqrt(A_t/np.pi)*(1/np.tan(np.deg2rad(theta)))*((cont**(1/3))-1)) - V_c

def vol_solver(A_t, L_c, theta, V_c, tol=0.001, n_max=100):
    
    left_bound = 0.000000001
    right_bound  = 20

    val, iterations = bisection_solver(lambda cont: volume_equation(A_t, L_c, cont, theta, V_c), left_bound, right_bound, tol, n_max)

    return val, iterations


def char_vel(gamma, R, temp):
    return np.sqrt((1/gamma)*(((gamma+1)/2)**((gamma+1)/(gamma-1)))*R*temp)

# p02/p01
def stagnation_across_shock_ratio(mach_1, gamma):
    return (( (((gamma+1)/2) * mach_1**2)/(1+((gamma-1)/2) * mach_1**2) ) ** (gamma/(gamma-1))) * \
        (((((2*gamma)/(gamma+1)) * mach_1**2)-((gamma-1)/(gamma+1))) ** (-1/(gamma-1)))

# returns mach 2 (subsonic) across a shock
def mach_after_shock(mach_1, gamma):
    return np.sqrt(((mach_1**2)+(2/(gamma-1)))/(((2*gamma/(gamma-1))*mach_1**2)-1))

# normal shock in nozzle, finds exit mach
def shock_exit_mach(p0_1, astar_1, p_e, A_e, gamma):
    return np.sqrt( (-1/(gamma-1)) + np.sqrt((1/((gamma-1)**2))+(2/(gamma-1))*((2/(gamma+1))**((gamma+1)/(gamma-1)))*(p0_1*astar_1/(p_e*A_e))**2) )

def static_across_shock(mach, gamma):
    return ((2*gamma*(mach**2)) - (gamma-1))/(gamma+1)

def prandtl(mach, gamma):
    term_1 = np.sqrt((gamma+1)/(gamma-1))
    term_2 = np.arctan( np.sqrt( ((gamma-1)/(gamma+1)) * ((mach**2) - 1) ) )
    term_3 = np.arctan( np.sqrt((mach**2)-1) )
    val = term_1 * term_2 - term_3
    return val

def pressure_finder(pres_0, gamma, area_ratios):

    # find supersonic and subsonic machs
    mach_sub, _ = mach_solver(gamma, area_ratios[-1], supersonic=False, tol=1e-9)
    mach_super, _ = mach_solver(gamma, area_ratios[-1], supersonic=True, tol=1e-9)

    exitpres_sub = pres_ratio(mach_sub, gamma) * pres_0
    exitpres_super = pres_ratio(mach_super, gamma) * pres_0


    backshock_pres = exitpres_super*((((2*gamma)/(gamma+1))*mach_super**2)-((gamma-1)/(gamma+1)))  # NS at exit
    p_over = (exitpres_super + backshock_pres) / 2 # oblique shock
    p_under = (exitpres_super*0.6)  # slightly less than isentropic

    exitpres_half = (backshock_pres + exitpres_sub) / 2  # shock in nozzle

    p_sonic = exitpres_sub * 1.2
    return exitpres_sub, exitpres_super, backshock_pres, exitpres_half, p_under, p_over, p_sonic, mach_sub, mach_super


def isshock(p_amb, exitpres_sub, backshock_pres, exitpres_super):
    # when ambient is same as exitshock_pres, there is a shock at exit
    if p_amb > exitpres_sub:
        shock_loc = "Subsonic"
    elif p_amb == exitpres_sub:
        shock_loc = "Sonic"
    elif p_amb == backshock_pres:
        shock_loc = "Normal Shock at Exit"
    elif exitpres_sub > p_amb > backshock_pres:
        shock_loc = "Normal Shock Inside"  # very overexpanded
    elif backshock_pres > p_amb > exitpres_super:
        shock_loc = "Oblique Shock"  # overexpanded
    elif p_amb == exitpres_super:
        shock_loc = "Ideal"
    elif p_amb < exitpres_super:
        shock_loc = "Expansion Fan"  # underexpanded
    else:
        shock_loc = "Error"

    return shock_loc


def shock_finder(pres_0, A_t, p_amb, area_ratios, gamma, machs, x_pos):

    mach_exit = shock_exit_mach(pres_0, A_t, p_amb, area_ratios[-1]*A_t, gamma)
    Ae_Astar2 = area_mach(mach_exit, gamma)

    Astar2 = (1/Ae_Astar2)*(A_t*area_ratios[-1])
    p02_p01 = np.zeros(area_ratios.size, dtype=np.double)
    Astar1_Astar2 = A_t/Astar2  # this is equal to P02/P01 due to nozzle continuity

    throat_ind = np.argmin(np.abs(x_pos)) # ensures supersonic values, so function doesnt pull complex numbers
    for i in range(throat_ind, area_ratios.size):
        p02_p01[i] = stagnation_across_shock_ratio(machs[i], gamma)

    shock_index = np.argmin(np.abs(p02_p01-Astar1_Astar2))
    shock_loc_x = x_pos[shock_index]

    return shock_loc_x, shock_index

def oblique_shock(p_amb, gamma, exitpres_super, mach_super):
    # calculate turning angle
    beta = np.arcsin(np.sqrt(((p_amb / exitpres_super - 1) * ((gamma + 1) / (2 * gamma)) + 1) / (mach_super ** 2)))
    beta = (180 / np.pi) * beta

    return beta

def expansion(mach_e, pres_0, p_amb, gamma):

    if p_amb <= 0:
        mach_2 = np.inf
    else:
        mach_2 = np.sqrt(((p_amb / pres_0) ** ((gamma - 1) / (-gamma)) - 1) * (2 / (gamma - 1)))

    theta_2 = prandtl(mach_2, gamma)*(180/np.pi)

    theta_1 = prandtl(mach_e, gamma)*(180/np.pi)

    theta = theta_2 - theta_1

    return theta


def exit_pres(pres_atm, Pe_sub, Pe_super, Pb_prime):
    cond = isshock(pres_atm, Pe_sub, Pb_prime, Pe_super)
    if cond == "Normal Shock Inside" or cond == "Subsonic" or cond == "Sonic":
        return pres_atm
    else:
        return Pe_super


def corrected_CF(area_rat, pep0, pap0, gamma, lamb):
    term1 = (2 * gamma ** 2) / (gamma - 1)
    term2 = ((2 / (gamma + 1)) ** ((gamma + 1) / (gamma - 1)))
    term3 = (1 - pep0 ** ((gamma - 1) / gamma))

    return (np.sqrt(term1*term2*term3)*lamb) + ((pep0 - pap0) * area_rat)

def corrected_ISP(cstar, C_F, g_0):
    return (cstar*C_F)/g_0


def corrected_F(A_t, p0, Cf):
    return A_t*p0*Cf



