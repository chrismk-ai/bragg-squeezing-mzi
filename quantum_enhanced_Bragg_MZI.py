"""
Filename:       quantum_enhanced_Bragg_MZI.py
Author:         Christian Karres
Date:           2026-05-12
Description:    Central document for generating the figures from the main text 
                and the Appendix. This includes calculations of the time 
                evolution of resonant first-order Bragg diffraction (cf. main 
                text Fig. 1 and 10), interference signals of an MZI (cf. main text 
                Fig. 3), quasi-probability distributions of atomic states on the 
                angular momentum sphere (cf. main text Fig. 4), as well as phase
                uncertainties of the MZI operated with a One-Axis-Twisted (OAT)
                states polarized along S_x. (cf. main text Figs. 5-9)
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import multiprocessing as mp
import time
import os
from matplotlib.legend_handler import HandlerTuple
from matplotlib.colors import ListedColormap
import math
from sympy.physics.wigner import wigner_3j
from scipy.special import sph_harm_y
from scipy.special import binom
import plotly.graph_objects as go


### Head ###
current_dir = os.path.dirname(__file__)
im_path = current_dir

paper_style = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    
    "pgf.preamble": (
        r"\usepackage[american]{babel} "
        r"\usepackage[utf8x]{inputenc} "
        r"\usepackage[T1]{fontenc} "
        r"\usepackage{siunitx} "
        r"\usepackage{braket} "
        r"\usepackage{amsmath} "
        r"\usepackage{mathrsfs}"
    ),
    
    "figure.figsize": [3.2, 1.6],  # In inches
    
    "axes.labelsize": 9,
    "font.size": 9,
    "axes.titlesize": 9,
    
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "image.cmap": "viridis",
    "axes.xmargin": 0.035,
    "axes.ymargin": 0.035,
    
    "pgf.texsystem": "lualatex"
}
plt.rcParams.update(paper_style)

green = '#6db600ff'
light_green = '#BAEA00'
blue = '#0c5c94ff'
dark_blue = "#064b7cff"
orange = '#ff9000'
dark_orange = '#F44600'
red = '#cc0800ff'
dark_red='#310000'
gray='#CDCDCD'

#### Angular Momentum Quadratures: Means and Variances #####

def mean_jx(N: int, mu: float): 
    """
    Parameters
    ----------
    N : int
        atom number
    mu : float
        squeezing parameter mu = 2 chi

    Returns
    -------
    float
        mean value of the angular momentum quadrature in S_x direction
    """

    return N / 2 * np.cos(mu / 2)**(N - 1)

def var_jx( N: int, mu):
    """variance of the angular momentum quadrature in S_x direction of the 
    One-Axis-Twisted state 
    $e^{-i \alpha \hat{S}_1}\hat{U}_\mt{oat}\ket{\frac{\pi}{2},0}_\mt{css}$

    Parameters
    ----------
    N : int
        atom number
    mu : float
        squeezing parameter mu = 2 chi

    Returns
    -------
    float
        variance of the OAT state in S_x direction
    """
    A = 1 - np.cos(mu)**(N - 2)

    return N / 4 * (
        N * (1 - np.cos(mu / 2)**(2*N - 2)) - 1/2 * (N - 1) * A
    )

def var_jy(N: int, mu, alpha, alignWithXYPlane = False):
    """variance of the angular momentum quadrature in S_y direction of the 
    One-Axis-Twisted state 
    $e^{-i \alpha \hat{S}_1}\hat{U}_\mt{oat}\ket{\frac{\pi}{2},0}_\mt{css}$

    Parameters
    ----------
    N : int
        atom number
    mu : float
        squeezing parameter mu = 2 chi
    alpha : float
        rotation of OAT state around S_x after twisting
    alignWithXYPlane : bool, optional
        if set to True have alpha=-alpha0, i.e., rotating OAT ellipse onto 
        equator, by default False

    Returns
    -------
    float
        variance of the OAT state in S_y direction
    """
    A = 1 - np.cos(mu)**(N - 2)
    B = 4 * np.sin(mu/2) * np.cos(mu/2)**(N - 2)
    alpha0 = np.arctan2(B, A) / 2

    if alignWithXYPlane:
        alpha = -alpha0
        
    return N / 4 * (
        1 + 1 / 2 * (N / 2 - 1) * (A + np.sqrt(A**2 + B**2) 
                                   * np.cos(2 * (alpha + alpha0)))
    )

def var_jz(N: int, mu: float, alpha: float, alignWithXYPlane = False):
    """variance of the angular momentum quadrature in S_z direction of the 
    One-Axis-Twisted state 
    $e^{-i \alpha \hat{S}_1}\hat{U}_\mt{oat}\ket{\frac{\pi}{2},0}_\mt{css}$

    Parameters
    ----------
    N : int
        atom number
    mu : float
        squeezing parameter mu = 2 chi
    alpha : float
        rotation of OAT state around S_x after twisting
    alignWithXYPlane : bool, optional
        if set to True have alpha=-alpha0, i.e., rotating OAT ellipse onto 
        equator, by default False

    Returns
    -------
    float
        variance of the OAT state in S_z direction
    """

    A = 1 - np.cos(mu)**(N - 2)
    B = 4 * np.sin(mu/2) * np.cos(mu/2)**(N - 2)
    alpha0 = np.arctan2(B, A) / 2

    if alignWithXYPlane:
        alpha = -alpha0
        
    return N / 4 * (
        1 + 1 / 2 * (N / 2 - 1 / 2) * (A - np.sqrt(A**2 + B**2) 
                                       * np.cos(2 * (alpha + alpha0)))
    )

def REcov_jyjz(N: int, mu, alpha, alignWithXYPlane = False):
    """
    Parameters
    ----------
    N : int
        atom number
    mu : float
        squeezing parameter mu = 2 chi
    alpha : float
        rotation of OAT state around S_x after twisting
    alignWithXYPlane : bool, optional
        if set to True have alpha=-alpha0, i.e., rotating OAT ellipse onto 
        equator, by default False

    Returns
    -------
    float
        covariance of angular momentum S_y and S_z
    """
    A = 1 - np.cos(mu)**(N - 2)
    B = 4 * np.sin(mu/2) * np.cos(mu/2)**(N - 2)
    alpha0 = np.arctan2(B, A) / 2
    
    if alignWithXYPlane:
        alpha = -alpha0
    
    return N / 4 * (
        np.sin(2 * alpha)  / 2 * (N / 2 - 1 / 2) * np.sqrt(A**2 + B**2) 
        * np.cos(2 * alpha0) + np.cos(2 * alpha) * (N - 1) 
        * np.cos(mu/2)**(N-2) * np.sin(mu/2)
    )

#### Optimal squeezing parameter ####

def mu_opt(N: int):
    """
    Parameters
    ----------
    N : int
        atom number

    Returns
    -------
    _type_
        optimal squeezing parameter minimizing var(S_z)/mean(S_x)^2
    """
    mu = np.linspace(1.2/N**(2/3), 2.5/N**(2/3), 1000)

    return mu[np.argmin(var_jz(N, mu, 0, True)/mean_jx(N, mu)**2)]

#### initial inclination of OAT state w.r.t. the equator

def alpha_0(N: int, mu):
    """inclination of the OAT ellipse w.r.t. equator of the angular momentum
    sphere of the state 
    $\exp\big(-\ii\chi\hat{S}_3^2\big)\ket{\tfrac{\pi}{2},0}_\mt{css}$

    Parameters
    ----------
    mu : float
        twisting strength mu = 2 chi

    Returns
    -------
    float
        initial inclination
    """
    A = 1 - np.cos(mu)**(N - 2)
    B = 4 * np.sin(mu/2) * np.cos(mu/2)**(N - 2)
    return np.arctan2(B, A) / 2

############ Analytical results ############

############ Definitions ############

def t(v, tau_p):
    f = np.sqrt(1 + v**2)
    arg = f * tau_p / 2
    return np.cos(arg) - 1j * v * np.sin(arg) / f

def r(v, tau_p):
    f = np.sqrt(1 + v**2)
    return -1j * np.sin(f * tau_p / 2) / f

def t_Eps(tau_p, f, v, epsilon):
    arg = (-1 + 2 * v**2) * tau_p * epsilon**2 / (32 * f)
    return np.cos(arg) - 1j * v * np.sin(arg) / f

def r_Eps(tau_p, f, v, epsilon):
    arg = (-1 + 2 * v**2) * tau_p * epsilon**2 / (32 * f)
    return - 1j * np.sin(arg) / f

# t, r and t_Eps, r_Eps are 'transmission and reflection coefficients' w.r.t. the 
# diffraction between the main classes n=0,1. The former corresponds to the 
# known time evolution $\mathcal{U}_0$, the latter to the unitary that removes 
# secular terms $\mathcal{U}_2$

def tBack(v, tau_p, epsilon):
    """
    Parameters
    ----------
    v : float
        dimensionless Doppler detuning in units of Rabi frequency \Omega_0  
    tau_p : float
        pulse area time * \Omega_0
    epsilon : float
        dimensionless Rabi frequency (\Omega_0 in units of recoil frequency 
        \omega_k)

    Returns
    -------
    float
        'transmission coefficient' associated with the suitable picture 
        transformation (sub space of main momentum classes n=0,1) 
    """
    f = np.sqrt(1 + v**2)
    return (-np.conj(r_Eps(tau_p, f, v, epsilon)) * r(v, tau_p) 
            + t(v, tau_p) * t_Eps(tau_p, f, v, epsilon))

def rBack(v, tau_p, epsilon):
    """
    Parameters
    ----------
    v : float
        dimensionless Doppler detuning in units of Rabi frequency \Omega_0  
    tau_p : float
        pulse area time * \Omega_0
    epsilon : float
        dimensionless Rabi frequency (\Omega_0 in units of recoil frequency 
        \omega_k)

    Returns
    -------
    float
        'reflection coefficient' associated with the suitable picture 
        transformation (sub space of main momentum classes n=0,1) 
    """
    f = np.sqrt(1 + v**2)
    return (np.conj(t_Eps(tau_p, f, v, epsilon)) * r(v, tau_p) 
            + r_Eps(tau_p, f, v, epsilon) * t(v, tau_p))


def gammaII(epsilon, v, tau_p):
    """
    Parameters
    ----------
    epsilon : float
        dimensionless Rabi frequency (\Omega_0 in units of recoil frequency 
        \omega_k)
    v : float
        dimensionless Doppler detuning in units of Rabi frequency \Omega_0  
    tau_p : float
        pulse area time * \Omega_0

    Returns
    -------
    float
        part of the diagonal matrix elements of the time evolution of resonant 
        first-order Bragg diffraction in the suitable picture 
        (cf. main text \gamma_\mathrm{d} Appx. A.2)
    """
    f = np.sqrt(1 + v**2)
    term1 = 2 * f**2 * (
        -2 * f +
        np.exp(-0.5j * (4/epsilon + f + 3 * v) * tau_p) *
        (f - v + np.exp(1j * f * tau_p) * (f + v))
    )
    term2 = -6j * v * np.sin(f * tau_p)
    return (term1 + term2) / (64 * f**3)

def gammaIJ(epsilon, v, tau_p):
    """
    Parameters
    ----------
    epsilon : float
        dimensionless Rabi frequency (\Omega_0 in units of recoil frequency 
        \omega_k)
    v : float
        dimensionless Doppler detuning in units of Rabi frequency \Omega_0  
    tau_p : float
        pulse area time * \Omega_0

    Returns
    -------
    float
        part of the off-diagonal matrix elements of the time evolution of resonant 
        first-order Bragg diffraction in the suitable picture 
        (cf. main text \gamma_\mathrm{od} Appx. A.2)
    """
    f = np.sqrt(1 + v**2)
    numerator = (
        (-1 + np.exp(1j * f * tau_p)) *
        (
            2 * np.exp(-0.5j * (f + 3 * v + 4/epsilon) * tau_p) * f**3 * (f - v)
            + 3 * f * v * (-1 + np.exp(-1j * f * tau_p) + 2 * (f - v) * v)
        )
    )
    return numerator / (64 * f**4 * (f - v))

#### Functions that need to be weighted with the momentum distribution ####

def A0_0(vk, epsilon, NO_ADJ_CLASSES=False):
    """epsilon^0-magnitude part of the integrand of the population-difference 
    contribution to the interference signal 
    A0(vk,phi=0) = A0_0 + epsilon^2 A0_eps

    Parameters
    ----------
    vk : float
        dimensionless Doppler detuning (\nu_k in units of recoil frequency 
        \omega_k)) 
    epsilon : float
        dimensionless Rabi frequency (\Omega_0 in units of recoil frequency 
        \omega_k)
    NO_ADJ_CLASSES : bool, optional
        specifies whether the adjacent classes n=-1,2 are taken into account 
        (False) or not (True), by default False

    Returns
    -------
    float
        epsilon^0-magnitude part of the integrand of the population-difference 
    contribution to the interference signal 
    """
    phi = 0
    v = vk / epsilon

    tB = tBack(v,np.pi/2,epsilon)
    rhoB = rBack(v, np.pi/2,epsilon)
    rhoM = rBack(v, np.pi,epsilon)
    Bbalance = 2 * np.abs(rhoB)**2 - 1

    if NO_ADJ_CLASSES:
        tB = t(v,np.pi/2)
        rhoB = r(v, np.pi/2)
        rhoM = r(v, np.pi)
        Bbalance = 2 * np.abs(rhoB)**2 - 1

    A00th = (Bbalance**2 * np.abs(rhoM)**2 
             - 4 * np.cos(phi) * np.real(tB * rhoM**2 * np.conj(tB) * np.conj(rhoB)**2))
    return A00th

def A0_eps(vk, epsilon):
    """epsilon^2-magnitude part of the integrand of the population-difference
      contribution to the interference signal 
      A0(vk,phi=0) = A0_0 + epsilon^2 A0_eps

    Parameters
    ----------
        see `A0_0`...
    """
    phi = 0
    v = vk / epsilon

    tB = tBack(v,np.pi/2,epsilon)
    rhoB = rBack(v, np.pi/2,epsilon)
    tM = tBack(v, np.pi,epsilon)
    rhoM = rBack(v, np.pi,epsilon)

    Bbalance = 2 * np.abs(rhoB)**2 - 1
    g22 = gammaII(epsilon, v, np.pi/2)
    g32 = gammaIJ(epsilon, v, np.pi/2)
    gM22 = gammaII(epsilon, v, np.pi)
    gM32 = gammaIJ(epsilon, v, np.pi)

     # helpers
    conj_tB = np.conj(tB)
    conj_rhoB = np.conj(rhoB)
    conj_rhoM = np.conj(rhoM)
    conj_g32 = np.conj(g32)
    conj_gM22 = np.conj(gM22)
    conj_gM32 = np.conj(gM32)
    cos_phi = np.cos(phi)
    exp_iphi = np.exp(1j * phi)

    # 2nd order epsilon phase insensitive part
    term1amp = 2 * Bbalance * np.real(2 * g22 + gM22)
    term2amp = -8 * np.real(g32 * rhoB * conj_tB)
    term3amp = -2 * np.real(Bbalance * gM32 * rhoM * np.conj(tM))
    A02ndAmp = Bbalance * (np.abs(rhoM)**2 * (term1amp + term2amp) + term3amp)

    # 2nd order epsilon phase sensitive part
    inner1 = rhoB * (2 * np.real(g22) * tB + g32 * rhoB) - tB**2 * conj_g32
    term1ph = -4 * np.real(
        exp_iphi * rhoB * inner1 * conj_tB * conj_rhoM**2
    )     
    inner2 = (
        rhoM * (g32 - 2 * (g22 * tB + g32 * rhoB + tB * conj_gM22) * conj_rhoB)
        + 2 * tM * conj_gM32 * tB * conj_rhoB
    )
    term2ph = 4 * cos_phi * np.real(
        rhoM * conj_tB * conj_rhoB * inner2
    )
    A02ndPhase = term1ph + term2ph

    return A02ndAmp + A02ndPhase

def A10_0(vk, epsilon, NO_ADJ_CLASSES=False):
    """epsilon^0-magnitude part of the integrand of the first-order-coherence 
    contribution to the interference signal 
    A10(vk,phi = 0,phi_0 = 0) = A10_0 + epsilon^2 A10_eps

    Parameters
    ----------
        see `A0_0`...
    """
    phi = 0
    phi0 = 0
    v = vk / epsilon

    tB = tBack(v,np.pi/2,epsilon)
    rhoB = rBack(v, np.pi/2,epsilon)
    rhoM = rBack(v, np.pi,epsilon)
    Bbalance = 2 * np.abs(rhoB)**2 - 1

    if NO_ADJ_CLASSES:
        ##### needed when backtrafo taylored
        tB = t(v, np.pi/2)
        rhoB = r(v, np.pi/2)
        rhoM = r(v, np.pi)
        Bbalance = 2 * np.abs(rhoB)**2 - 1

    # helpers
    conj_tB = np.conj(tB)
    conj_rhoM = np.conj(rhoM)
    exp_iphi = np.exp(1j * phi)
    conj_exp_iphi = np.conj(exp_iphi)

    term01 = 2 * Bbalance * rhoB * rhoM * conj_tB * conj_rhoM
    term02 = 2 * (
        exp_iphi * tB**2 + conj_exp_iphi * rhoB**2
    ) * rhoB * conj_tB * conj_rhoM**2
    A100th = term01 + term02

    return (A100th) # * np.exp(-1j * phi0)

def A10_eps(vk, epsilon):
    """ epsilon^2-magnitude part of integrand of the first-order-coherence 
    contribution to the interference signal 
    A10(vk,phi = 0,phi_0 = 0) = A10_0 + epsilon^2 A10_eps
    
    Parameters
    ----------
        see `A0_0`...
    """
    phi = 0
    phi0 = 0
    v = vk / epsilon

    tB = tBack(v,np.pi/2,epsilon)
    rhoB = rBack(v, np.pi/2,epsilon)
    tM = tBack(v, np.pi,epsilon)
    rhoM = rBack(v, np.pi,epsilon)
    Bbalance = 2 * np.abs(rhoB)**2 - 1

    g22 = gammaII(epsilon, v, np.pi/2)
    g33 = gammaII(epsilon, -v, np.pi/2)
    g32 = gammaIJ(epsilon, v, np.pi/2)
    g23 = gammaIJ(epsilon, -v, np.pi/2)
    gM22 = gammaII(epsilon, v, np.pi)
    gM33 = gammaII(epsilon, -v, np.pi)
    gM32 = gammaIJ(epsilon, v, np.pi)
    gM23 = gammaIJ(epsilon, -v, np.pi)

    # helpers
    conj_tB = np.conj(tB)
    conj_rhoM = np.conj(rhoM)
    conj_rhoB = np.conj(rhoB)
    conj_g22 = np.conj(g22)
    conj_g32 = np.conj(g32)
    conj_gM23 = np.conj(gM23)
    conj_tM = np.conj(tM)
    conj_gM33 = np.conj(gM33)
    exp_iphi = np.exp(1j * phi)
    conj_exp_iphi = np.conj(exp_iphi)
            
    ImRhot = np.imag(rhoB * conj_tB)

    # 2nd order epsilon phase insensitive part
    term11 = -4j * Bbalance * ImRhot * np.real(gM32 * rhoM * np.conj(tM))
    term12_inner1 = Bbalance * (Bbalance * np.imag(g32) + 2 * (
        np.imag(g22 * tB * conj_rhoB) - ImRhot * np.real(g33 + gM22)
    ))
    term12_inner2 = 4 * ImRhot * np.real(g23 * tB * conj_rhoB)
    term12 = 2j * np.abs(rhoM)**2 * (-term12_inner1 + term12_inner2)
    A102ndAmp = term11 + term12

    term21 = 4 * rhoB * conj_tB * conj_rhoM**2 * (
        exp_iphi * tB**2 * np.real(g22 + g33)
        + conj_exp_iphi * rhoB**2 * (g33 + conj_g22)        
    ) 

    term22_sub1 = -2 * rhoB * (gM32 - conj_gM23) * conj_tB * conj_tM * conj_rhoM
    term22_sub2 = -Bbalance * (g23 - conj_g32) * conj_rhoM**2
    term22_sub3 = 2 * rhoB * (gM22 + conj_gM33) * conj_tB * conj_rhoM**2
    term22 = (exp_iphi * tB**2 + conj_exp_iphi * rhoB**2) * (
        term22_sub1 + term22_sub2 + term22_sub3
    )

    term23 = 4 * tB * (g23 - conj_g32) * conj_tB * np.real(
        exp_iphi * rhoM**2 * conj_rhoB**2
    )

    A102ndPhase = term21 + term22 + term23

    return (A102ndAmp + A102ndPhase) # * np.exp(-1j * phi0)

def R0_0(vk, epsilon, NO_ADJ_CLASSES=False):
    """ epsilon^0-magnitude part of the integrand of the fraction of atoms 
    exiting the MZI if only n=0-input is populated 
    R0 = R0_0 + epsilon^2 R0_eps (= 1 - Loss)
    
    Parameters
    ----------
        see `A0_0`...
    """
    v = vk / epsilon

    rhoM = rBack(v,np.pi,epsilon)
    if NO_ADJ_CLASSES:
        rhoM = r(v, np.pi)

    R00th = np.abs(rhoM)**2
    
    return R00th

def R0_eps(vk, epsilon):
    """ epsilon^2-magnitude part of the integrand of the fraction of atoms 
    exiting the MZI if only n=0-input is populated 
    R0 = R0_0 + epsilon^2 R0_eps (= 1 - Loss)
    
    Parameters
    ----------
        see `A0_0`...
    """
    phi = 0
    v = vk / epsilon

    tM = tBack(v, np.pi,epsilon)
    rhoM = rBack(v, np.pi,epsilon)

    g22 = gammaII(epsilon, v, np.pi/2)
    gM22 = gammaII(epsilon, v, np.pi)
    gM32 = gammaIJ(epsilon, v, np.pi)

    # helpers
    Abs_sq_rhoM = np.abs(rhoM)**2

    # 2nd order epsilon
    R02nd_term1 = 2 * np.real(-gM32 * rhoM * np.conj(tM))
    R02nd_term2 = 2 * Abs_sq_rhoM * np.real(2 * g22 + gM22)
    R02nd = R02nd_term1 + R02nd_term2

    return R02nd 

def R10_eps(vk, epsilon):
    """epsilon^2-magnitude part of the integrand of the first-order coherence
     (~psi_1^dag*psi_0) contribution to the atom fraction exiting the MZI
     (both inputs populated) R10 = epsilon**2 R10_eps
    
    Parameters
    ----------
        see `A0_0`...
    """
    phi = 0
    v = vk / epsilon

    tB = tBack(v,np.pi/2,epsilon)
    rhoB = rBack(v, np.pi/2,epsilon)
    tM = tBack(v, np.pi,epsilon)
    rhoM = rBack(v, np.pi,epsilon)

    g33 = gammaII(epsilon, -v, np.pi/2)
    g32 = gammaIJ(epsilon, v, np.pi/2)
    gM22 = gammaII(epsilon, v, np.pi)
    gM32 = gammaIJ(epsilon, v, np.pi)

    # helpers
    Abs_sq_rhoM = np.abs(rhoM)**2
    conj_rhoB = np.conj(rhoB)
    exp_iphi = np.exp(1j * phi)
    conj_exp_iphi = np.conj(exp_iphi)

    term1_inner1 = Abs_sq_rhoM * np.real(g33 + gM22)
    term1_inner2 = np.real(gM32 * rhoM * np.conj(tM))
    term1 = 4 * np.real(tB * conj_rhoB * (term1_inner1 - term1_inner2))

    term2 = 2 * Abs_sq_rhoM * np.real(g32)
    term3 = -exp_iphi * 2 * np.real(rhoM**2 * g32 * np.conj(tB)**2)
    term4 = conj_exp_iphi * 2 * np.real(conj_rhoB**2 * g32 * rhoM**2)

    return (term1 + term2 + term3 + term4) #* np.exp(-1j * phi0)

def A0Deriv_eps(vk, epsilon):
    """derivative of A0_eps w.r.t. the interferometer phase phi; 
    dA0/dPhi(vk,phi=0) = epsilon^2 A0Deriv_eps
    
    Parameters
    ----------
        see `A0_0`...
    """
    phi = 0
    v = vk / epsilon

    tB = tBack(v,np.pi/2,epsilon)
    rhoB = rBack(v, np.pi/2,epsilon)
    rhoM = rBack(v, np.pi,epsilon)

    g22 = gammaII(epsilon, v, np.pi/2)
    g32 = gammaIJ(epsilon, v, np.pi/2)

    # helpers
    conj_tB = np.conj(tB)
    conj_rhoM = np.conj(rhoM)
    conj_g32 = np.conj(g32)
    exp_iphi = np.exp(1j * phi)        

    # 2nd order epsilon phase sensitive part
    inner1 = rhoB * (2 * np.real(g22) * tB + g32 * rhoB) - tB**2 * conj_g32
    term1ph = -4 * np.real(
        1j * exp_iphi * rhoB * inner1 * conj_tB * conj_rhoM**2
    )     

    A0Deriv2ndPhase = term1ph

    return A0Deriv2ndPhase

def A10Deriv_0(vk, epsilon, NO_ADJ_CLASSES=False):
    """derivative of A10_0 w.r.t. the interferometer phase phi; 
    dA0/dPhi(vk,phi=0) = epsilon^2 A0Deriv_eps;
    dA10/dPhi(v,phi=0,phi_0=0) = A10Deriv_0 + epsilon^2 A10Deriv_eps
    (phi_0 phase of first beam splitter pulse)
    
    Parameters
    ----------
        see `A0_0`...
    """
    phi = 0
    phi0 = 0
    v = vk / epsilon

    tB = tBack(v,np.pi/2,epsilon)
    rhoB = rBack(v, np.pi/2,epsilon)
    rhoM = rBack(v, np.pi,epsilon)

    if NO_ADJ_CLASSES:
        tB = t(v, np.pi/2)
        rhoB = r(v, np.pi/2)
        rhoM = r(v, np.pi)

    # helpers
    conj_tB = np.conj(tB)
    conj_rhoM = np.conj(rhoM)
    exp_iphi = np.exp(1j * phi)
    conj_exp_iphi = np.conj(exp_iphi)

    term02 = 2 * 1j * (
        exp_iphi * tB**2 - conj_exp_iphi * rhoB**2
    ) * rhoB * conj_tB * conj_rhoM**2
    A100th = term02 

    return (A100th) # * np.exp(-1j * phi0)

def A10Deriv_eps(vk, epsilon):
    """derivative of A10_eps w.r.t. the interferometer phase phi; 
    dA0/dPhi(vk,phi=0) = epsilon^2 A0Deriv_eps;
    dA10/dPhi(v,phi=0,phi_0=0) = A10Deriv_0 + epsilon^2 A10Deriv_eps
    (phi_0 phase of first beam splitter pulse)
    
    Parameters
    ----------
        see `A0_0`...
    """
    phi = 0
    phi0 = 0
    v = vk / epsilon

    tB = tBack(v,np.pi/2,epsilon)
    rhoB = rBack(v, np.pi/2,epsilon)
    tM = tBack(v, np.pi,epsilon)
    rhoM = rBack(v, np.pi,epsilon)
    Bbalance = 2 * np.abs(rhoB)**2 - 1

    g22 = gammaII(epsilon, v, np.pi/2)
    g33 = gammaII(epsilon, -v, np.pi/2)
    g32 = gammaIJ(epsilon, v, np.pi/2)
    g23 = gammaIJ(epsilon, -v, np.pi/2)
    gM22 = gammaII(epsilon, v, np.pi)
    gM33 = gammaII(epsilon, -v, np.pi)
    gM32 = gammaIJ(epsilon, v, np.pi)
    gM23 = gammaIJ(epsilon, -v, np.pi)

    # helpers
    conj_tB = np.conj(tB)
    conj_rhoM = np.conj(rhoM)
    conj_rhoB = np.conj(rhoB)
    conj_g22 = np.conj(g22)
    conj_g32 = np.conj(g32)
    conj_gM23 = np.conj(gM23)
    conj_tM = np.conj(tM)
    conj_gM33 = np.conj(gM33)
    exp_iphi = np.exp(1j * phi)
    conj_exp_iphi = np.conj(exp_iphi)

    term21 = 4 * rhoB * conj_tB * conj_rhoM**2 * (
        1j * exp_iphi * tB**2 * np.real(g22 + g33)
        - 1j * conj_exp_iphi * rhoB**2 * (g33 + conj_g22)        
    )

    term22_sub1 = -2 * rhoB * (gM32 - conj_gM23) * conj_tB * conj_tM * conj_rhoM
    term22_sub2 = -Bbalance * (g23 - conj_g32) * conj_rhoM**2
    term22_sub3 = 2 * rhoB * (gM22 + conj_gM33) * conj_tB * conj_rhoM**2
    term22 = 1j * (exp_iphi * tB**2 - conj_exp_iphi * rhoB**2) * (
        term22_sub1 + term22_sub2 + term22_sub3
    )
    term23 = 4 * tB * (g23 - conj_g32) * conj_tB * np.real(
        1j * exp_iphi * rhoM**2 * conj_rhoB**2
    )

    A10Deriv2ndPhase = term21 + term22 + term23
    
    return (A10Deriv2ndPhase)# * np.exp(-1j * phi0)

#### Phase uncertainty ####

def phaseUncertainty(N: int, mu, alpha, alignWithXYPlane=False, 
                     NO_ADJ_CLASSES=False):
    """analytically calculated, secular-term-free phase uncertainty for an MZI 
    that is driven by first-order Bragg diffraction and operated with a 
    One-Axis-Twisted (OAT) state that has been created from a spin coherent 
    state initially pointing in S_x-direction and - after OAT-dynamics - rotated
    around the S_x-axis about the angle alpha

    Parameters
    ----------
    N : int
        atom number
    mu : float
        twisting strength mu = 2 chi
    alpha : float
        subsequent rotation around the S_x axis
    alignWithXYPlane : bool, optional
        if set to True have alpha=-alpha0, i.e., rotating OAT ellipse onto 
        equator, by default False
    NO_ADJ_CLASSES : bool, optional
        specifies whether the adjacent classes n=-1,2 are taken into account 
        (False) or not (True), by default False

    Returns
    -------
    ndarray
        phase uncertainty for the OAT state polarized in S_x direction for given
        weighted quantities calculated via `weighting(epsilon_val, sigma_vk)` 
        in dependence on epsilon_val
    """
    wed_A0 = wed_A0_0 + epsilon_val**2 * wed_A0_eps
    wed_A10 = wed_A10_0 + epsilon_val**2 * wed_A10_eps
    wed_R0 = wed_R0_0 + epsilon_val**2 * wed_R0_eps
    wed_A0Deriv = epsilon_val**2 * wed_A0Deriv_eps
    
    wed_A10Deriv = wed_A10Deriv_0 + epsilon_val**2 * wed_A10Deriv_eps
    wed_R10 = epsilon_val**2 * wed_R10_eps

    if NO_ADJ_CLASSES:
        wed_A0 = wed_A0_0
        wed_A10 = wed_A10_0
        wed_R0 = wed_R0_0
        wed_A0Deriv = epsilon_val**2 * 0
        wed_A10Deriv = wed_A10Deriv_0 
        wed_R10 = epsilon_val**2 * 0

    return np.sqrt(
        (np.real(wed_A0Deriv) * N/2 
         + np.real(wed_A10Deriv) * mean_jx(N, mu))**(-2) * (
        (np.real(wed_A10)**2 * var_jx(N, mu) 
         + np.imag(wed_A10)**2 * var_jy(N, mu, alpha, alignWithXYPlane)) 
        + np.real(wed_A0)**2 * var_jz(N, mu, alpha, alignWithXYPlane) 
        + (2 * np.imag(wed_A10) * np.real(wed_A0) 
           * REcov_jyjz(N, mu, alpha, alignWithXYPlane))
        + (N / 4 * (np.real(wed_R0) - np.real(wed_A0)**2 - np.abs(wed_A10)**2) 
           + mean_jx(N, mu) * np.real(wed_R10) / 2)
        )
    )

def PScontributions(vk, epsilon):
    """collection of all functions that have to be weighted (integrated over) 
    with the momentum distribution

    Parameters
    ----------
    vk : float
        dimensionless Doppler detuning (\nu_k in units of recoil frequency 
        \omega_k)) 
    epsilon : float
        dimensionless Rabi frequency (\Omega_0 in units of recoil frequency 
        \omega_k)

    Returns
    -------
    ndarray
       array of functions to be weighted evaluated at (vk, epsilon)
    """
    funcs = [A0_0, A0_eps, A10_0, A10_eps, R0_0, R0_eps, 
             R10_eps, A0Deriv_eps, A10Deriv_0, A10Deriv_eps]
    return np.array([f(vk, epsilon) for f in funcs])

#### Weighting ####

def wed_quantities(epsilon, sigma_vk):
    """weighting the functions collected by `PScontributions(vk, epsilon)` with 
    the initial momentum distribution of the atoms  

    Parameters
    ----------
    epsilon : float
        dimensionless Rabi frequency (\Omega_0 in units of recoil frequency 
        \omega_k)
    sigma_vk : float
        standard deviation of the momentum distribution in units of 
        dimensionless doppler detuning (vk = \nu_k / \omega_k)

    Returns
    -------
    ndarray
        weighted functions as collected by `PScontributions(vk, epsilon)`
    """   
    def integrand(vk):
        return PScontributions(vk, epsilon) * gaussian(vk, sigma_vk)

    # using np.trapz
    n_trapz = 2**7
    v_wk_val = np.linspace(-5*sigma_vk, 5*sigma_vk, n_trapz)
    y = np.array([integrand(vk) for vk in v_wk_val]) # Shape: (n_trapz, ...)
    result = np.trapezoid(y, x=v_wk_val, axis=0)

    return result

# calculate weighted quantities
def weighting(epsilon_val, sigma_vk):
    """parallelized calling of `PScontributions(vk, epsilon)` to calculate 
    weighted functions

    Parameters
    ----------
    epsilon_val : ndarray
        values for the dimensionless Rabi frequency (\Omega_0 in units of 
        recoil frequency \omega_k)
    sigma_vk : float
        standard deviation of the momentum distribution

    Returns
    -------
    ndarray
        weighted functions as collected by `PScontributions(vk, epsilon)` in 
        dependence on epsilon_val
    """  
    global wed_A0_0, wed_A0_eps, wed_A10_0, wed_A10_eps, wed_R0_0, wed_R0_eps
    global wed_R10_eps, wed_A0Deriv_eps, wed_A10Deriv_0, wed_A10Deriv_eps

    # parallelized computation of weighted quantities for eps values
    with mp.Pool() as pool:
        inputs = zip(epsilon_val, (sigma_vk,) * len(epsilon_val))
        wed_results = pool.starmap(wed_quantities, inputs)
        pool.close()
        pool.join()

    wed_A0_0, wed_A0_eps, wed_A10_0, wed_A10_eps, \
        wed_R0_0, wed_R0_eps, wed_R10_eps, wed_A0Deriv_eps, \
            wed_A10Deriv_0, wed_A10Deriv_eps = np.array(wed_results).T

############ Numerical results ############

#### Bragg Hamiltonian ####

def Hamiltonian(epsilon, vk):
    """Hamiltonian describing resonant first-order Bragg diffraction for system 
    with dimensionless time \lambda = time * \omega_k and initial laser pulse 
    phase \theta = 0

    Parameters
    ----------
    epsilon : float
        dimensionless Rabi frequency (\Omega_0 in units of recoil frequency 
        \omega_k)
    vk : float
        dimensionless Doppler detuning (\nu_k in units of recoil frequency 
        \omega_k)) 

    Returns
    -------
    ndarray
        Hamiltonian
    """
    return .5 * np.array([
        [12 + 5 * vk, epsilon, 0, 0, 0, 0],
        [epsilon, 4 + 3 * vk, epsilon, 0, 0, 0],
        [0, epsilon, vk, epsilon, 0, 0],
        [0, 0, epsilon, -vk, epsilon, 0],
        [0, 0, 0, epsilon, 4 - 3 * vk, epsilon],
        [0, 0, 0, 0, epsilon, 12 - 5 * vk]
        ], dtype=complex)

#### Phase uncertainty and its contributions A0, A10, R0, R10 ####

def A0_num(tBp, rhoBp, tBm, rhoBm, rhoMp, rhoMm, as_deriv = False):
    """numerical calculation of the population-difference contribution to the 
    interference signal A0(vk,phi=0) or (if as_deriv) its derivative. 
    Transmission and reflection coefficients are determined numerically using
    `solve_ode()`

    Parameters
    ----------
    tBp : float
        transmission coefficient beam splitter pulse transition n=1 -> n=1
    rhoBp : float
        reflection coefficient mirror pulse transition n=1 -> n=0
    tBm : float
        transmission coefficient beam splitter pulse transition n=0 -> n=0
    rhoBm : float
        reflection coefficient mirror pulse transition n=0 -> n=1
    rhoMp : float
        reflection coefficient mirror pulse transition n=1 -> n=0
    rhoMm : float
        reflection coefficient mirror pulse transition n=0 -> n=1
    as_deriv : bool, optional
        determines whether A0(vk,phi=0) (False) or its derivative w.r.t. the 
        interferometer phase phi is returned, by default False

    Returns
    -------
    float
        population-difference contribution to the interference 
        signal A0(vk,phi=0) or (if as_deriv) its derivative w.r.t. the 
        interferometer phase phi
    """
    phi = 0
    
    if as_deriv:
        cross_term = 2 * np.real(
            1j * np.exp(1j * phi) * tBm * rhoMm * np.conj(rhoBm) *
            (-rhoBp * np.conj(tBm) + tBp * np.conj(rhoBm)) *
            np.conj(rhoMp)
        )
        return cross_term

    term1 = (np.abs(tBp)**2 - np.abs(rhoBp)**2) * np.abs(tBm * rhoMm)**2
    term2 = (np.abs(rhoBm)**2 - np.abs(tBm)**2) * np.abs(rhoBm * rhoMp)**2
    cross_term = 2 * np.real(
        np.exp(1j * phi) * tBm * rhoMm * np.conj(rhoBm) *
        (-rhoBp * np.conj(tBm) + tBp * np.conj(rhoBm)) *
        np.conj(rhoMp)
    )
    return term1 + term2 + cross_term

def A10_num(tBp, rhoBp, tBm, rhoBm, rhoMp, rhoMm, as_deriv = False):
    """numerical calculation of the first-order-coherence contribution to the 
    interference signal A0(vk,phi=0) or (if as_deriv) its derivative.
    
    Parameters
    ----------
        see `A0_num`...
    """
    phi = 0
    phi0 = 0

    if as_deriv:
        phase_factor = np.exp(-1j * phi0)
        term3 = -1j * np.exp(-1j * phi) * rhoBm * rhoMp * np.conj(rhoBp) * (
            rhoBm * np.conj(tBp) - tBm * np.conj(rhoBp)
        ) * np.conj(rhoMm)
        term4 = 1j * np.exp(1j * phi) * tBm * rhoMm * np.conj(tBp) * (
            -rhoBp * np.conj(tBm) + tBp * np.conj(rhoBm)
        ) * np.conj(rhoMp)
        
        return phase_factor * (term3 + term4)

    phase_factor = np.exp(-1j * phi0)
    term1 = rhoBm * np.abs(rhoMp)**2 * (np.abs(rhoBm)**2 
                                        - np.abs(tBm)**2) * np.conj(tBp)
    term2 = tBm * np.abs(rhoMm)**2 * np.conj(rhoBp) * (np.abs(tBp)**2 
                                                       - np.abs(rhoBp)**2)
    term3 = np.exp(-1j * phi) * rhoBm * rhoMp * np.conj(rhoBp) * (
        rhoBm * np.conj(tBp) - tBm * np.conj(rhoBp)
    ) * np.conj(rhoMm)
    term4 = np.exp(1j * phi) * tBm * rhoMm * np.conj(tBp) * (
        -rhoBp * np.conj(tBm) + tBp * np.conj(rhoBm)
    ) * np.conj(rhoMp)
    
    return phase_factor * (term1 + term2 + term3 + term4)

def R0_num(tBp, rhoBp, tBm, rhoBm, rhoMp, rhoMm):
    """numerical calculation of the fraction of atoms exiting the MZI if only 
    n=0-input is populated R0(vk) (= 1 - Loss)
    
    Parameters
    ----------
        see `A0_num`...
    """
    phi = 0

    term1 = (np.abs(tBm)**2 + np.abs(rhoBm)**2) * np.abs(rhoBm * rhoMp)**2
    term2 = (np.abs(tBp)**2 + np.abs(rhoBp)**2) * np.abs(tBm * rhoMm)**2
    cross_term = 2 * np.real(
        np.exp(1j * phi) * tBm * rhoMm * np.conj(rhoBm) *
        (rhoBp * np.conj(tBm) + tBp * np.conj(rhoBm)) *
        np.conj(rhoMp)
    )
    return term1 + term2 + cross_term

def R10_num(tBp, rhoBp, tBm, rhoBm, rhoMp, rhoMm):
    """numerical calculation of the first-order coherence (~psi_1^dag*psi_0) 
    contribution to the atom fraction exiting the MZI (both inputs populated) 
    R10(vk, phi=0, phi_0=0)
    
    Parameters
    ----------
        see `A0_num`...
    """
    phi = 0
    phi0 = 0

    phase_factor = np.exp(-1j * phi0)
    
    term1 = rhoBm * (np.abs(tBm)**2 
                     + np.abs(rhoBm)**2) * np.abs(rhoMp)**2 * np.conj(tBp)
    term2 = tBm * (np.abs(tBp)**2 
                   + np.abs(rhoBp)**2) * np.abs(rhoMm)**2 * np.conj(rhoBp)
    
    term3 = np.exp(-1j * phi) * rhoBm * rhoMp * np.conj(rhoBp) * (
        rhoBm * np.conj(tBp) + tBm * np.conj(rhoBp)
    ) * np.conj(rhoMm)
    
    term4 = np.exp(1j * phi) * tBm * rhoMm * np.conj(tBp) * (
        rhoBp * np.conj(tBm) + tBp * np.conj(rhoBm)
    ) * np.conj(rhoMp)
    
    return phase_factor * (term1 + term2 + term3 + term4)

def PScontributions_num(vk, epsilon, time_dep_pulse=False):
    """
    Parameters
    ----------
    vk : float
        dimensionless Doppler detuning (\nu_k in units of recoil frequency 
        \omega_k)) 
    epsilon : float
        dimensionless Rabi frequency (\Omega_0 in units of recoil frequency 
        \omega_k)
    time_dep_pulse : bool, optional
        determines whether box pulses (False) or Blackman pulses (True) 
        are used, by default False

    Returns
    -------
    ndarray
        returns contributions to phase uncertainty at (phi = 0, phi0 = 0) in 
        following order: [A0, A10, R0, R10, A0Deriv, A10Deriv] 
        in dependence on vk = 2 p / (\hbar k)
    """
        
    # transmission and reflection coefficients (computed numerically via Runge-Kutta)
    Bsol_p, Msol_p = solve_ode(vk, epsilon, 
                               np.array([0,0,1,0,0,0], dtype=complex), 
                               time_dep_pulse).T
    Bsol_m, Msol_m = solve_ode(vk, epsilon, 
                               np.array([0,0,0,1,0,0], dtype=complex), 
                               time_dep_pulse).T

    tBp = Bsol_p[2]
    rhoBp = Bsol_p[3]
    tBm = Bsol_m[3]
    rhoBm = Bsol_m[2]
    rhoMp = Msol_p[3]
    rhoMm = Msol_m[2]

    # contributions
    A0_val = A0_num(tBp, rhoBp, tBm, rhoBm, rhoMp, rhoMm)
    A10_val = A10_num(tBp, rhoBp, tBm, rhoBm, rhoMp, rhoMm)
    R0_val = R0_num(tBp, rhoBp, tBm, rhoBm, rhoMp, rhoMm)
    R10_val = R10_num(tBp, rhoBp, tBm, rhoBm, rhoMp, rhoMm)
    A0Deriv_val = A0_num(tBp, rhoBp, tBm, rhoBm, rhoMp, rhoMm, as_deriv=True)
    A10Deriv_val = A10_num(tBp, rhoBp, tBm, rhoBm, rhoMp, rhoMm, as_deriv=True)

    return np.array([A0_val, A10_val, R0_val, R10_val, A0Deriv_val, A10Deriv_val])

def gaussian(vk, sigma_v):
    """Gaussian momentum distribution"""
    return np.exp(- (vk / sigma_v)**2 / 2) / (np.sqrt(2*np.pi) * sigma_v)

def phaseUncertaintySQ_num(N: int, mu, alpha, alignWithXYPlane=False):
    """numerically calculated, squared phase uncertainty for an MZI that is 
    driven by first-order Bragg diffraction and operated with a One-Axis-Twisted 
    (OAT) state that has been created from a spin coherent state initially 
    pointing in S_x-direction and - after OAT-dynamics - rotated around the 
    S_x-axis about the angle alpha

    Parameters
    ----------
    N : int
        atom number
    mu : float
        twisting strength mu = 2 chi
    alpha : float
        subsequent rotation around the S_x axis
    alignWithXYPlane : bool, optional
        if set to True have alpha=-alpha0, i.e., rotating OAT ellipse onto 
        equator, by default False

    Returns
    -------
    ndarray
        phase uncertainty for the OAT state polarized 
        in S_x direction for given, numerically -via `solve_ode()`- 
        calculated quantities weighted via 
        `weighting_num(epsilon_val, sigma_vk)` in dependence on epsilon_val
    """
    return (np.real(wed_A0Deriv_num) * N/2 + np.real(wed_A10Deriv_num) * mean_jx(N, mu))**(-2) * (
            (np.real(wed_A10_num)**2 * var_jx(N, mu) 
             + np.imag(wed_A10_num)**2 * var_jy(N, mu, alpha, alignWithXYPlane))
        + np.real(wed_A0_num)**2 * var_jz(N, mu, alpha, alignWithXYPlane)
        + 2 * np.imag(wed_A10_num) * np.real(wed_A0_num) * REcov_jyjz(N, mu, alpha, alignWithXYPlane) 
        + (N / 4 * (np.real(wed_R0_num) - np.real(wed_A0_num)**2 - np.abs(wed_A10_num)**2) 
           + mean_jx(N, mu) * np.real(wed_R10_num) / 2)
    )

#### Complex ODE solver ####

def dgdt(lmda, g, vk, epsilon, time_dep_pulse=False, lmda_final=0): 
    """right side of Heisenbergg equation of motion for resonant first-order 
    Bragg diffraction

    Parameters
    ----------
    lmda : float
        dimensionles time
    g : ndarray
        vector of couplings to solve for
    vk : float
        dimensionless Doppler detuning (\nu_k in units of recoil frequency 
        \omega_k)) 
    epsilon : float
        dimensionless Rabi frequency (\Omega_0 in units of recoil frequency 
        \omega_k)
    phi : float
        light pulse phase
    time_dep_pulse : bool, optional
        determines whether box- (False) or Blackman-ramp is chosen for the 
        laser pulses (True) , by default False
    lmda_final : int, optional
        final dimensionless time (only needed if time_dep_pulse==True), 
        by default 0

    Returns
    -------
    ndarray
        right side of Heisenberggs equation of motion
    """
    if not time_dep_pulse:
        return -1j * Hamiltonian(epsilon, vk) @ g
    
    # blackman pulse pulse
    def blackman_eps(lmda, epsilon, lmda_final):
        """
        Returns
        -------
        ndarray
            time-dependent intensity ramp of a Blackman pulse
        """
        return epsilon * (.42 - .5 * np.cos(2 * np.pi * lmda/lmda_final) 
                          + .08 * np.cos(4 * np.pi * lmda/lmda_final)) 

    return -1j * Hamiltonian(blackman_eps(lmda, epsilon, lmda_final), vk) @ g

def solve_ode(vk, epsilon, g0, time_dep_pulse=False, single_eval=False):
    """solving the Heisenberg equation of motion corresponding to `Hamiltonian()`
    using standard Runge-Kutta method of numpy callable `solve_ivp`

    Parameters
    ----------
    vk : float
        dimensionless Doppler detuning (\nu_k in units of recoil frequency 
        \omega_k)) 
    epsilon : float
        dimensionless Rabi frequency (\Omega_0 in units of recoil frequency 
        \omega_k)
    g0 : ndarray
        initial conditions
    time_dep_pulse : bool, optional
        determines whether box- (False) or Blackman-ramp is chosen for the 
        laser pulses (True) , by default False

    Returns
    -------
    ndarray
        time_dep_pulse==False
            solutions at times lmda = np.pi/(2*epsilon) and np.pi/epsilon
        time_dep_pulse==True
            solutions at times lmda_final = np.pi / (2 * .42 * epsilon) and
            np.pi / (.42 * epsilon)
    """

    if single_eval:
        tau_final = 2 * np.pi

        if not time_dep_pulse:
            lmda_final = tau_final / epsilon
            lmda_val = np.linspace(0, lmda_final, 200)

            sol = solve_ivp(dgdt, (0, lmda_final), g0, 
                            args=(vk, epsilon), 
                            method='RK45', 
                            t_eval=lmda_val) 
            return sol.y
        
        lmda_final = tau_final / (.42 * epsilon)
        lmda_val = np.linspace(0, lmda_final, 200)
        sol_blackman = solve_ivp(dgdt, (0, lmda_final), g0, 
                                args=(vk, epsilon, time_dep_pulse, lmda_final), 
                                method='RK45', 
                                t_eval=lmda_val)
        return sol_blackman.y

    if not time_dep_pulse:
        sol = solve_ivp(dgdt, (0, np.pi/epsilon), g0, 
                        args=(vk, epsilon), method='RK45', 
                        t_eval=[np.pi/(2*epsilon), np.pi/epsilon]) 
        return sol.y
    
    lmda_final = np.pi / (2 * .42 * epsilon)
    sol_blackman_BS = solve_ivp(dgdt, (0, lmda_final), g0, 
                                args=(vk, epsilon, time_dep_pulse, lmda_final), 
                                method='RK45', t_eval=[lmda_final]) 

    lmda_final = np.pi / (.42 * epsilon)
    sol_blackman_MR = solve_ivp(dgdt, (0, lmda_final), g0, 
                                args=(vk, epsilon, time_dep_pulse, lmda_final), 
                                method='RK45', t_eval=[lmda_final])

    return np.array([sol_blackman_BS.y[:,0], sol_blackman_MR.y[:,0]]).T

#### Weighting ####

def wed_quantities_num(epsilon, sigma_vk, time_dep_pulse):
    """weighting the functions collected by `PScontributions_num()` with 
    the initial momentum distribution of the atoms using `np.trapezoid()`

    Parameters
    ----------
    epsilon : float
        dimensionless Rabi frequency (\Omega_0 in units of recoil frequency 
        \omega_k)
    sigma_vk : float
        standard deviation of the momentum distribution
    time_dep_pulse : bool, optional
        determines whether box pulses (False) or Blackman pulses (True) 
        are used, by default False

    Returns
    -------
    ndarray
        weighted functions as collected by `PScontributions_num()`
    """  
    def integrand(vk):
        return PScontributions_num(vk, epsilon, time_dep_pulse) * gaussian(vk, sigma_vk)

    # using np.trapz
    n_trapz = 2**7
    v_wk_val = np.linspace(-5*sigma_vk, 5*sigma_vk, n_trapz)
    y = np.array([integrand(vk) for vk in v_wk_val])
    result = np.trapezoid(y, x=v_wk_val, axis=0)

    return result

# calculate weighted quantities
def weighting_num(epsilon_val, sigma_vk, parallelize, time_dep_pulse):
    """parallelized (in series, if not parallelize) calling of 
    `PScontributions_num()` to calculate weighted functions

    Parameters
    ----------
    epsilon_val : ndarray
        values for the dimensionless Rabi frequency (\Omega_0 in units of 
        recoil frequency \omega_k)
    sigma_vk : float
        standard deviation of the momentum distribution
    parallelize : bool
        determines whether weighting is done in parallelized or not
    time_dep_pulse : bool, optional
        determines whether box- (False) or Blackman-ramp is chosen for the 
        laser pulses (True) , by default False

    Returns
    -------
    ndarray
        weighted functions as collected by `PScontributions_num()` in 
        dependence on epsilon_val
    """  
    global wed_A0_num, wed_A10_num, wed_R0_num
    global wed_R10_num, wed_A0Deriv_num, wed_A10Deriv_num

    if parallelize:
        # parallelized computation of weighted quantities for eps values
        with mp.Pool() as pool:
            inputs = zip(epsilon_val, (sigma_vk,) * len(epsilon_val), 
                         (time_dep_pulse, ) * len(epsilon_val)) # similar to [(eps, sigma_vk) for eps in epsilon_val]
            wed_results = pool.starmap(wed_quantities_num, inputs)
            pool.close()
            pool.join()
    
    else:
        wed_results = []
        for eps in epsilon_val:
            print(fr'eps = {eps:.2f}      ')
            wed_results.append(wed_quantities_num(eps, sigma_vk))
        wed_results = np.array(wed_results)
    
    wed_A0_num, wed_A10_num, wed_R0_num, wed_R10_num, \
        wed_A0Deriv_num, wed_A10Deriv_num = np.array(wed_results).T

#### no velocity selectivity ####

def phaseUncertaintySQ_noVS(N: int, mu, alpha, alignWithXYPlane=False):
    """squared phase uncertainty for an MZI - with NO VELOCITY SELECTIVITY (VS) 
    present - that is driven by first-order Bragg diffraction and operated with 
    a One-Axis-Twisted (OAT) state that has been created from a spin coherent 
    state initially pointing in S_x-direction and - after OAT-dynamics - rotated
    around the S_x-axis about the angle alpha

    Parameters
    ----------
    N : int
        atom number
    mu : float
        twisting strength mu = 2 chi
    alpha : float
        subsequent rotation around the S_x axis
    alignWithXYPlane : bool, optional
        if set to True have alpha=-alpha0, i.e., rotating OAT ellipse onto 
        equator, by default False

    Returns
    -------
    float
        squared phase uncertainty for the OAT state polarized in S_x direction 
        for given weighted quantities calculated via 
        `weighting(epsilon_val, sigma_vk)` in dependence on epsilon_val
    """
    # A0_noVS =  -1 + epsilon_val**2 / 8 * (3 - np.sqrt(2) * np.cos(np.pi / epsilon_val))
    # A10_noVS = 1j * epsilon_val**2 / 16 * (np.pi + 2 * np.sqrt(2) * np.cos(np.pi / epsilon_val))
    # R0_noVS = 1 + epsilon_val**2 / 8 * (-3 + np.sqrt(2) * np.cos(np.pi / epsilon_val))
    # R10_noVS = np.sqrt(2) / 8 * epsilon_val**2 * np.sin(np.pi / epsilon_val)
    # A0Deriv_noVS = - epsilon_val**2 * np.sin(np.pi / epsilon_val) / (np.sqrt(2) * 8)
    # A10Deriv_noVS = -1 + epsilon_val**2 / 8 * (3 - np.sqrt(2) * np.cos(np.pi / epsilon_val)) # = A0_noVS

    # Alternatively
    A0_noVS = A0_0(0,epsilon_val) + epsilon_val**2 * A0_eps(0,epsilon_val)
    A10_noVS = A10_0(0,epsilon_val) + epsilon_val**2 * A10_eps(0,epsilon_val)
    R0_noVS = R0_0(0,epsilon_val) + epsilon_val**2 * R0_eps(0,epsilon_val)
    R10_noVS = epsilon_val**2 * R10_eps(0,epsilon_val)
    A0Deriv_noVS = epsilon_val**2 * A0Deriv_eps(0,epsilon_val)
    A10Deriv_noVS = A10Deriv_0(0,epsilon_val) + epsilon_val**2 * A10Deriv_eps(0,epsilon_val)

    return (np.real(A0Deriv_noVS) * N/2 + np.real(A10Deriv_noVS) * mean_jx(N, mu))**(-2) * (
            (np.real(A10_noVS)**2 * var_jx(N, mu) + np.imag(A10_noVS)**2 * var_jy(N, mu, alpha, alignWithXYPlane))
        + np.real(A0_noVS)**2 * var_jz(N, mu, alpha, alignWithXYPlane) 
        + 2 * np.imag(A10_noVS) * np.real(A0_noVS) * REcov_jyjz(N, mu, alpha, alignWithXYPlane) 
        + N / 4 * (np.real(R0_noVS) - np.real(A0_noVS)**2 - np.abs(A10_noVS)**2) 
        + mean_jx(N, mu) * np.real(R10_noVS) / 2
    )

#### orientation/squeezing optimisation ####

def orientation_opt_PS_SQ(N: int, mu, alpha_min, alpha_max, n_alpha):
    """calls `phaseUncertainty()` for compensation-rotation angles alpha 
    between alpha_min and alpha_max and finds the minimum 

    Parameters
    ----------
    N : int
        atom number
    mu : float
        twisting strength mu = 2 chi
    alpha_min : float
        lower bound for alpha values
    alpha_max : float
        upper bound for alpha values
    n_alpha : int
        number of alpha values iterated over 

    Returns
    -------
    ndarray
        optimal inclination values (phi = alpha + alpha0) and 
        min_{alpha_min <= alpha <= alpha_max} `phaseUncertainty()` in dependence 
        on epsilon_val
    """
    alpha_values = np.linspace(alpha_min, alpha_max, n_alpha)
    PS_values = np.array([phaseUncertainty(N, mu, phi)**2 for phi in alpha_values])
    alpha_opt_args = np.argmin(PS_values, axis=0)
    print(fr"$alpha_min = {np.min(alpha_values[alpha_opt_args])}, \
          alpha_max = {np.max(alpha_values[alpha_opt_args])}")
    optimised_PS = PS_values[alpha_opt_args, np.arange(PS_values.shape[1])]

    opt_inclination = alpha_values[alpha_opt_args] + alpha_0(N, mu)

    return opt_inclination, optimised_PS

def squeezing_opt_PS_SQ(N: int, mu_min, mu_max, n_mu, equator_phi=False):
    """calls `phaseUncertainty()` for twisting strengths mu 
    between mu_min and mu_max and finds the minimum. The compensation-rotation
    angle alpha is set to alpha=-alpha_0(N, mu=mu_opt(N)) for not equator_phi, 
    i.e. the OAT state is aligned with the equator is the twisting strength is 
    set to mu=mu_opt(N). For equator_phi, the angle is set according to the
    twisting strength that minimizes `phaseUncertainty()` mu_best, i.e., 
    alpha=alpha_0(N, mu_best)

    Parameters
    ----------
    N : int
        atom number
    mu_min : float
        lower bound for twisting strength
    mu_max : float
        upper bound for twisting strength
    n_mu : int
        number of mu values iterated over
    equator_phi : bool, optional
        determines whether alpha=-alpha_0(N, mu=mu_opt(N)) (False) or 
        alpha=alpha_0(N, mu_best) (True), by default False

    Returns
    -------
    ndarray
        optimal twisting strength chi = mu/2 and corresponding phase uncertainty
        squared; both in dependence on epsilon_val
    """
    mu_values = np.linspace(mu_min, mu_max, n_mu)
    if equator_phi:
        '''align ellipse with eqator for all mu'''
        PS_values = np.array([phaseUncertainty(N, mu, 0, alignWithXYPlane=True)**2 
                              for mu in mu_values])
    else:
        '''align ellipse with eqator only for the case where mu = mu_opt(N)'''
        PS_values = np.array([phaseUncertainty(N, mu, 
                                               alpha=-alpha_0(N, mu_opt(N)), 
                                               alignWithXYPlane=False)**2 
                              for mu in mu_values])

    mu_opt_args = np.argmin(PS_values, axis=0)
    print(fr"$mu_min = {np.min(mu_values[mu_opt_args])}, \
          mu_max = {np.max(mu_values[mu_opt_args])}")
    optimised_PS = PS_values[mu_opt_args, np.arange(PS_values.shape[1])]
    
    opt_twisting = mu_values[mu_opt_args]/2

    return opt_twisting, optimised_PS


############ Plotting ############

#### Rabi cycle ####

def plot_Rabi_cycle_momentum(epsilon, sigma_vk, time_dep_pulse=False):
    """plots a complete Rabi cycle in momentum space for an atom starting in 
    class n=0

    Parameters
    ----------
    epsilon : float
        dimensionless Rabi frequency (\Omega_0 in units of recoil frequency 
        \omega_k)
    sigma_vk : float
        standard deviation of the momentum distribution in units of 
        dimensionless Doppler detuning vk 
    time_dep_pulse : bool, optional
        determines whether box pulses (False) or Blackman pulses (True) 
        are used, by default False
    """
    ### parameters ###
    tau_final = 2 * np.pi
    lmda_val = np.linspace(0, tau_final /
                           (.42 * epsilon if time_dep_pulse else epsilon), 200)
    vk_val = np.linspace(- 3 * .16 - 1, 2 + 3 * .16, 200) * 2 
    g0 = np.array([0,0,0,1,0,0], dtype=complex)

    ### calculating numerical solution depending on time, mode and momentum ### 
    g_result = np.zeros((6, len(lmda_val), len(vk_val)), dtype=complex)

    progress = 0
    for i, vk in enumerate(vk_val):
        gaussians = np.ones_like(g_result[0,:,0], dtype=complex)
        for j in range(6):
            gaussians = gaussian(vk + (j - 3) * 2, sigma_vk)
            g_result[j,:,i] = np.abs(
                solve_ode(vk + (j - 3) * 2, epsilon, g0, 
                          time_dep_pulse=time_dep_pulse, single_eval=True)[j]
                )**2 * gaussians * 2 # last factor for conversion to units: probability/(\hbar k)
        progress += 1
        print(f'progress: {int(progress/len(vk_val)*100)}%', 
              end='\r', flush=True)

    g_result = g_result[1:5,:,:]
    g_result = np.sum(g_result, axis=0)

    X,Y = np.meshgrid(vk_val, lmda_val)
    Z = g_result

    fig, ax = plt.subplots(figsize=(3.452,1.7))
    cf = ax.contourf(X, Y, Z, levels=400, cmap='gist_heat_r')

    cf.set_rasterized(True)

    ax.grid(ls='--', axis='x')

    ax.set_ylabel(r'$\tau$')
    ax.set_yticks(np.array([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])/epsilon, 
                labels=['0', r'$\frac{{\pi}}{{2}}$', 
                        r'$\pi$', r'$\frac{{3\pi}}{{2}}$', r'$2\pi$'])
    ax.set_xticks(np.array([-1,0,1,2])*2, labels=[r'',r'',r'',r''])
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    cb = fig.colorbar(cf, ticks=[0,.6,1.2,1.8,2.4,3,3.6], pad=0.01)
    cb.set_label('probability density')
    cb.ax.set_yticklabels([0,.6,1.2,1.8,2.4,3.0,3.6])

    plt.savefig(im_path + "/Bragg_rabi_cycle.pdf")
    plt.show()

#### cropped vs full signal ####

def plot_cropped_vs_full_signal(epsilon, sigma_vk, T_Omega):
    """plots spatial distribution of atoms propagating through the MZI, as
    well as corresponding interference signals and phase uncertainties 
    for full and cropped detection

    Parameters
    ----------
    epsilon : float
        dimensionless Rabi frequency (\Omega_0 in units of recoil frequency 
        \omega_k)
    sigma_vk : float
        standard deviation of the momentum distribution in units of 
        dimensionless Doppler detuning vk = 2 p / (\hbar k)
    T_Omega : float
        pulse area associated with the interrogation time
    """
    def psi_p0(vk, phi2, epsilon, T_Omega, wL):
        """MZI time evolution in momentum space (modulo constant phase factors) of 
        atoms in class n=0 for p_resonant = 0, phi0=phi1=0, in dependence on phi2 
        (since the outer interferometric contributions do not have phi0-2phi1+phi2 
        as phase but something else)

        Parameters
        ----------
        vk : float
            dimensionless Doppler detuning (\nu_k in units of recoil frequency 
            \omega_k)) 
        phi2 : float
            phase of last beam splitter
        epsilon : float
            dimensionless Rabi frequency (\Omega_0 in units of recoil frequency 
            \omega_k)
        T_Omega : float
            pulse area associated with the interrogation time T
        wL : float
            detuning between the lasers in units of Rabi frequency \Omega_0

        Returns
        -------
        float
            MZI time evolution in momentum space of atoms in class n=0 
            (wave function psi0(p) = psi_p0 * initial_mom_mode_func)
        """
        v = vk / epsilon

        t0 = np.exp(- 1j * wL * np.pi/4) * t(v, np.pi/2)
        t1 = np.exp(- 1j * wL * np.pi/2) * t(v, np.pi)
        t2 = t0
        r0 = np.exp(- 1j * wL * np.pi/4) * r(v, np.pi/2)
        r1 = np.exp(- 1j * wL * np.pi/2) * r(v, np.pi)
        r2 = np.exp(- 1j * wL * np.pi/4) * np.exp(- 1j * phi2)*r(v, np.pi/2)
        
        U01 = - np.conj(r2) * r1 * np.conj(t0)
        U10 = - np.conj(t2 * r1) * r0
        U00 = np.exp(1j * (vk / epsilon) * T_Omega) * np.conj(t2 * t1 * t0)
        U11 = - np.exp(- 1j * (vk / epsilon) * T_Omega) * np.conj(r2) * t1 * r0
        
        def phi_all(vk, epsilon, T_Omega):
            v = vk/epsilon
            # return epsilon * v**2/2 * (2 * np.pi + T_Omega) + v * (np.pi + T_Omega)
            return (epsilon * v**2/2.) * (np.pi + T_Omega) + v * np.pi/4 + v * T_Omega

        return np.exp(-1j * phi_all(vk, epsilon, T_Omega)) * (U01 + U10 + U11 + U00)

    def psi_p1(vk, phi2, epsilon, T_Omega, wL):
        """MZI time evolution in momentum space (modulo constant phase factors) of 
        atoms in class n=1 for p_resonant = 0, phi0=phi1=0, in dependence on phi2 
        (since the outer interferometric contributions do not have phi0-2phi1+phi2 
        as phase but something else)

        Parameters
        ----------
        vk : float
            dimensionless Doppler detuning (\nu_k in units of recoil frequency 
            \omega_k)) 
        phi2 : float
            phase of last beam splitter
        epsilon : float
            dimensionless Rabi frequency (\Omega_0 in units of recoil frequency 
            \omega_k)
        T_Omega : float
            pulse area associated with the interrogation time T
        wL : float
            detuning between the lasers in units of Rabi frequency \Omega_0

        Returns
        -------
        float
            MZI time evolution in momentum space of atoms in class n=1 
            (wave function psi1(p) = psi_p1 * initial_mom_mode_func)
        """
        
        v = vk / epsilon

        t0 = np.exp(- 1j * wL * np.pi/4) * t(v, np.pi/2)
        t1 = np.exp(- 1j * wL * np.pi/2) * t(v, np.pi)
        t2 = t0
        r0 = np.exp(- 1j * wL * np.pi/4) * r(v, np.pi/2)
        r1 = np.exp(- 1j * wL * np.pi/2) * r(v, np.pi)
        r2 = np.exp(- 1j * wL * np.pi/4) * np.exp(- 1j * phi2)*r(v, np.pi/2)
        
        U01 = t2 * r1 * np.conj(t0)
        U10 = - r2 * np.conj(r1) * r0
        U00 = np.exp(1j * (vk / epsilon) * T_Omega) * r2 * np.conj(t1 * t0)
        U11 = np.exp(- 1j * (vk / epsilon) * T_Omega) * t2 * t1 * r0
        
        def phi_all(vk, epsilon, T_Omega):
            v = vk/epsilon
            # return epsilon * v**2/2 * (2 * np.pi + T_Omega) + v * (np.pi + T_Omega)
            return (epsilon * v**2/2.) * (np.pi + T_Omega) + v * np.pi/4 + v * T_Omega

        return np.exp(-1j * phi_all(vk, epsilon, T_Omega)) * (U01 + U10 + U11 + U00)

    def psi_abs_SQ(vk, phi2, epsilon):
        """Computes absolute square of the MZI time evolution for classes n=0,1 
        (full signals and cropped signals). The fast oscillating terms 
        (Ujj^*Uij, where i!=j) are already excluded!

        Parameters
        ----------
        vk : float
            dimensionless Doppler detuning (\nu_k in units of recoil frequency 
            \omega_k)) 
        phi2 : float
            phase of last beam splitter
        epsilon : float
            dimensionless Rabi frequency (\Omega_0 in units of recoil frequency 
            \omega_k)

        Returns
        -------
        ndarray
                Absolute square of the MZI time evolution for classes n=0,1 
                (full signals and cropped signals); given in form: 
                np.array([full_signal_abs_0, full_signal_abs_1, 
                        cropped_signal_abs_0, cropped_signal_abs_1])
                (i.e. |psi0_full|^2 = full_signal_abs_0 * initial_mom_distribution)
        """
        v = vk / epsilon

        t0 = np.exp(- 1j * wL * np.pi/4) * t(v, np.pi/2)
        t1 = np.exp(- 1j * wL * np.pi/2) * t(v, np.pi)
        t2 = t0
        r0 = np.exp(- 1j * wL * np.pi/4) * r(v, np.pi/2)
        r1 = np.exp(- 1j * wL * np.pi/2) * r(v, np.pi)
        r2 = np.exp(- 1j * wL * np.pi/4) * np.exp(- 1j * phi2)*r(v, np.pi/2)
        
        U01g = - np.conj(r2) * r1 * np.conj(t0)
        U10g = - np.conj(t2 * r1) * r0
        U00g = np.exp(1j * (vk / epsilon) * T_Omega) * np.conj(t2 * t1 * t0)
        U11g = - np.exp(- 1j * (vk / epsilon) * T_Omega) * np.conj(r2) * t1 * r0

        U01e = t2 * r1 * np.conj(t0)
        U10e = - r2 * np.conj(r1) * r0
        U00e = np.exp(1j * (vk / epsilon) * T_Omega) * r2 * np.conj(t1 * t0)
        U11e = np.exp(- 1j * (vk / epsilon) * T_Omega) * t2 * t1 * r0

        full_0 = np.conjugate(U01g + U10g)*(U01g + U10g) \
                + np.abs(U00g)**2 + np.abs(U11g)**2
        full_1 = np.conjugate(U01e + U10e)*(U01e + U10e) \
                + np.abs(U00e)**2 + np.abs(U11e)**2
        cropped_0 = np.conjugate(U01g + U10g)*(U01g + U10g)
        cropped_1 = np.conjugate(U01e + U10e)*(U01e + U10e)

        return np.array([full_0, full_1, cropped_0, cropped_1])

    ###### parameters ######
    wL = 1e7 # laser frequecy (doesn't do anything)
    dz = 2 * T_Omega / epsilon # distance that is covered after propagation with momentum \hbar k for time T

    kappa_val = np.linspace(-.5 * dz, 2.5 * dz, 2**7) # dimensionless position \kappa = z * k
    phi2_val = np.linspace(0, 2*np.pi, 2**7) 
    vk_val = np.linspace(-5 * sigma_vk, 5 * sigma_vk, 10000) # dimensionless momentum vk = v / \omega_k = 2 p / (\hbar k)

    results = np.zeros((len(kappa_val), len(phi2_val)), dtype=complex)

    ###### numerically calculating wave fct in position space ######

    def FT_integrand_0(vk, kappa, phi2):
        """
        Parameters
        ----------
        vk : float
            dimensionless Doppler detuning (\nu_k in units of recoil frequency 
            \omega_k)) 
        kappa : float
            dimensionless position variable z/dz
        phi2 : float
            phase of second beam splitter

        Returns
        -------
        float
            wave function in momentum space of atoms in momentum class n=0 after 
            the MZI sequence in 
            dependence on vk, kappa and phi2
        """
        return np.exp(1j * kappa * vk / 2) * psi_p0(vk, phi2, epsilon, T_Omega, wL) \
                * np.sqrt(gaussian(vk, sigma_vk))

    def FT_integrand_1(vk, kappa, phi2):
        """
        Parameters
        ----------
        vk : ndarray
            dimensionless Doppler detuning (\nu_k in units of recoil frequency \omega_k)) 
        kappa : ndarray
            dimensionless position variable z/dz
        phi2 : ndarray
            phase of second beam splitter

        Returns
        -------
        ndarray
            wave function in momentum space of atoms in momentum class n=1 after 
            the MZI sequence in dependence on vk, kappa and phi2
        """
        return np.exp(1j * kappa * vk / 2) \
                * psi_p1(vk - 2, phi2, epsilon, T_Omega, wL) \
                * np.sqrt(gaussian(vk - 2, sigma_vk))

    FT_integrand_arr_0 = FT_integrand_0(vk_val[None,None,:], 
                                        kappa_val[:,None,None], 
                                        phi2_val[None,:,None])
    FT_integrand_arr_1 = FT_integrand_1(vk_val[None,None,:] + 2, 
                                        kappa_val[:,None,None], 
                                        phi2_val[None,:,None])

    results_0 = np.trapezoid(FT_integrand_arr_0, x=vk_val, axis=2)
    results_1 = np.trapezoid(FT_integrand_arr_1, x=vk_val,  axis=2)

    X, Y = np.meshgrid(phi2_val, kappa_val)
    Zg = np.abs(results_0)**2 * dz / (4 * np.pi)
    Ze = np.abs(results_1)**2 * dz / (4 * np.pi) # factor dz is for conversion of prob density units to units of 1/dz;  1/(4 * np.pi) is needed to normalize wavefunction to 1


    ###### calculating interferometer signals ######

    def signal_integrand(vk, phi2):
        """integrand of interference signal for full and cropped signals in MZI 
        exits (momentum classes) n=0,1

        Parameters
        ----------
        vk : ndarray
            dimensionless Doppler detuning (\nu_k in units of recoil frequency 
            \omega_k)) 
        phi2 : ndarray
            phase of second beam splitter

        Returns
        -------
        ndarray
            integrand of interference signal. Form: full_signal_0, full_signal_1, 
            cropped_signal_0, cropped_signal_1
        """
        return psi_abs_SQ(vk, phi2, epsilon) * gaussian(vk, sigma_vk)

    signal_integrand_arr = signal_integrand(vk_val[:,None], phi2_val[None,:])
    signal_result = np.trapezoid(signal_integrand_arr, x=vk_val,  axis=1) # signal_result[0] is \psi_fullg(\phi) e.g.

    ###### calculating phase uncertainty for observable J3 ######
    def Vis(v):
        """
        Parameters
        ----------
        v : float
            dimensionless Doppler detuning (v = vk/epsilon)

        Returns
        -------
        float
            momentum-dependent visibility of full and cropped interference signal
        """
        f = np.sqrt(1 + v**2)
        cos_term = np.cos(0.25 * np.pi * f)
        sin_term = np.sin(0.25 * np.pi * f)
        numerator = (8 * cos_term**2 * 
                    (1 + 2*v**2 + np.cos(0.5 * np.pi * f)) * 
                    sin_term**4)
        denominator = f**6

        return numerator / denominator

    def OffsetFULL(v):
        """
        Parameters
        ----------
        v : float
            dimensionless Doppler detuning (v = vk/epsilon)

        Returns
        -------
        float
            momentum-dependent offset of full interference signal
        """
        f = np.sqrt(1+v**2)
        return -((v**2 + np.cos(np.pi * f / 2))**2 *
                (v**2 + np.cos(np.pi * f))) / (f**6)

    def OffsetCUT(v):
        """
        Parameters
        ----------
        v : float
            dimensionless Doppler detuning (v = vk/epsilon)

        Returns
        -------
        float
            momentum-dependent offset of cropped interference signal
        """
        f = np.sqrt(1 + v**2)
        return ((v**2 + np.cos(np.pi * f / 2))**2 * np.sin(np.pi * f / 2)**2) / (f**6)  

    def phase_uncertainty(phi, fraction_measured_particles, weighted_Vis, Offset):
        """phase uncertainty for MZI measuring pseudo-angular momentum 
        J_3 = (N_1 - N_0)/2

        Parameters
        ----------
        phi : ndarray
            interferometer phase 
        fraction_measured_particles : float
            either 1 for full signal or \int dp |r1(p)|^2 for cropped signal
        weighted_Vis : float
            visibility weighted with momentum distribution
        Offset : float
            offset weighted with momentum distribution

        Returns
        -------
        ndarray
            phase uncertainty as a function of phase
        """
        N0 =1
        return np.sqrt(
            1 / N0 * (
                (fraction_measured_particles - Offset**2) / (weighted_Vis * np.sin(phi))**2
                + 2 * Offset / weighted_Vis * np.cos(phi) / np.sin(phi)**2
                - (np.cos(phi) / np.sin(phi))**2
            )
        )

    # weighted quantities
    v = vk_val / epsilon
    weighted_Vis = np.trapezoid(Vis(v) 
                                * gaussian(vk_val, sigma_vk), x=vk_val)
    weighted_OffsetFULL = np.trapezoid(OffsetFULL(v) 
                                    * gaussian(vk_val, sigma_vk), x=vk_val)
    weighted_OffsetCUT = np.trapezoid(OffsetCUT(v) 
                                    * gaussian(vk_val, sigma_vk), x=vk_val)
    fraction_measured_particles_full = 1
    fraction_measured_particles_cropped = np.trapezoid(np.abs(r(v, np.pi))**2 
                                                    * gaussian(vk_val, sigma_vk), x=vk_val)

    ############ plotting ############

    ### colormaps ###
    orig_map=plt.cm.get_cmap('gist_heat')
    reversed_cmap = orig_map.reversed()

    def alpha_fct(x,k=40):
        '''function that emphasizes values closer to 1''' 
        return 1.5*.5 / (.5 + np.exp(-k * x)) - .5

    # lowering alpha in colormap 
    colors = reversed_cmap(np.arange(reversed_cmap.N)) 
    # Apply transparency: lower z -> more transparent
    alpha = alpha_fct(np.linspace(0,1,reversed_cmap.N))
    colors[..., -1] = alpha
    print(np.shape(colors))  # set the alpha channel
    aplha_cmap = ListedColormap(colors)


    fig = plt.figure(figsize=(3.2, 2))
    fig, axs = plt.subplots(2, 2, figsize=(3.2, 1.8))

    gs = fig.add_gridspec(2, 2, width_ratios=[1,1], wspace=0, hspace=0)

    # left top: |\psi_1|^2
    ax0 = axs[0, 1]
    # left bottom: |\psi_0|^2
    ax1 = axs[1, 1]

    # shading cropped out regions
    ax0.plot(phi2_val, np.ones_like(phi2_val)*3*dz/2, color='gray', ls='--')
    ax0.fill_between(phi2_val, y1=dz/2, y2=3*dz/2, color='lightgray')
    ax0.plot(phi2_val, np.ones_like(phi2_val)*dz/2, color='gray', ls='--')

    ax1.plot(phi2_val, np.ones_like(phi2_val)*3*dz/2, color='gray', ls='--')
    ax1.fill_between(phi2_val, y1=dz/2, y2=3*dz/2, color='lightgray')
    ax1.plot(phi2_val, np.ones_like(phi2_val)*dz/2, color='gray', ls='--')

    cf_top = ax0.contourf(X, Y, Ze, levels=200, cmap=aplha_cmap, vmin=0, vmax=4)

    # avoide residual lines in pdf export 
    cf_top.set_rasterized(True)

    ax0.set_ylabel(r'$z/\delta z$')
    ax0.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax0.set_xticklabels([])
    ax0.set_yticks([0, dz, 2*dz], labels=['0','1','2'])
    ax0.grid(ls='--',alpha=.5)
    ax0.text(1.3*np.pi, 1.8*dz, r'$|\psi_1|^2$')
    ax0.yaxis.tick_right()
    ax0.yaxis.set_label_position("right")
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position("right")

    cf_bottom = ax1.contourf(X, Y, Zg, levels=200, cmap=aplha_cmap, vmin=0, vmax=4)
    cf_bottom.set_rasterized(True)

    ax1.set_ylabel(r'$z/\delta z$')
    ax1.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], 
                labels=['', r'', r'$\pi$', r'', r'$2\pi$'])
    ax1.set_yticks([0, dz, 2*dz], labels=['0','1','2'])
    ax1.grid(ls='--',alpha=.5)
    ax1.text(1.3*np.pi, 1.8*dz, r'$|\psi_0|^2$')


    # interference signals
    ax2 = axs[0, 0]
    ax2.plot(phi2_val, (signal_result[1] - signal_result[0])/2, 
            color=dark_red, label=r'full')
    ax2.plot(phi2_val, (signal_result[3] - signal_result[2])/2, 
            color=red, ls='--', label=r'cropped')
    ax2.grid(ls='--')
    ax2.set_ylabel(r'$\langle\hat{J}_{3,\rho}\rangle$')
    ax2.set_ylim(-.4,.4)
    ax2.set_yticks([-.4,-.2,0,.2,.4], labels=['','','0.0','','0.4'])
    ax2.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], labels=[], )
    ax2.set_xlim(0,2*np.pi)
    ax2.legend(loc= 'upper center', borderpad=.25, ncol=2, 
            bbox_to_anchor=(.498, 1.395), handlelength=1.3, 
            columnspacing=.4, handletextpad=.4)
    ax2.text(-1.76,-.38,'-0.4')
    ax2.text(-1.5,-.52,'2.5')
    ax2.text(-.49,-.4353,r'\}')

    # phase sensitivities
    ax3 = axs[1, 0]
    ax3.set_xlabel(r'$\phi$', x=1)
    ax3.set_ylabel(r'$\sqrt{N}\Delta\phi_\rho$')
    ax3.plot(phi2_val, phase_uncertainty(phi2_val, 
                                        fraction_measured_particles_full, 
                                        weighted_Vis, weighted_OffsetFULL), 
                                        color=dark_red, 
                                        label=r'$\Delta\phi_\mathrm{H}$')
    ax3.plot(phi2_val, phase_uncertainty(phi2_val, 
                                        fraction_measured_particles_cropped, 
                                        weighted_Vis, weighted_OffsetCUT), 
                                        color=red, ls='--', 
                                        label=r'$\Delta\phi_\rho$')
    ax3.grid(ls='--')
    ax3.set_ylim(1,2.5)
    ax3.set_yticks([1,1.5,2,2.5], labels=['1.0','1.5','2.0',''])
    ax3.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], 
                labels=['0', r'', r'$\pi$', r'', r'$2\pi$'])
    ax3.set_xlim(0,2*np.pi)

    plt.subplots_adjust(wspace=0.03, hspace=0)

    bbox = axs[0, 1].get_position()   

    cax = fig.add_axes((
        bbox.x0,          # left
        bbox.y1 + 0.01,   # bottom (just above)
        bbox.width,       # same width as subplot
        0.3             # height
    ))
    cax.axis('off')

    cb = fig.colorbar(cf_top, ax=cax, orientation='horizontal',  
                    label='probability density', pad=.1, 
                    ticks=[0,1,2,3, max([np.max(Ze), np.max(Zg)])])
    cb.ax.set_xticklabels(['0', '1', '2', '3', 
                        f'{max([np.max(Ze), np.max(Zg)]):.1f}'])
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')

    plt.savefig(im_path + '/full_vs_cropped_signal.pdf', bbox_inches='tight')
    plt.show()

#### plot quasi probability distribution on angular momenutm sphere
def quasi_prob_contrib_k(k, j, m_values, phi, theta, psi_angular_coef):
    """
    Parameters
    ----------
    k : int
        'angular momentum' index
    j : int
        'angular momentum' index
    m_values : ndarray
        values of magnetic quantum number m for which the contribution to the 
        quasi probability distribution does not vanish
    phi : ndarray
        azimuthal angle
    theta : ndarray
        polar angle
    psi_angular_coef : ndarray
        expansion coefficients of atomic states wrt to Dicke states 

    Returns
    -------
    ndarray
        (k,j) contribution of the atomic state to its spherical Wigner 
        distribution as a function of (theta,phi)
    """
    quasi_prob_distr_k = np.zeros((len(phi), len(theta)), dtype=np.complex128)
    sqrt_factor = (-1)**j * np.sqrt(2*k + 1)

    for q in range(-k, k + 1): 
        adj_spherical_tensor_moment_k_q = np.sum(
            sqrt_factor 
            * np.array([
                (-1)**float(-m) * np.conj(
                    float(wigner_3j(int(j), int(k), int(j), -int(m), int(q), int(mm))) 
                    if (
                        ((q-m+mm == 0 and not((m==0 and mm==0 and q==0))) 
                        or ((m==0 and mm==0 and q==0) and (2*j + k)%2==0))
                        ) 
                        else 0
                    ) 
                * np.conj(psi_angular_coef[mm + j]) 
                * psi_angular_coef[m + j]
                for m in m_values for mm in m_values
                ])
        )

        quasi_prob_distr_k += sph_harm_y(k, q, theta[:, None], phi[None, :]) \
                                * adj_spherical_tensor_moment_k_q

    return quasi_prob_distr_k

def plot_ang_mom_sphere_QP_distr(N, psi_angular_coef_fct, 
                                        pt_points, fct_key, im_name, 
                                        alpha=0, beta=0):
        """plots the quasi-probability distribution of an arbitrary atomic 
        state on the angular-momentum sphere (cf. main text Fig. 4)

        Parameters
        ----------
        N : int
            atom number, needs to be even!
        psi_angular_coef_fct : callable
            function returning expansion coefficients of atomic state expanded 
            w.r.t. Dicke states
        pt_points : ndarray
            number of sample points for angular grid
        fct_key : str
            specifies kind of quasi-probability distribution, e.g.,
            Husimi-Q func by setting 'HusimiQ', by default 'Wigner'
        im_name : str
            name of the produced image
        alpha : int, optional
            x-rotation of the distribution, by default 0
        beta : int, optional
            y-rotation of the distribution, by default 0
        """

        def quasi_prob_function_for_arb_spin_j_state(N: int, psi_angular_coef_fct, 
                                                    theta, phi, fct_key='Wigner'):
            """
            Parameters
            ----------
            N : int
                atom number, needs to be even!
            psi_angular_coef_fct : callable
                function returning expansion coefficients of atomic state expanded 
                w.r.t. Dicke states
            theta : ndarray
                polar angle
            phi : ndarray
                azimuthal angle
            fct_key : str, optional
                specifies kind of quasi-probability distribution, e.g., Husimi-Q func 
                by setting 'HusimiQ', by default 'Wigner'

            Returns
            -------
            ndarray
                quasi-probability distribution for given atomic state as a function of 
                (theta,phi)
            """

            if N%2 != 0:
                print("N must be even!")
                return None
            
            j = N//2
            psi_angular_coef = np.array([psi_angular_coef_fct(m) for m in range(-j, j+1)])
            wigner_distr = np.zeros((len(phi), len(theta)), dtype=np.complex128)
            
            non_zero_psi_coef_index = np.nonzero(psi_angular_coef)[0]
            m_values = non_zero_psi_coef_index - j

            k_val = np.arange(N + 1)
            with mp.Pool() as pool:
                inputs = zip(k_val, (j,) * len(k_val), 
                            (m_values,) * len(k_val), 
                            (phi,) * len(k_val), 
                            (theta,) * len(k_val), 
                            (psi_angular_coef,) * len(k_val)) # similar to [(eps, sigma_v_wk) for eps in epsilon_val]
                wigner_k_summands = pool.starmap(quasi_prob_contrib_k, inputs)
                pool.close()
                pool.join()
            
            if fct_key == 'HusimiQ':
                f_ks = np.array([np.sqrt(math.factorial(N) * math.factorial(N + 1) 
                                        / (math.factorial(N - k) 
                                            * math.factorial(N + k + 1)))
                                for k in range(N + 1)])[:,None,None]
            
            elif fct_key== 'Wigner':
                f_ks = 1

            wigner_distr = np.sum(wigner_k_summands * f_ks, axis=0)

            return wigner_distr


        ### plotting ###

        theta = np.linspace(0, np.pi, pt_points)
        phi = np.linspace(0, 2 * np.pi, pt_points)
        phi2D, theta2D = np.meshgrid(phi,theta)
        x = np.sin(theta2D)*np.cos(phi2D)
        y = np.sin(theta2D)*np.sin(phi2D)
        z = np.cos(theta2D)

        # arbitrary rotation of distribution, by rotation  coordinates
        coords = np.stack([x, y, z])        
        Rx = np.array([
            [1, 0,             0],
            [0, np.cos(alpha), -np.sin(alpha)],
            [0, np.sin(alpha),  np.cos(alpha)]
        ])

        Ry = np.array([
            [np.cos(beta),  0, np.sin(beta)],
            [0,              1, 0],
            [-np.sin(beta), 0, np.cos(beta)]
        ])
        
        # Combined rotation matrix (applied in order: Z, then Y, then X)
        R = Ry @ Rx

        # Using einsum for clean matrix multiplication across the grid
        # 'ij,jkl->ikl' means: multiply matrix R (ij) with coords (jkl)
        coords_rot = np.einsum('ij,jkl->ikl', R, coords)

        x_rot, y_rot, z_rot = coords_rot[0], coords_rot[1], coords_rot[2]

        # quasi probability data
        quasi_prob_distr \
            = quasi_prob_function_for_arb_spin_j_state(N, 
                                                    psi_angular_coef_fct, 
                                                    theta, phi, fct_key)
        data = np.real(quasi_prob_distr)

        # Create surface plot
        tick_vals = np.linspace(data.min(), data.max(), 5)  # 5 ticks
        tick_text = [f"{val:.1f}" for val in tick_vals]

        surface = go.Surface(
            x=x_rot,
            y=y_rot,
            z=z_rot,
            surfacecolor=data,
            colorscale='Viridis',
            showscale=False,
            opacity=1.0,
            colorbar=dict(
                tickvals=tick_vals.tolist(),
                ticktext=tick_text,
                tickfont=dict(size=40, color='black'),
                xpad=0,  # horizontal padding (in px)
                ypad=0,   # vertical padding
                x=.87,  # moves colorbar right (1.0 = plot edge)
                y=0.43,   # center vertically
                len=.7,  # shorter colorbar
                thickness=20
            )
        )

        # Add coordinate grid lines (meridians and parallels)
        lines = []

        # longitudes
        for ph in np.linspace(0, 2 * np.pi, 12, endpoint=False):
            x_lg = np.cos(ph) * np.sin(theta)
            y_lg = np.sin(ph) * np.sin(theta)
            z_lg = np.cos(theta)
            lines.append(go.Scatter3d(x=x_lg, y=y_lg, z=z_lg, mode='lines', 
                                    line=dict(color='gray', width=3), 
                                    showlegend=False))

        # lattitudes
        for th in [.33*np.pi/2, .66*np.pi/2, np.pi/2, 
                1.33*np.pi/2, 1.66*np.pi/2]:
            x_bg = np.cos(phi) * np.sin(th)
            y_bg = np.sin(phi) * np.sin(th)
            z_bg = np.full_like(phi, np.cos(th))
            lines.append(go.Scatter3d(x=x_bg, y=y_bg, z=z_bg, mode='lines', 
                                    line=dict(color='gray' if th != np.pi/2 else 'gray', 
                                                width=3 if th != np.pi/2 else 5), 
                                                showlegend=False))

        # create axes 
        length=1.21
        directions = {
            'x': np.array([length, 0, 0]),
            'y': np.array([0, length, 0]),
            'z': np.array([0, 0, length]),
        }

        axes = []
        heads = []
        labels = []
        for label, vec in directions.items():
            axes.append(go.Scatter3d(
                x=[0,vec[0]], 
                y=[0,vec[1]],
                z=[0,vec[2]],
                mode='lines',
                line=dict(color='gray', width=12),
                showlegend=False
            ))

            # create arrows
            heads.append(go.Cone(
                x=[vec[0]],
                y=[vec[1]],
                z=[vec[2]],
                u=[vec[0]],
                v=[vec[1]],
                w=[vec[2]],
                sizemode='absolute',
                sizeref=0.15,
                anchor='tail',
                colorscale=[[0, 'gray'], [1, 'gray']],
                showscale=False
            ))

        # camera parameters
        theta_eye = 1.94*np.pi/4 
        phi_eye = np.pi/20 
        r_eye = 1.3
        x_eye = r_eye * np.sin(theta_eye)*np.cos(phi_eye)
        y_eye = r_eye * np.sin(theta_eye)*np.sin(phi_eye)
        z_eye = r_eye * np.cos(theta_eye)

        # Create layout
        layout = go.Layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode='cube',
                camera=dict(eye=dict(x=x_eye, y=y_eye, z=z_eye))
            ),
            margin=dict(l=0, r=0, t=0, b=0)
        )

        # Plot all
        fig = go.Figure(
            data=[surface] + lines + axes + heads + labels,
            layout=layout
        )
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', 
                        plot_bgcolor='rgba(0,0,0,0)', 
                        margin=dict(l=0, r=0, t=0, b=0))

        fig.write_image(im_path + "/" + im_name + ".svg", width=850, height=600)

#### plot analytics ####

def plotPhaseUncertainty(sigma_values, N):
    """plots the `phaseUncertainty()` of the MZI driven by first-order Bragg 
    diffraction and operated with the OAT state polarized in S_x direction and
    rotated ont the equator (cf. main text Fig. 5)

    Parameters
    ----------
    sigma_values : list
        values for the standard deviation of the atoms' initial momentum 
        distribution in units of the dimensionless doppler detuning
        (vk = \nu_k / \omega_k) the phase uncertainty is plotted for
    N : int
        atom number
    """
    global NO_ADJ_CLASSES, wed_A0_0, wed_A10_0, wed_R0_0, wed_A10Deriv_0
    mu_optimal = mu_opt(N)
    alpha = np.linspace(.2,1,len(sigma_values))[::-1]

    fig, ax = plt.subplots(figsize=(3.2, 1.65))

    for i, sig in enumerate(sigma_values):
        print(i)
        weighting(epsilon_val, sig)
        plt.plot(epsilon_val, 
                 N * phaseUncertainty(N, mu_optimal, 0, alignWithXYPlane=True)**2, 
                 label=r"$\sigma_q = {:,g}\% \hbar k$".format(sig/2*100), 
                 color=blue, alpha=alpha[i])
        if i == 0: 
            NO_ADJ_CLASSES=True
            def integrand(vk):
                global wed_A0_0, wed_A10_0, wed_R0_0, wed_A10Deriv_0
                def contributions(vk, epsilon_val, NO_ADJ_CLASSES):
                    funcs = [A0_0, A10_0, R0_0, A10Deriv_0]
                    return np.array([f(vk, epsilon_val, NO_ADJ_CLASSES) for f in funcs])
                return contributions(vk, epsilon_val, NO_ADJ_CLASSES) * gaussian(vk, sig)

            # using np.trapz
            n_trapz = 2**7
            v_wk_val = np.linspace(-5*sig, 5*sig, n_trapz)
            y = np.array([integrand(vk) for vk in v_wk_val])  # Shape: (n_trapz, ...)
        
            wed_A0_0, wed_A10_0, wed_R0_0, wed_A10Deriv_0 = np.trapezoid(y, x=v_wk_val, axis=0)

            plt.plot(epsilon_val, 
                     N * phaseUncertainty(N, mu_optimal, 0, 
                                          alignWithXYPlane=True, 
                                          NO_ADJ_CLASSES=True)**2,
                    label=r"$\mathrm{only\ velocity\ selectivity\quad \ \ \ .}$", 
                    color=orange, ls='--')
            NO_ADJ_CLASSES=False

    ax.plot(epsilon_val, N * phaseUncertaintySQ_noVS(N, mu_optimal, 0, True), 
            color=green, ls='--', label=r"$\sigma_q = 0\hbar k$")
    ax.axhline(y=1, color=red, lw=1)
    ax.text(x=.45, y=1.2, s=r'SNL', color=red)
    ax.set_yscale('log')
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel(r"$N\Delta\phi^2$")
    ax.set_ylim(bottom=9.3e-4,top=50)
    ax.set_yticks([1e-3,1e-2,1e-1,1,1e1])
    ax.set_xticks([0,.1,.2,.3,.4,.5])

    # order labels in other order
    handles, labels = plt.gca().get_legend_handles_labels()
    # specify order
    print(labels)
    order = [3, 4, 2, 1,]

    # pass handle & labels lists along with order as below
    leg = ax.legend([handles[i] for i in order], 
                [labels[i] for i in order],
                fontsize=9, ncols=2, 
                loc='upper right', 
                bbox_to_anchor=(1.026, 1.41),
                # bbox_to_anchor=(0,0,.5,.2),
                handlelength=1.4, 
                columnspacing=.5,
                handletextpad=.5,
                borderpad=.5)
    
    extra_leg = ax.legend(
        handles=[handles[0]],
        loc='upper right',
        bbox_to_anchor=(1.03, 1.41),
        frameon=False,
        handlelength=1.4, 
        columnspacing=.4,
        handletextpad=.5,
        borderpad=.5)

    extra_leg._legend_box.align = "left"
    ax.add_artist(leg)

    ax.grid(ls='--',alpha=.5)

    plt.savefig(im_path + "/PS_Bragg_VS_and_parasitic_diff.pdf", bbox_inches='tight')
    # plt.show()

#### plot analytical vs numerical solution ####

def analytics_vs_numerics(sigma_vk, N):
    """plots the analytical `phaseUncertainty()` and the numerical expression
    `phaseUncertaintySQ_num()` as well as the residual between them 
    (cf. main text Fig. 6)

    Parameters
    ----------
    sigma_vk : float
        standard deviation of the momentum distribution in units of 
        dimensionless doppler detuning (vk = \nu_k / \omega_k)
    N : int
        atom number
    """
    mu_optimal = mu_opt(N)

    fig, (ax0, ax1) = plt.subplots(
        2, 1,
        figsize=(3.2, 2.5),
        sharex=True,
        gridspec_kw={'height_ratios': [1, 1], 'hspace': 0}
    )

    # --- Analytical vs numerical (top plot)
    weighting(epsilon_val, sigma_vk)
    ana_data = phaseUncertainty(N, mu_optimal, 0, True)**2
    ax0.plot(epsilon_val, N * ana_data, label=r"Dyson", color=blue)

    weighting_num(epsilon_val, sigma_vk, True, False)
    num_data = phaseUncertaintySQ_num(N, mu_optimal, 0, True)
    ax0.plot(epsilon_val, N * num_data, label=r"Numerics", color=orange, ls='--')

    ax0.set_yscale('log')
    ax0.set_ylabel(r"$N\Delta\phi^2$")
    ax0.legend(handlelength=1.4, loc='upper left')
    ax0.grid(ls='--', alpha=.5)

    # --- Error plot (bottom)
    ax1.plot(epsilon_val, np.abs(num_data - ana_data),
             color=green, label=r'$R_{\Delta\phi}$')
    ax1.plot(epsilon_val, epsilon_val**2,
             color=orange, label=r'$\varepsilon^2$')
    ax1.plot(epsilon_val, epsilon_val**4,
             color=red, label=r'$\varepsilon^4$')

    ax1.set_xticks([0,.1,.2,.3,.4,.5])
    ax1.set_yscale('log')
    ax1.set_xlabel(r"$\varepsilon$")
    ax1.legend(handlelength=1.4, loc='lower right')
    ax1.grid(ls='--', alpha=.5)

    ax0.tick_params(labelbottom=False)

    plt.savefig(im_path + "/PS_Bragg_num_vs_ana.pdf", bbox_inches='tight')
    # plt.show()

#### box vs blackman pulses ####

def blackman_vs_box_pulse(sigma_vk, N):
    """plots numerically calculated phase uncertainty `phaseUncertaintySQ_num()`
    for Blackman as well as box pulses for the OAT state 
    (cf. main text Fig. 7)

    Parameters
    ----------
    sigma_vk : float
        standard deviation of the initial momentum distribution in units of 
        dimensionless doppler detuning (vk = \nu_k / \omega_k)
    N : int
        atom number
    """
    mu_optimal = mu_opt(N)
    
    fig = plt.figure(figsize=(3.2, 1.65))

    weighting_num(epsilon_val, sigma_vk, True, False)
    box_data = phaseUncertaintySQ_num(N, mu_optimal, 0, True)
    plt.plot(epsilon_val, N * box_data, label=r"Box pulse", color=blue)


    weighting_num(epsilon_val, sigma_vk, True, True)
    blackman_data = phaseUncertaintySQ_num(N, mu_optimal, 0, True)
    plt.plot(epsilon_val, N * blackman_data, label=r"Blackman pulse", 
             color=orange, ls='--')
    
    plt.axhline(y=1, color=red, lw=1)
    plt.text(x=.13, y=.55, s=r'SNL', color=red)

    plt.xlabel(r"$\varepsilon$")
    plt.ylabel(r"$N\Delta\phi^2$")
    plt.xticks([0,.1,.2,.3,.4,.5])
    plt.yscale('log')
    plt.grid(ls='--', alpha=.5)
    plt.legend(loc='upper center', 
               ncol=2,
               handlelength=1.4,
               columnspacing=.6,
               handletextpad=.6)
    plt.savefig(im_path + "/PS_Bragg_box_vs_Blackman.pdf", bbox_inches='tight')
    # plt.show()

#### PS for CSS on equator ####

def fock_state_after_BS(sigma_values, N):
    """plots phase uncertainty `phaseUncertainty()` for the CSS polarized along
    the S_x direction as well as for the OAT state rotated onto the equator
    (cf. main text Fig. 8)
    
    sigma_values : list
        values for the standard deviation of the atoms' initial momentum 
        distribution in units of the dimensionless doppler detuning
        (vk = \nu_k / \omega_k) the phase uncertainty is plotted for
    N : int
        atom number
    """
    alpha = np.linspace(.3,1,len(sigma_values))

    mu_optimal = mu_opt(N)

    handles = []

    fig = plt.figure(figsize=(3.2, 1.65))
    for i, sig in enumerate(sigma_values[::-1]):
        weighting(epsilon_val, sig)
        line1, = plt.plot(epsilon_val, N * phaseUncertainty(N, 0, 0, True)**2, 
                          label=r"$\sigma_q = {:,g}\% \hbar k$".format(sig/2*100), 
                          color=blue, alpha = alpha[i])
        line2, = plt.plot(epsilon_val, 
                          N * phaseUncertainty(N, mu_optimal, 0, True)**2, 
                          label=None, color=blue, alpha = alpha[i], 
                          linestyle='--')

        handles.append((line1, line2))

    labels = [r"$\sigma_q = {:,g}\% \hbar k$".format(sig/2*100) 
              for sig in sigma_values[::-1]]

    plt.yscale('log')
    plt.xticks([0,.1,.2,.3,.4,.5])
    plt.xlabel(r"$\varepsilon$")
    plt.ylabel(r"$N\Delta\phi^2$")
    plt.ylim(top=10, bottom=0.6)
    plt.axhline(y=1, color=red, lw=1)
    plt.text(x=.45, y=1.2, s=r'SNL', color=red)
    plt.grid(ls='--', alpha=.5, which='both')

    class HandlerLinesVertical(HandlerTuple):
        def create_artists(self, legend, orig_handle,
                    xdescent, ydescent, width, height, fontsize,
                    trans):
            ndivide = len(orig_handle)
            
            a_list = []
            for i, handle in enumerate(orig_handle):
                y = (height / float(ndivide)) * i - ydescent - height/2.
                line = plt.Line2D(np.array([0,1])*width, [-y,-y])
                line.update_from(handle)
                # line.set_marker(None)
                point = plt.Line2D(np.array([.5])*width, [-y])
                point.update_from(handle)
                for artist in [line, point]:
                    artist.set_transform(trans)
                a_list.extend([line,point])
            return a_list
    
    plt.legend(handles, labels, fontsize=9, loc='upper right', 
               bbox_to_anchor=(1.01,1.02), handlelength=1.4, 
               handler_map={tuple: HandlerLinesVertical(ndivide=1)})
    

    plt.savefig(im_path + "/PS_Fock_state.pdf", bbox_inches='tight')
    # plt.show()

#### orientation and squeezing optimisation ####

def plot_mu_awa_phi_opt_phaseUncertainty(sigma_vk, N, alpha_min, alpha_max, 
                                  mu_min, mu_max, n_steps):
    """plotting the phase uncertainty `phaseUncertainty()` separately minimized 
    w.r.t. the OAT state's inclination alpha and the twisting strength mu = 2 chi

    Parameters
    ----------
    sigma_vk : float
        standard deviation of the momentum distribution in units of 
        dimensionless doppler detuning (vk = \nu_k / \omega_k)
    N : int
        atom number
    alpha_min : float
        lower bound for the angle alpha for the optimization
    alpha_max : float
        upper bound for the angle alpha 
    mu_min : float
        lower bound for the twisting strength
    mu_max : float
        upper bound for the twisting strength
    n_steps : int
        steps iterated over between lower and upper bound for the alpha- and 
        mu-optimization
    """
    mu_optimal = mu_opt(N)
    
    weighting(epsilon_val, sigma_vk)
    opt_inclination_values, phi_opt_PS_SQ = orientation_opt_PS_SQ(N, mu_optimal, 
                                                                  alpha_min,
                                                                  alpha_max, 
                                                                  n_steps)
    opt_twisting_values, mu_opt_PS_SQ = squeezing_opt_PS_SQ(N, mu_min, mu_max, 
                                                            n_steps)

    fig, (ax0, ax1, ax2) = plt.subplots(
        3, 1,
        figsize=(3.2, 3.2),
        sharex=True,
        gridspec_kw={'height_ratios': [1, 1, 1], 'hspace': 0}
    )

    ax0.tick_params(labelbottom=False)
    ax0.plot(epsilon_val, 
             N * phaseUncertainty(N, mu_optimal, 0, alignWithXYPlane=True)**2, 
             color=blue, ls=(0, (1, 1)), label=r"$\varphi=0$")
    ax0.plot(epsilon_val, N * phi_opt_PS_SQ, color=green, label=r"optimized")
    ax0.plot(epsilon_val, N * mu_opt_PS_SQ, color=orange, ls='--')
    ax0.axhline(y=1, color=red, lw=1)
    ax0.text(x=.45, y=1.2, s=r'SNL', color=red)
    ax0.set_yscale('log')
    ax0.set_ylabel(r"$N\Delta\phi^2$")
    ax0.set_ylim(bottom=4.5e-3,top=20)
    ax0.set_yticks([1e-2,1e-1,1,10], labels=[r"$10^{-2}$","", r"$10^{0}$",""])
    ax0.grid(ls='--', alpha=.5)

    ax1.tick_params(labelbottom=False)
    ax1.plot(epsilon_val[1:], np.zeros_like(epsilon_val[1:]), 
             color = blue, ls=(0, (1, 1)), label=r"$\varphi=0$")
    ax1.plot(epsilon_val[1:], opt_inclination_values[1:], 
             color = green, label=r"optimized")
    ax1.set_ylabel(r"$\varphi$")
    ax1.set_ylim(-.01, np.max(opt_inclination_values[1:]+.015))
    ax1.grid(ls='--', alpha=.5)
    ax1.legend(handlelength=1.4, bbox_to_anchor=(0.49, 1.0), 
               columnspacing=.5, handletextpad=.5)

    ax2.plot(epsilon_val, np.ones_like(epsilon_val) * mu_optimal/2*1000,
                      color=blue, ls=(0, (1, 1)), 
                      label=fr'$\chi_0 = {mu_optimal/2:.4f}$')
    ax2.plot(epsilon_val, opt_twisting_values*1000, 
                     color=orange, ls='--', label=r'optimized')
    ax2.grid(ls='--', alpha=.5)
    ax2.set_ylim(top=mu_optimal/2*1e3 + .13)
    ax2.set_ylabel(r"$\chi\times10^{{3}}$")
    ax2.set_xticks([0,.1,.2,.3,.4,.5])
    ax2.set_xlabel(r"$\varepsilon$")
    ax2.legend(handlelength=1.4, loc='lower center', 
               bbox_to_anchor=(0.3, 0), columnspacing=.5, handletextpad=.5)

    plt.savefig(im_path + "/PS_Bragg_mu_phi_optimised.pdf", bbox_inches='tight')
    # plt.show()

#### plot analytics vs numerics: couplings first-order Bragg ####
def plot_pert_vs_numerics_couplings(epsilon, vk, tau_final, g0):
    """plots time evolution of the momentum classes population ptobability
    for a first-order Bragg pulse up to pulse area tau_final and displays the 
    residuals between numerical and perturbative solutions in complex argument 
    and modulus  

    Parameters
    ----------
    epsilon : float
        dimensionless Rabi frequency (\Omega_0 in units of recoil frequency 
        \omega_k)
    vk : float
        dimensionless Doppler detuning (\nu_k in units of recoil frequency 
        \omega_k))
    tau_final : float
        pulse area up to which plotted
    g0 : ndarray
        initial condition shape (6,) 

    Returns
    -------
    _type_
        _description_
    """
    v = vk / epsilon
    ### Analytic solution w/o secular terms ###
    def U2x2DysonNosecTerms(epsilon, v, tau_p, phi):
        """secular-term-free solution of resonant first-order Bragg diffraction on 
        the sub system of the relevant momentum classes n=0,1 

        Parameters
        ----------
        epsilon : float
            dimensionless Rabi frequency (\Omega_0 in units of recoil frequency 
            \omega_k)
        v : float
            dimensionless Doppler detuning in units of Rabi frequency \Omega_0
        tau_p : float
            pulse area time * \Omega_0
        phi : float
            phase of the laser pulse (argument of complex Rabi frequency)

        Returns
        -------
        ndarray
            matrix describing time evolution of first-order Bragg 
            diffraction on the sub system of the relevant classes n=0,1
        """
        U2x2_int_pic = np.array(
        [
            [1 + epsilon**2 * gammaII(epsilon, v, tau_p), 
            epsilon**2 * np.exp(-1j * phi) * gammaIJ(epsilon, -v, tau_p)],
            [epsilon**2 * np.exp(1j * phi) * gammaIJ(epsilon, v, tau_p),
            1 + epsilon**2 * gammaII(epsilon, -v, tau_p)]
        ], dtype=complex)

        UBack = np.array(
        [
            [tBack(v, tau_p, epsilon), 
            rBack(v, tau_p, epsilon)],
            [-np.conj(rBack(v, tau_p, epsilon)), 
            np.conj(tBack(v, tau_p, epsilon))]
        ], dtype=complex)

        return np.exp(1j * epsilon * tau_p / 8) * UBack @ U2x2_int_pic

    def UDysonNosecTerms(epsilon, v, t, phi=0):
        """secular-term-free solution of the main momentum classes n=0,1 in 
        resonant first-order Bragg diffraction, also accounting for the couplings to 
        the adjacent classes n=-1,2

        Parameters
        ----------
        epsilon : float
            dimensionless Rabi frequency (\Omega_0 in units of recoil frequency 
            \omega_k)
        v : float
            dimensionless Doppler detuning in units of Rabi frequency \Omega_0
        t : float
            time
        phi : float
            phase of the laser pulse (argument of complex Rabi frequency)

        Returns
        -------
        ndarray
            matrix describing the time evolution of the main momentum classes n=0,1 
            including the adjacent momentum classes n=-1,2 during resonant 
            first-order Bragg diffraction
        """
        f = np.sqrt(1 + v**2)
        
        # U12
        U01 = (
            np.exp(-0.5 * 1j * ((f - 3*v) * epsilon * t + 2 * phi)) * epsilon * (
                -np.exp(0.25 * 1j * (8 + epsilon**2) * t) * (f + v)**2 * (4 + f*epsilon - 3*v*epsilon)
                + np.exp(0.25 * 1j * (8 + 4*f*epsilon + epsilon**2) * t) * (-4 + f*epsilon + 3*v*epsilon)
                + np.exp(0.5 * 1j * (f - 3*v) * epsilon * t) * f * (8*(f + v) + (-1 + (f - 5*v)*(f + v))*epsilon)
            )
        ) / (32 * f * (f + v)) * (np.exp(1j / 8 * epsilon**2 * t) 
                                * np.exp(1j / 16 * epsilon**2 * (-4 + v*epsilon) * t) 
                                * np.exp(- 1j / 2 * (4 + 3*v*epsilon) * t))

        # U13
        U02 = -(
            np.exp(-0.5 * 1j * ((f - 3*v)*epsilon*t + 4*phi)) * epsilon * (
                -2 * np.exp(0.5 * 1j * (f - 3*v) * epsilon * t) * f * epsilon
                + np.exp(0.25 * 1j * (8 + epsilon**2) * t) * (4 + f*epsilon - 3*v*epsilon)
                + np.exp(0.25 * 1j * (8 + 4*f*epsilon + epsilon**2) * t) * (-4 + f*epsilon + 3*v*epsilon)
            )
        ) / (32 * f) * (np.exp(1j / 8 * epsilon**2 * t) 
                        * np.exp(1j / 16 * epsilon**2 * (-4 + v*epsilon) * t) 
                        * np.exp(- 1j / 2 * (4 + 3*v*epsilon) * t))

        # U42
        U31 = -(
            np.exp(-0.5 * 1j * (f + 3*v) * epsilon * t + 2 * 1j * phi) * epsilon * (
                -2 * np.exp(0.5 * 1j * (f + 3*v) * epsilon * t) * f * epsilon
                + np.exp(1j * (2 + f*epsilon) * t) * (-4 + f*epsilon - 3*v*epsilon)
                + np.exp(0.25 * 1j * (8 + epsilon**2) * t) * (4 + f*epsilon + 3*v*epsilon)
            )
        ) / (32 * f) * (np.exp(1j / 8 * epsilon**2 * t) 
                        * np.exp(- 1j / 16 * epsilon**2 * (4 + v*epsilon) * t) 
                        * np.exp(1j / 2 * (-4 + 3*v*epsilon) * t))

        # U43
        U32 = (
            np.exp(1j * phi) * epsilon * (
                4
                + np.exp(0.5 * 1j * (4 + f*epsilon - 3*v*epsilon) * t) * (f + v)**2 * (-4 + f*epsilon - 3*v*epsilon)
                - np.exp(-0.25 * 1j * (-8 + (2*f + 6*v - epsilon)*epsilon) * t) * (4 + f*epsilon + 3*v*epsilon)
                + 4 * (f + v) * (f + v + f*v*epsilon)
            )
        ) / (32 * f * (f + v)) * (np.exp(1j / 8 * epsilon**2 * t) 
                                * np.exp(- 1j / 16 * epsilon**2 * (4 + v*epsilon) * t) 
                                * np.exp(1j / 2 * (-4 + 3*v*epsilon) * t))


        result = np.zeros((4,4), dtype=complex)
        
        result[1:3, 1:3] = U2x2DysonNosecTerms(epsilon, v, t * epsilon, phi)
        result[0,1] = U01
        result[0,2] = U02
        result[3,1] = U31
        result[3,2] = U32

        return result

    def solve_ode(vk, epsilon, g0):
        """solves resonant first-order Bragg diffraction up until tau_final

        Parameters
        ----------
        v : float
            dimensionless Doppler detuning in units of Rabi frequency \Omega_0
        epsilon : float
            dimensionless Rabi frequency (\Omega_0 in units of recoil frequency 
            \omega_k)
        phi : float
            interferometer phase
        g0 : ndarray
            initial condition

        Returns
        -------
        (ndarray, ndarray)
            dimensionless time array and corresponding solution for first-order 
            Bragg diffraction
        """

        sol = solve_ivp(dgdt, (0, tau_final/epsilon), g0, 
                        args=(vk, epsilon), method='RK45') 
        # sol = odeint(dgdt, g0, )
        return sol.t, sol.y

    ### parameters ###


    ### calculating numerical solution ### 
    t_val, g_num = solve_ode(vk, epsilon, g0)

    ### calculating the analytical solution w/o secular terms ###
    g_ana = np.array([UDysonNosecTerms(epsilon, v, t) 
                    @ np.array([0,g0[2],g0[3],0], dtype=complex) for t in t_val]).T

    ### plotting ###

    functions = []
    functions_num = []
    for i in range(12):
        if i < 4: 
            functions.append(np.abs(g_ana[3-i])**2)
            functions_num.append(np.abs(g_num[4-i])**2)
        elif i < 8: functions.append(np.abs(np.abs(g_num[4-i%4]) - np.abs(g_ana[3-i%4])))
        else: functions.append(np.abs(np.angle(g_num[4-i%4]) - np.angle(g_ana[3-i%4])))

    # Create figure and axes
    fig, axes = plt.subplots(3, 4, figsize=(7.5, 5), sharex='col', sharey='row', 
                            gridspec_kw={'height_ratios': [1, 1, 1]})
    fig.subplots_adjust(wspace=0, hspace=0)

    # Row and column labels
    row_labels = ['population', 
                r'$|\mathrm{Arg}(c_n) - \mathrm{Arg}(c_{\mathrm{num}})|$',
                r'$|\mathrm{Arg}(c_n) - \mathrm{Arg}(c_{\mathrm{num}})|$']
    col_labels = ['-1', '0', '1', '2']

    # Plotting loop
    for i, ax in enumerate(axes.flat):
        fct = functions[i]
        ax.plot(t_val, fct * (50 if (i == 0 or i == 3) else 1), 
                color=(blue if i<4 else green), 
                lw=(1 if i>=4 else 1.5), 
                label=(r'Dyson'))
        if i < 4: 
            fct_num = functions_num[i]
            ax.plot(t_val, fct_num * (50 if (i == 0 or i == 3) else 1), 
                    color=orange, ls=(0, (3, 2)), 
                    label='Numerics')

        ax.margins(.1,.1)
        ax.grid(ls='--')
        ax.set_xticks(np.array([0, 2*np.pi, 4*np.pi])/epsilon, 
                    labels=[r'0', r'$2\pi$', r'$4\pi$'])
        ax.tick_params(axis='x', direction='in', bottom=True, top=True)
        ax.tick_params(axis='y', direction='in', left=True, right=True)

        if i >= 4:
            ax.axhline(y=epsilon**2, color=red, lw=1)
            ax.axhline(y=epsilon**3, color=orange, lw=1)
            ax.set_yscale('log')
            ax.set_ylim(1e-6, 10*epsilon**2)

        elif i >= 8:
            ax.set_xlabel(r'$\tau_{\mathrm{pulse}}$')
            ax.set_ylim(1e-6, 20*epsilon**2)

    for i, ax in enumerate(axes[0]):
        if i==0: ax.set_title(rf'$n = {i-1}$')
        else: ax.set_title(rf'${i-1}$')

    axes[2,1].set_xlabel(r'$\tau$', x=1)
    axes[0,3].plot([],[], color=green, label=r'Residua')
    axes[0,3].plot([],[], color=red)
    axes[0,3].legend(loc='upper right')
    axes[0,0].set_ylabel(r'$|g_n|^2$')
    axes[1,0].set_ylabel(r'$R_\mathrm{abs}$')
    axes[2,0].set_ylabel(r'$R_\mathrm{arg}$')

    axes[1,0].text(1, epsilon**2+.01, r'$\varepsilon^2$', color=red)
    axes[2,0].text(1, epsilon**2+.01, r'$\varepsilon^2$', color=red)

    axes[1,1].text(1, epsilon**3-.0062, r'$\varepsilon^3$', color=orange)
    axes[2,1].text(1, epsilon**3-.0062, r'$\varepsilon^3$', color=orange)

    axes[0,0].text(.5*np.pi/epsilon,.7, r'$\times 50$')
    axes[0,3].text(.5*np.pi/epsilon,.25, r'$\times 50$')

    axes[0,0].set_yticks([0,.5,1])
    axes[1,0].set_yticks([1e-2,1e-4,1e-6])
    axes[2,0].set_yticks([1e-2,1e-4,1e-6])
    axes[1,0].minorticks_off()
    axes[2,0].minorticks_off()

    plt.savefig(im_path + "/pops_1stO_Bragg.pdf", bbox_inches="tight")
    plt.show()


############ Run ############


if __name__ == '__main__': # this is required for multiprocessing...
    steps=2**8  # 2**8
    eps_min=.005 # .005
    eps_max=.5 #.5
    sigma_vk=.01
    N=2e4
    time_dep_pulse=False # blackman (True), box pulses (False) (former requires ~two times of latter's computation time)
    NO_ADJ_CLASSES = False

    epsilon_val = np.linspace(eps_min, eps_max, steps) # Array of Rabi frequencies: epsilon = \Omega_0 / \omega_k

    start = time.time()

    #### main text Fig. 1
    plot_Rabi_cycle_momentum(epsilon=1.5, sigma_vk=.2,
                             time_dep_pulse=False)


    #### main text Fig. 3
    plot_cropped_vs_full_signal(epsilon = .1, sigma_vk = .1, T_Omega = 1e3)


    # main text Fig. 4 and insets in Fig. 5, 7-9
    def CSS(theta, phi, N, m):
        """coeff. of coherent spin state

        Parameters
        ----------
        theta : float
            polar angle of CSS
        phi : _type_
            azimuthal angle of CSS
        N : int
            atom number
        m : int
            magnetic quantum number

        Returns
        -------
        float
            expansion coefficient corresponding to m of CSS's expansion into 
            Dicke states
        """

        return np.sqrt(binom(N, m + N/2)) * np.sin(theta / 2)**(N/2 + m) \
                * np.cos(theta / 2)**(N/2 - m) * np.exp(-1j * (N/2 + m) * phi)
    
    def OAT_CSS(theta, phi, N, m, mu):
        """coeff. of one-axis-twisted state [Kitagawa et al. Phys. Rev. A 47, 5138 ]

        Parameters
        ----------
        theta : float
            polar angle of CSS
        phi : _type_
            azimuthal angle of CSS
        N : int
            atom number
        m : int
            magnetic quantum number
        mu : float
            twisting strength mu = 2 chi

        Returns
        -------
        float
            expansion coefficient corresponding to m of OAT state's expansion into 
            Dicke states
        """
        return np.exp(-1j * mu/2 * m**2) * CSS(theta, phi, N, m)

    def rot_OAT_TFS(N, m, mu):
        """coeff. of a rotated and one-axis-twisted twin-Fock state

        Parameters
        ----------
        N : int
            atom number
        m : int
            magnetic quantum number
        mu : float
            twisting strength mu = 2 chi

        Returns
        -------
        float
            expansion coefficient corresponding to m of rotated OAT twin-Fock state's 
            expansion into Dicke states
        """

        result = 0.0
        j = N//2

        n_vals = range(0, j + 1)
        if m < 0: n_vals = range(-int(m), j + 1)

        for n in n_vals:
            result += (-1/2)**n / (math.factorial(n) * math.factorial(n + m)) \
                        * math.factorial(j + n) / math.factorial(j - n)

        return  result * np.exp(-1j * mu/2 * m**2) \
                * np.sqrt(math.factorial(j + m) / math.factorial(j - m))

    # examples
    n=30
    plot_ang_mom_sphere_QP_distr(N=n, 
                            psi_angular_coef_fct=lambda m: CSS(np.pi/2,0,n,m),
                            pt_points=100, 
                            fct_key="Wigner", 
                            im_name = 'CSS',
                            beta=np.pi/4)
    # plot_ang_mom_sphere_QP_distr(N=n, 
    #                          psi_angular_coef_fct=lambda m: OAT_CSS(np.pi/2,0,n,m,mu_opt(n)), 
    #                          pt_points=100, fct_key="Wigner", 
    #                          im_name = 'OAT_opt',
    #                          alpha=0)
    # plot_ang_mom_sphere_QP_distr(N=n, 
    #                          psi_angular_coef_fct=lambda m: 1 if m==-n//2 else 0, 
    #                          pt_points=100, 
    #                          fct_key="Wigner", 
    #                          im_name = 'SP_CSS')
    # plot_ang_mom_sphere_QP_distr(N=n, 
    #                          psi_angular_coef_fct=lambda m: 1 if m==0 else 0, 
    #                          pt_points=100, 
    #                          fct_key="HusimiQ", 
    #                          im_name = 'twin_fock_HusimiQ')


    #### main text Fig. 5
    plotPhaseUncertainty(sigma_values=[.01, .02, .08], N=N)


    #### main text Fig. 6
    analytics_vs_numerics(sigma_vk=sigma_vk, N=N)


    #### main text Fig. 7
    blackman_vs_box_pulse(sigma_vk=sigma_vk, N=N)


    #### main text Fig. 8
    fock_state_after_BS(sigma_values=[.01, .02, .08], N=N)


    #### main text Fig. 9
    plot_mu_awa_phi_opt_phaseUncertainty(sigma_vk=.01, N=N, alpha_min=-.06, 
                                         alpha_max=.07, mu_min=.0008, 
                                         mu_max=.0031, n_steps=2000)

    #### Appendix Fig. 10
    plot_pert_vs_numerics_couplings(epsilon=.2, vk=.04, tau_final=4 * np.pi,
                                    g0=np.array([0,0,0,1,0,0], dtype=complex))        


    print("computation time: ", time.time() - start)




# %%
