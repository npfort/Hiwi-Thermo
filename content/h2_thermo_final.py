"""
Thermodynamics of ortho-, para- and normal-hydrogen
========================================================================================================
Implementation following Ratnakar, R.R. (2026), "Thermodynamics and equilibrium
thermochemistry of ortho- and para-hydrogen: Integrating quantum mechanics
with classical EOS modeling", Int. J. Hydrogen Energy 245, 155702.
https://doi.org/10.1016/j.ijhydene.2026.155702.
 
IDEAL part: quantum mechanics, Eq. (4)-(13)
RESIDUAL part: Peng-Robinson-78 + Peneloux, Eq. (14)-(30)
"""

import numpy as np

# ======================================================================================================
# CONSTANTS
# ======================================================================================================

R = 8.31446261815324            # J/(mol*K)
NA = 6.02214076e23              # 1/mol
h = 6.62607015e-34              # J*s
kB = 1.380649e-23               # J/K
c = 299792458.0                 # m/s
I_H2 = 4.67e-48                 # kg*m^2

theta_coeff = h**2 / (8 * np.pi**2 * I_H2* kB)        # from Eq.(4)

# ======================================================================================================
# PR-78 EOS PARAMETERS FOR PARA-, ORTHO- AND NORMAL HYDROGEN (from Table 1)
# ======================================================================================================
EOS_PARAMS = {
    "p":  dict(Tc=32.938, Pc=12.858e5, omega=-0.22, Vc=65.45e-6, 
               Tt=13.8033, Pt=0.07041e5, Cpen=-5.48e-6),
    "o":  dict(Tc=33.22,  Pc=13.106e5, omega=-0.22, Vc=64.78e-6, 
               Tt=14.008,  Pt=0.07461e5, Cpen=-5.76e-6),
    "n":  dict(Tc=33.145, Pc=12.964e5, omega=-0.22, Vc=65.34e-6, 
               Tt=13.957,  Pt=0.0736e5,  Cpen=-5.34e-6),
}

# ======================================================================================================
# IDEAL PART
# ======================================================================================================
# ROTATIONAL PARTITION FUNCTIONS (Eq. 4—5) ============================================================= 
def theta(j):
    return j*(j+1)*theta_coeff        # Eq.(4)

def Qj(j, T):
    return (2*j+1)*np.exp(-theta(j)/T)          # Eq.(4)

def rotational_sums(T, species, Jmax=15):       # Eq.(5)
    if species == "p":
        j_vals = np.arange(0, Jmax, 2)     # even values
        w = 1.0
    elif species == "o":
        j_vals = np.arange(1, Jmax, 2)     # odd values
        w = 3.0
    elif species == "eq":
        j_vals_even = np.arange(0, Jmax, 2)
        j_vals_odd = np.arange(1, Jmax, 2)
        Qp = 1.0 * np.sum(Qj(j_vals_even, T))
        Qo = 3.0 * np.sum(Qj(j_vals_odd, T))
        sum_theta_w_Q_p = np.sum(theta(j_vals_even) * Qj(j_vals_even, T))              # for Eq.(6)
        sum_theta_w_Q_o = 3.0 * np.sum(theta(j_vals_odd) * Qj(j_vals_odd, T))          # for Eq.(6)
        sum_theta2_w_Q_p = np.sum(theta(j_vals_even)**2 * Qj(j_vals_even, T))          # for Eq.(11)
        sum_theta2_w_Q_o = 3.0 * np.sum(theta(j_vals_odd)**2 * Qj(j_vals_odd, T))      # for Eq.(11)
        return Qp + Qo, sum_theta_w_Q_p + sum_theta_w_Q_o, sum_theta2_w_Q_p + sum_theta2_w_Q_o
    else:
        raise ValueError("Species must be 'p', 'o' or 'eq'")

    Qj_vals = Qj(j_vals, T)
    Qs = w * np.sum(Qj_vals)
    sum_theta_w_Q = w * np.sum(theta(j_vals) * Qj_vals)           # for Eq.(6)
    sum_theta2_w_Q = w * np.sum(theta(j_vals)**2 * Qj_vals)       # for Eq.(11)
    return Qs, sum_theta_w_Q, sum_theta2_w_Q

def Q_partition(T, species, Jmax=15):
    return rotational_sums(T, species, Jmax)[0]

# THERMODYNAMIC PROPERTIES (Eq. 6—13) ==================================================================
def ideal_state_pure(T, V, species, ref=None, Jmax=15):
    if ref is None:
        ref = dict(Uref=0.0, Href=0.0, Sref=0.0, Gref=0.0)        # Change for desired refecente state

    Qs, sum1, sum2 = rotational_sums(T, species, Jmax)

    Es = R * sum1 / Qs                                             # Eq.(6)
    U0 = ref["Uref"] + 1.5 * R * T + Es                            # Eq.(7)
    H0 = ref["Href"] + 2.5 * R * T + Es                            # Eq.(8)
    S0 = (ref["Sref"] + R * (np.log(V) + 1.5*np.log(T) + 1.5)
          + R * np.log(Qs) + Es / T)                               # Eq.(9)
    G0 = H0 - T * S0                                               # Eq.(10)
 
    mean_theta = sum1 / (Qs * T)                                   # for Eq.(11)
    mean_theta2 = sum2 / (Qs * T**2)                               # for Eq.(11)
    
    #Cp0 = 2.5 * R + mean_theta2 - mean_theta**2                    # Eq.(11) -- ORIGINAL paper equation
    Cp0 = 2.5 * R + R*(mean_theta2 - mean_theta**2)                # Eq.(11) -- MODIFIED
                                                                   # mean_theta and mean_theta2 are
                                                                   # dimensionless, so they must be
                                                                   # multiplied by R to match the unit
                                                                   # of 2.5*R
    
    Cv0 = Cp0 - R                                                  # Eq.(12)
 
    return dict(U=U0, H=H0, S=S0, G=G0, Cp=Cp0, Cv=Cv0, Q=Qs)

def ideal_properties(T, V, species, ref=None, Jmax=15):
    if species in ("p", "o", "eq"):
        return ideal_state_pure(T, V, species, ref, Jmax)
    elif species == "n":    # Eq.(13)
        p_coeff = ideal_state_pure(T, V, "p", ref, Jmax)
        o_coeff = ideal_state_pure(T, V, "o", ref, Jmax)
        out = {k: 0.25 * p_coeff[k] + 0.75 * o_coeff[k] for k in ("U", "H", "S", "G", "Cp", "Cv")}
        out["Q"] = p_coeff["Q"]**0.25 * o_coeff["Q"]**0.75   
        return out
    else:
        raise ValueError("Species must be 'p','o','eq' or 'n'")

# ======================================================================================================
# RESIDUAL PART: PR-78 + Peneloux (Eq. 14—30, without quantum corrections showed in Eq. 18—21)
# ======================================================================================================
alpha1 = 1 + np.sqrt(2)
alpha2 = 1 - np.sqrt(2)

def m_factor(omega):                # Eq.(15)
    if omega <= 0.49:                  # for hydrogen omega=-0.22
        return 0.37464 + 1.54226*omega - 0.26992*omega**2    # in the paper they made a mistake and 
                                                             # wrote 0.026992 instead of 0.26992
    return 0.379642 + 1.48503*omega - 0.164423*omega**2 + 0.016666*omega**3

def eos_ac_b(species):              # Eq.(16)
    params = EOS_PARAMS[species]
    ac = 0.45724 * R**2 * params["Tc"]**2 / params["Pc"]
    b  = 0.07780 * R * params["Tc"] / params["Pc"]
    m  = m_factor(params["omega"])
    return ac, b, m, params["Tc"], params["Cpen"] 

def alpha_PR(T, Tc, m):             # for Eq.(14)
    return (1 + m*(1 - np.sqrt(T/Tc)))**2
 
def a_of_T(T, species):                  # for Eq.(14)
    ac, b, m, Tc, Cpen = eos_ac_b(species)
    return ac * alpha_PR(T, Tc, m)
 
def da_dT(T, species, dT=1e-4):    # for Eq.(22)–(30)
   return (a_of_T(T+dT, species) - a_of_T(T-dT, species)) / (2*dT)
 
def d2a_dT2(T, species, dT=1e-2):  # for Eq.(22)–(30)
   return (da_dT(T+dT, species) - da_dT(T-dT, species)) / (2*dT)

def molar_volume_PR(T, P, species, phase="vapor"):    # Eq.(17)
    """
    Solves PR-78 (Eq. (14)) for molar volume V (m^3/mol), then
    applies the Peneloux volume-translation correction.
    """
    ac, b, m, Tc, Cpen = eos_ac_b(species)
    a = a_of_T(T, species)

    # ----------------------------------------------------------------------------
    # Eq.(14): P = RT/(V-b) - a(T)/[(V+alpha1*b)(V+alpha2*b)]
    # This is not solvable for V directly. Substitute the dimensionless
    # compressibility factor Z = P*V/(R*T) -> V = Z*R*T/P into Eq.(14),
    # so it becomes: 
    #   1 = 1/(Z-B) - A/[(Z+alpha1*B)(Z+alpha2*B), where:
    #      A = a*P/(R*T)^2
    #      B = b*P/(R*T)
    # ----------------------------------------------------------------------------
    A = a * P / (R*T)**2
    B = b * P / (R*T)
 
    # ----------------------------------------------------------------------------
    # Cross-multiplying the new equation turns it into a cubic polynomial:
    #   Z^3 - (1-B)*Z^2 + (A-3B^2-2B)*Z - (AB-B^2-B^3) = 0
    # ----------------------------------------------------------------------------
    coeffs = [1, -(1-B), (A - 3*B**2 - 2*B), -(A*B - B**2 - B**3)]
    roots = np.roots(coeffs)
 
    # A cubic always has 3 roots (real or complex). Physically valid roots
    # must be real and give V > b (Z > B). The largest root is the vapor
    # branch (since the larger Z, the larger V, then molecules spread out),
    # the smallest is the liquid branch (molecules packed close together)
    real_roots = [rt.real for rt in roots if abs(rt.imag) < 1e-9 and rt.real > B]
    if not real_roots:
        raise ValueError(f"No valid roots in T={T}, P={P}")
    Z_phase = max(real_roots) if phase == "vapor" else min(real_roots)
 
    V = Z_phase * R * T / P      # back from Z to V
 
    # ----------------------------------------------------------------------------
    # Eq. (17): Peneloux volume-translation correction
    # ----------------------------------------------------------------------------
    V_corr = V - Cpen
    return V_corr, Z_phase

def residual_properties(T, P, species, phase="vapor"): # Change to desired phase (liquid or vapor)
    ac, b, m, Tc, Cpen = eos_ac_b(species)
    V_corr, Z_phase = molar_volume_PR(T, P, species, phase)
    V = V_corr + Cpen
 
    a = a_of_T(T, species)
    ap = da_dT(T, species)
    app = d2a_dT2(T, species)

    # ----------------------------------------------------------------------------
    # Eq.(22): Cv_res = d/dT [ integral_inf^V (T*dP/dT - P) dV ]
    #
    # Plugging Eq.(14) into (T*dP/dT - P) and simplifying:
    #   T*dP/dT - P = (a - T*a') / [(V+alpha1*b)(V+alpha2*b)]
    #
    # Differentiating that in T gives:
    #   d/dT(a - T*a') = a' - (a' + T*a'') = -T*a''
    #
    # So Cv_res = -T*a'' * integral_inf^V dV/[(V+alpha1*b)(V+alpha2*b)].
    # That integral is a standard partial-fractions form,
    # integral dx/[(x+p)(x+q)] = 1/(p-q) * ln[(x+q)/(x+p)], evaluated from
    # infinity (where the log term -> ln(1) = 0) to V. Flipping the sign
    # (ln(a/b) = -ln(b/a)) to absorb the leading minus gives:
    # ----------------------------------------------------------------------------
    Cv_res = (T*app) / ((alpha1-alpha2)*b) * np.log((V+alpha1*b)/(V+alpha2*b))

    # Eq.(25)
    dHres = (b*R*T/(V - b)
             + (a - T*ap) / ((alpha1-alpha2)*b) * np.log((V + alpha2*b) / (V + alpha1*b))
             + a/(alpha1-alpha2) * (alpha2/(V+alpha2*b) - alpha1/(V+alpha1*b)))
 
    # Eq.(27)
    dSres = R*np.log((V-b)/V) - ap/((alpha1-alpha2)*b) * np.log((V + alpha2*b) / (V + alpha1*b))
 
    # Derivatives of P for Eq. (23), (28) and (30)
    dPdT_V = R/(V-b) - ap/((V+alpha1*b)*(V+alpha2*b))
    dPdV_T = (-R*T/(V-b)**2
              + 2*a*(V+b) / ((V+alpha1*b)*(V+alpha2*b))**2)
 
    return dict(V_corr=V_corr, V=V, Z=Z_phase, dHres=dHres, dSres=dSres, Cv_res=Cv_res,
                dPdT_V=dPdT_V, dPdV_T=dPdV_T)


# ======================================================================================================
# REAL PROPERTIES (combination of ideal and residual parts)
# ======================================================================================================
def real_properties(T, P, species, ref=None, phase="vapor", Jmax=15):
    """Combination of ideal (Eq. 6-13) + residual (Eq. 22-30) -> phi = phi0 + phi_res (Eq.1)."""
    res = residual_properties(T, P, species, phase)
    V_corr = res["V_corr"]
    ideal = ideal_properties(T, V_corr, species, ref, Jmax)
 
    Cv = ideal["Cv"] + res["Cv_res"]                                # Eq.(23)
    Cp = Cv - T * (1/res["dPdV_T"]) * (res["dPdT_V"])**2            # Eq.(23)
    H  = ideal["H"] + res["dHres"]
    S  = ideal["S"] + res["dSres"]
 
    return dict(V_corr=V_corr, Z=res["Z"], Cv=Cv, Cp=Cp, H=H, S=S,
                dPdT_V=res["dPdT_V"], dPdV_T=res["dPdV_T"])
 
 
def sonic_velocity(T, P, species, Mw=2.01588e-3, ref=None, phase="vapor"):
    """Eq.(28)"""
    props = real_properties(T, P, species, ref, phase)
    V_corr = props["V_corr"]
    return np.sqrt(-(props["Cp"]/props["Cv"]) * V_corr**2/Mw * props["dPdV_T"])
    

def isothermal_compressibility(T, P, species, Mw=2.01588e-3, ref=None, phase="vapor"):
    """Eq.(29)"""
    props = real_properties(T, P, species, ref, phase)
    v = sonic_velocity(T, P, species, Mw, ref, phase)
    rho = Mw / props["V_corr"]
    return props["Cp"] / (v**2 * rho * props["Cv"])
    
 
def joule_thomson(T, P, species, ref=None, phase="vapor"):
    """Eq. (30), [K/Pa]"""
    props = real_properties(T, P, species, ref, phase)
    return -(1/props["Cp"]) * (T * (1/props["dPdV_T"]) * props["dPdT_V"] + props["V_corr"])



if __name__ == "__main__":
    # ======================================================================================================
    # VALIDATION
    # ======================================================================================================
    from scipy.optimize import brentq
    
    results = []  # (description, passed: bool)
     
    def check(description, computed, reference, rel_tol, note=""):
        """Print one validation line and record PASS/FAIL for the summary."""
        rel_err = abs(computed - reference) / abs(reference)
        passed = rel_err <= rel_tol
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {description}")
        print(f"       computed = {computed:.6g}   reference = {reference:.6g}   "
              f"rel. error = {rel_err:.2%}  (tolerance {rel_tol:.1%})")
        if note:
            print(f"       note: {note}")
        print()
        results.append((description, passed))
     
     
    # ============================================================================================
    # SECTION 1 -- Building blocks of the ideal (quantum-mechanical) part
    # ============================================================================================
    print("="*92)
    print("SECTION 1: rotational temperature constant theta_j = j(j+1)*theta_coeff")
    print("="*92)
    print(
    """The paper (right after Eq. 4) states the numeric value of this constant explicitly:
    "theta_j = 86.28*j(j+1) [in K]". This is a pure constants check (Planck's constant,
    moment of inertia, Boltzmann constant), if it's off, every single ideal-state number
    downstream is off too, so it's the first thing worth checking.
    """
    )
    check("theta_coeff = h^2 / (8*pi^2*I*kB)", theta_coeff, 86.28, rel_tol=0.002)
     
     
    # ============================================================================================
    # SECTION 2 -- Equilibrium ortho/para thermochemistry (Eq. 32—34, Sec 3.2)
    # ============================================================================================
    print("="*92)
    print("SECTION 2: equilibrium ortho <-> para conversion")
    print("="*92)
    print(
    """pH2 is the lower-energy state, so at low T almost all H2 is para; as T rises, oH2 
    becomes populated too, and y_eq(pH2) drifts down toward the high-T statistical limit
    of 1:3 (25% para). Eq. (34) gives y_eq = Qp/(Qp+Qo). The paper states in the text
    (Sec. 3.2) that Delta_G_R=0 (i.e. Qp=Qo, i.e. y_eq=50%) happens at T*=78.8 K.
    """
    )
    T_star = brentq(lambda T: Q_partition(T, "p") / (Q_partition(T, "p") + Q_partition(T, "o")) 
                    - 0.5, 20, 200)
    check("T* where y_eq(pH2) = 50%", T_star, 78.8, rel_tol=0.005)
     
    print("Two limits that must hold no matter what (pure physics, not paper-specific):")
    print("all-para at very low T, and the 1:3 statistical ratio at very high T.")
    print()
    y_low = Q_partition(5, "p") / (Q_partition(5, "p") + Q_partition(5, "o"))
    y_high = Q_partition(2000, "p") / (Q_partition(2000, "p") + Q_partition(2000, "o"))
    check("y_eq(pH2) at T=5K  -> should approach 1 (all para at low T)",
          y_low, 1.0, rel_tol=0.01)
    check("y_eq(pH2) at T=2000K -> should approach 0.25 (statistical 1:3 limit)",
          y_high, 0.25, rel_tol=0.01)
     
     
    # ============================================================================================
    # SECTION 3 -- Ideal-state enthalpy differences at the low-T limit
    # ============================================================================================
    print("="*92)
    print("SECTION 3: ideal-state enthalpy differences, T -> 0 limit")
    print("="*92)
    print(
    """These two numbers are stated explicitly in the text (Sec. 3.2):
      - Delta H0(oH2-pH2) = 1435 kJ/kmol = 1435 J/mol
      - Delta H0(nH2-eqH2) ~ 1075 kJ/kmol = 1075 J/mol
    """
    )
    T_low = 1.0  # [K], close enough to 0 that only the ground state contributes
    Hp = ideal_properties(T_low, 1.0, "p")["H"]
    Ho = ideal_properties(T_low, 1.0, "o")["H"]
    Hn = ideal_properties(T_low, 1.0, "n")["H"]
    Heq = ideal_properties(T_low, 1.0, "eq")["H"]
     
    theta_1 = 1 * 2 * theta_coeff  # j=1 -> j(j+1)=2
    check("Delta H0(oH2-pH2)",
          Ho - Hp, 1435.0, rel_tol=0.005)
    check("Delta H0(nH2-eqH2)",
          Hn - Heq, 1075.0, rel_tol=0.01)
     
     
    # ============================================================================================
    # SECTION 4 -- Ideal-state heat capacity: theoretical limits
    # ============================================================================================
    print("="*92)
    print("SECTION 4: ideal-state Cp0 -- theoretical limits")
    print("="*92)
    print(
    """The paper states explicit theoretical bounds (Sec. 3.1.2): at low T, rotation can be
    neglected (only translation contributes, f=3 degrees of freedom) so Cp0 -> 5/2*R; at
    high T, rotation becomes fully classical so Cp0 -> 7/2*R.
    """
    )
    Cp_p_low = ideal_properties(15, 1.0, "p")["Cp"]
    Cp_p_high = ideal_properties(600, 1.0, "p")["Cp"]
    check("Cp0(pH2) at T=15K -> low-T limit 5/2*R", Cp_p_low, 2.5*R, rel_tol=0.001)
    check("Cp0(pH2) at T=600K -> high-T limit 7/2*R", Cp_p_high, 3.5*R, rel_tol=0.02)
    
     
    # ============================================================================================
    # SECTION 5 -- Real-gas limits: isothermal compressibility, Cp, sonic velocity
    # ============================================================================================
    print("="*92)
    print("SECTION 6: real-gas properties reduce to ideal-gas limits at low pressure")
    print("="*92)
    print(
    """At high T and low P, any real-gas EOS should reduce to the ideal-gas limit. This does not
    test whether the numbers in the paper are reproduced, it tests whether the residual-property
    equations (Eq. 22—30) are internally consistent, independent of the paper, using only exact 
    ideal-gas relations:
      kappa_T,ideal = 1/P             (isothermal compressibility)
      Cp,ideal - Cv,ideal = R         (Mayer's relation)
    """
    )
    T_test, P_test = 298.15, 1.0e5
    kT = isothermal_compressibility(T_test, P_test, "p")
    check("kappa_T(pH2 vapor, 298K, 1bar) vs. ideal-gas limit 1/P", kT, 1/P_test, rel_tol=0.01)
     
    props = real_properties(T_test, P_test, "p")
    check("Cp - Cv (pH2 vapor, 298K, 1bar) vs. Mayer's relation R",
          props["Cp"] - props["Cv"], R, rel_tol=0.05)
     
     
    # ============================================================================================
    # SUMMARY
    # ============================================================================================
    print("="*92)
    print("SUMMARY")
    print("="*92)
    n_pass = sum(1 for _, p in results if p)
    n_total = len(results)
    for desc, passed in results:
        print(f"  [{'PASS' if passed else 'FAIL'}] {desc}")
    print(f"\n{n_pass}/{n_total} checks passed.")