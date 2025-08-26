import sympy as sp

def harmonic_energy_invariance():
    t = sp.symbols('t')
    w = sp.symbols('omega', positive=True)
    x = sp.Function('x')(t)

    # ODE: x'' + w^2 x = 0
    ode = sp.Eq(sp.diff(x, (t, 2)) + w**2 * x, 0)

    # Energy: E = 1/2 x'^2 + 1/2 w^2 x^2
    E = sp.Rational(1,2) * sp.diff(x, t)**2 + sp.Rational(1,2) * w**2 * x**2
    dE = sp.simplify(sp.diff(E, t))
    # Substitute x'' = -w^2 x
    dE_sub = sp.simplify(dE.subs(sp.diff(x, (t,2)), -w**2 * x))
    return sp.simplify(dE_sub)  # should be 0

if __name__ == "__main__":
    assert harmonic_energy_invariance() == 0
    print("SymPy check passed: dE/dt = 0 under x'' + ω^2 x = 0")
