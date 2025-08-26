from noetheragent.formal.sympy_invariant import harmonic_energy_invariance

def test_sympy_energy_invariance_is_zero():
    assert harmonic_energy_invariance() == 0
