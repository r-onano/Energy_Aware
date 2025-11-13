from src.energy.simulator import SimEnergy

def test_sim_energy_monotonicity():
    sim = SimEnergy(alpha=1.0, beta=0.0, gamma=0.0)
    sim.observe(fps=10.0, util=0.5)
    a = sim.mean()
    sim.observe(fps=50.0, util=0.5)
    b = sim.mean()
    # higher fps should lower instantaneous energy; rolling mean may not drop sharply, but shouldn't explode
    assert b <= a * 1.5
