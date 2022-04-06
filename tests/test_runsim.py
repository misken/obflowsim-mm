from obflowsim import obflow_sim


def test_runsim():
    assert not obflow_sim.main(['input/scenario_1.yaml'])
