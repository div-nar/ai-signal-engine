from strategy.risk import risk_off_cash, needs_rebalance

LAYER_MAP = {"A": "power", "B": "power", "C": "compute"}


def test_risk_off_only_on_extreme():
    assert risk_off_cash(credit_stress=False, vix=45.0) == 0.0
    assert risk_off_cash(credit_stress=True, vix=20.0) == 0.0
    assert risk_off_cash(credit_stress=True, vix=45.0) == 0.30


def test_no_rebalance_when_within_bands():
    cur = {"A": 0.50, "B": 0.10, "C": 0.40}
    tgt = {"A": 0.51, "B": 0.10, "C": 0.39}
    assert needs_rebalance(cur, tgt, LAYER_MAP) is False


def test_rebalance_on_name_drift():
    cur = {"A": 0.50, "B": 0.10, "C": 0.40}
    tgt = {"A": 0.40, "B": 0.20, "C": 0.40}  # A drifts 0.10 > 0.03
    assert needs_rebalance(cur, tgt, LAYER_MAP) is True


def test_rebalance_on_layer_drift():
    # Names within band individually but the layer aggregate shifts.
    cur = {"A": 0.30, "B": 0.30, "C": 0.40}  # power = 0.60
    tgt = {"A": 0.32, "B": 0.32, "C": 0.36}  # power = 0.64 -> 0.04 > 0.03
    assert needs_rebalance(cur, tgt, LAYER_MAP) is True


def test_rebalance_when_name_enters_or_exits():
    cur = {"A": 1.0}
    tgt = {"C": 1.0}
    assert needs_rebalance(cur, tgt, LAYER_MAP) is True
