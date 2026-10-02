import pytest
from src.models.components.wireless_module import WirelessModule

@pytest.mark.parametrize("num_users", [3, 7])
@pytest.mark.parametrize("num_subcarriers", [16, 64])
@pytest.mark.parametrize("center_doa_jammer", [0., 20.])
def test_wireless_module(num_users: int, num_subcarriers: int, center_doa_jammer: float):
    mses = WirelessModule(num_users=num_users, num_subcarriers=num_subcarriers, center_aoa_jammer=center_doa_jammer).run()
    print(mses)
    assert isinstance(mses, dict)
    assert all((len(m) == num_users for m in mses.values()))
    assert len(list(mses.keys())) == 3 # number of scenarios

    mse_ = WirelessModule(num_users=num_users, num_subcarriers=num_subcarriers)()
    assert len(mse_) == num_users
    assert all((isinstance(v, float) for v in mse_))