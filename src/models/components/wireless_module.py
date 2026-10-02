# import numpy as np
# from typing import Any, Optional, Callable, Literal, Dict
# from isac.utils import db2lin
# from resilient_comms.metrics import rx_performance_metric
# from resilient_comms.sim_helpers import get_channels
# from resilient_comms.bfra import iterative_waterfilling_rb_allocation
# from resilient_comms.receivers import mmse_receiver, max_sinr_receiver
# from resilient_comms.jammer import opt_jammer

# class WirelessModule:
#     def __init__(
#             self,
#             scenario: Literal["no_jammer", "no_protection", "w_protection"] = "no_jammer",
#             num_users: int = 3,
#             num_tx: int = 8,
#             num_rx: int = 16,
#             num_tx_jammer: int = 64,
#             bfra_algo: Optional[Callable] = None,
#             rx_algo: Optional[Callable] = None,
#             jammer_algo: Literal["opt_jammer", "barrage_jammer"] = "opt_jammer",
#             power_jammer_db: float = 30,
#             num_subcarriers: int = 64,
#             num_symbols: int = 14,
#             eta: float = 1.,
#             power_constraints_db: Optional[list | np.ndarray] = None,
#             num_rbs_per_user: Optional[list | np.ndarray] = None,
#             noise_power_db: float = -3.,
#             center_aoa: float = 0.,
#             aoa_spread: float = 10.,
#             aoa_spacing: float = 10.,
#             center_aoa_jammer: float = 20.,
#             aoa_spread_jammer: float = 5., 
#             path_loss: float = 10., 
#             num_paths: int = 128,
#             fc: float = 2.4e9
#         ) -> None:
#         """Models the transmission and reception under approximate worst-case MIMO-OFDM MAC adversarial attacks. 

#         :param num_users: Number of users in the system, defaults to 3
#         :type num_users: int, optional
#         :param num_tx: Number of transmit antennas for each user, defaults to 8
#         :type num_tx: int, optional
#         :param num_rx: Number of receive antennas at legitimate Rx, defaults to 16
#         :type num_rx: int, optional
#         :param num_tx_jammer: Number of transmit antennas at jammer, defaults to 64
#         :type num_tx_jammer: int, optional
#         :param bfra_algo: Beamforming, power allocation and scheduling strategy, defaults to None
#         :type bfra_algo: Optional[Callable], optional
#         :param rx_algo: Receive strategy, defaults to None
#         :type rx_algo: Optional[Callable], optional
#         :param jammer_algo: Jamming strategy, defaults to "opt_jammer"
#         :type jammer_algo: Literal["opt_jammer", "barrage_jammer"], optional
#         :param power_jammer_db: Jamming power in dB, defaults to 30
#         :type power_jammer_db: float, optional
#         :param num_subcarriers: Number of subcarriers per slot, defaults to 64
#         :type num_subcarriers: int, optional
#         :param num_symbols: Number of symbols per slot, defaults to 14
#         :type num_symbols: int, optional
#         :param eta: Resilience hyperparameter, defaults to 1.
#         :type eta: float, optional
#         :param power_constraints_db: Power constraints for each user, defaults to None.
#         If None, then all users will have a power constraint of 5 dBm.
#         :type power_constraints_db: Optional[list |  np.ndarray], optional
#         :param num_rbs_per_user: Defaults to None.
#         If None, all users will be allocated equal resources.
#         :type num_rbs_per_user: Optional[list |  np.ndarray], optional
#         :param noise_power_db: Defaults to -3.
#         :type noise_power_db: float, optional
#         :param center_aoa: Central angle of legitimate signals, defaults to 0.
#         :type center_aoa: float, optional
#         :param aoa_spread: Angle spread of legitimate signals, defaults to 10.
#         :type aoa_spread: float, optional
#         :param aoa_spacing: Angle spacing between legitimate users, defaults to 10.
#         :type aoa_spacing: float, optional
#         :param center_aoa_jammer: Central direction of jamming signals, defaults to 20.
#         :type center_aoa_jammer: float, optional
#         :param aoa_spread_jammer: Angle spread for jamming signals, defaults to 5.
#         :type aoa_spread_jammer: float, optional
#         :param path_loss: Path loss in dB, defaults to 10.
#         :type path_loss: float, optional
#         :param num_paths: Number of propagation paths, defaults to 128
#         :type num_paths: int, optional
#         :param fc: Carrier frequency, defaults to 2.4e9
#         :type fc: float, optional
#         """        
#         self.scenario = scenario
#         # general numerology
#         self.num_users = num_users
#         self.num_tx = num_tx
#         self.num_rx = num_rx
#         self.num_subcarriers = num_subcarriers
#         self.num_symbols = num_symbols
#         self.num_rbs = self.num_subcarriers * self.num_symbols
#         if power_constraints_db is None:
#             self.power_constraints = db2lin(5.) * np.ones((self.num_users,)) 
#         else:
#             self.power_constraints = db2lin(np.asarray(power_constraints_db))
        
#         if num_rbs_per_user is None:
#             self.num_rbs_per_user = np.floor(self.num_rbs / self.num_users * np.ones((self.num_users,))).astype(np.uint64)
        
#         # jammer settings
#         self.num_tx_jammer = num_tx_jammer
#         self.power_jammer = db2lin(power_jammer_db)
#         # jamming strategy
#         # self.jammer_algo = jammer_algo # choose between opt_jammer and barrage_jammer
#         self.jammer_algo = "barrage_jammer" # Test for now 

#         # tx and rx strategies
#         if bfra_algo is None:
#             self.bfra_algo = iterative_waterfilling_rb_allocation
#         else:    
#             self.bfra_algo = bfra_algo

#         if rx_algo is None:
#             self.rx_algo = max_sinr_receiver
#         else:
#             self.rx_algo = rx_algo
#         self.eta = eta
#         # channel settings
#         self.noise_power = db2lin(noise_power_db)
#         self.center_aoa = center_aoa
#         self.aoa_spread = aoa_spread
#         self.aoa_spacing = aoa_spacing
#         self.center_aoa_jammer = center_aoa_jammer
#         self.aoa_spread_jammer = aoa_spread_jammer
#         self.path_loss = path_loss
#         self.num_paths = num_paths
#         self.fc = fc

#     def run(self) -> Dict[str, np.ndarray]:
#         """Runs the wireless simulation and returns the communication MSEs for each user
#         in the protected, unprotected and jammer-less cases.

#         :return: Communication MSEs as :class`np.ndarray` with shape (num_users,).
#         :rtype: np.ndarray
#         """        
#         channels, channel_jammer, _, rx_arr, _, mpcc_jammer = get_channels(
#             num_users=self.num_users, num_rx=self.num_rx, num_tx=self.num_tx,
#             num_tx_jammer=self.num_tx_jammer, aoa_spread=self.aoa_spread,
#             aoa_spacing=self.aoa_spacing, center_aoa=self.center_aoa,
#             center_aoa_jammer=self.center_aoa_jammer, aoa_spread_jammer=self.aoa_spread_jammer,
#             path_loss=self.path_loss, fc=self.fc,
#             num_sc=self.num_subcarriers, num_syms=self.num_symbols,
#             num_paths=self.num_paths 
#         )
#         noise_covariance = (self.noise_power / self.num_rx)*np.eye(self.num_rx)
#         noise_covariance = noise_covariance.reshape(1, 1, *noise_covariance.shape)

#         aoas = mpcc_jammer.aoas
#         steering_mat = rx_arr.steering_matrix(grid=aoas, axis=1)
#         tmp = steering_mat @ steering_mat.conj().T
#         cov_mat_aoa = self.eta * tmp.reshape(1, 1, *tmp.shape) + noise_covariance
    
#         allocs, pows, precoders, receivers, mses_no_jammer = \
#             self.run_algorithm(channels=channels, noise_covariance=noise_covariance)

#         if self.jammer_algo == "opt_jammer":
#             jammer_cov = opt_jammer(channel_jammer=channel_jammer, 
#                                     p_j=self.power_jammer, 
#                                     channels=channels, 
#                                     pows=pows, 
#                                     allocs=allocs)
#         elif self.jammer_algo == "barrage_jammer":  # "barrage_jammer"
#             jammer_cov = self.barrage_jammer_covariance(channel_jammer=channel_jammer)
        
#         cov_mat_jammer = channel_jammer @ jammer_cov @ channel_jammer.transpose((0, 2, 1)).conj() + noise_covariance
#         mses_no_protection = rx_performance_metric(
#             channels=channels, noise_covariance=cov_mat_jammer, 
#             receivers=receivers, precoders=precoders, powers=pows, allocs=allocs
#         )[-1]

#         allocs_p, pows_p, precoders_p, receivers_p, _ = \
#             self.run_algorithm(channels=channels, noise_covariance=cov_mat_aoa)
#         mses_protection = rx_performance_metric(
#             channels=channels, noise_covariance=cov_mat_jammer, 
#             receivers=receivers_p, precoders=precoders_p, powers=pows_p, allocs=allocs_p
#         )[-1]

#         return {"no_jammer": mses_no_jammer, "no_protection": mses_no_protection, "w_protection": mses_protection}
    
#     def __call__(self) -> list[float]:
#         """Wrapper around :class:`self.run()` which returns the value for the correct scenario.

#         :return: MSEs for the scenario defined by :class:`self.scenario` as a list of floats
#         with length :class:`self.num_users`.
#         :rtype: list[float]
#         """        
#         res = self.run()
#         return res[self.scenario].tolist()

#     def run_algorithm(self, channels: np.ndarray, noise_covariance: np.ndarray, **kwargs):
#         """Runs tx and receive strategies using the provided channels and noise covariance.

#         :param channels: Channel matrices for each user and RB as tensor (num_users, num_rbs, num_rx, num_tx)
#         :type channels: np.ndarray
#         :param noise_covariance: Noise covariances in the same format (num_users, num_rbs, num_rx, num_rx)
#         :type noise_covariance: np.ndarray
#         :return: Per user MSEs
#         :rtype: np.ndarray
#         """        
#         allocs, pows, precoders, _ = self.bfra_algo(
#             channels=channels, 
#             noise_covariance=noise_covariance, 
#             power_constraints=self.power_constraints, 
#             num_rbs_per_user=self.num_rbs_per_user, 
#             verbose=False,
#             **kwargs)
#         receivers = self.rx_algo(
#             channels=channels, 
#             precoders=precoders, 
#             powers=pows, 
#             allocs=allocs, 
#             noise_covariance=noise_covariance, 
#             **kwargs)
#         mses = rx_performance_metric(
#             channels=channels, 
#             receivers=receivers, 
#             precoders=precoders, 
#             powers=pows, 
#             allocs=allocs, 
#             noise_covariance=noise_covariance
#         )[-1]
#         return allocs, pows, precoders, receivers, mses
    
#     def barrage_jammer_covariance(self, channel_jammer: np.ndarray) -> np.ndarray:
#         """Generates the covariance matrix for a barrage jammer.

#         :param channel_jammer: The channel matrix for the jammer
#         :type channel_jammer: np.ndarray
#         :return: The covariance matrix for the barrage jammer
#         :rtype: np.ndarray
#         """
#         num_rx_jammer, num_tx_jammer = channel_jammer.shape[1:3]
#         power_jammer = self.power_jammer / (num_tx_jammer * self.num_rbs)
#         cov_matrix = power_jammer * np.eye(num_tx_jammer)
#         return cov_matrix



import numpy as np
from typing import Any, Optional, Callable, Literal, Dict
from isac.utils import db2lin
from resilient_comms.metrics import rx_performance_metric
from resilient_comms.sim_helpers import get_channels
from resilient_comms.bfra import iterative_waterfilling_rb_allocation
from resilient_comms.receivers import mmse_receiver, max_sinr_receiver
from resilient_comms.jammer import opt_jammer

class WirelessModule:
    def __init__(
            self,
            scenario: Literal["no_jammer", "no_protection", "w_protection"] = "no_jammer",
            num_users: int = 3,
            num_tx: int = 8,
            num_rx: int = 16,
            num_tx_jammer: int = 64,
            bfra_algo: Optional[Callable] = None,
            rx_algo: Optional[Callable] = None,
            power_jammer_db: float = 30,
            num_subcarriers: int = 64,
            num_symbols: int = 14,
            eta: float = 1.,
            power_constraints_db: Optional[list | np.ndarray] = None,
            num_rbs_per_user: Optional[list | np.ndarray] = None,
            noise_power_db: float = -3.,
            center_aoa: float = 0.,
            aoa_spread: float = 10.,
            aoa_spacing: float = 10.,
            center_aoa_jammer: float = 20.,
            aoa_spread_jammer: float = 5., 
            path_loss: float = 10., 
            num_paths: int = 128,
            fc: float = 2.4e9
        ) -> None:
        """Models the transmission and reception und approximate worst-case MIMO-OFDM MAC adversarial attacks. 

        :param num_users: Number of users in the system, defaults to 3
        :type num_users: int, optional
        :param num_tx: Number of transmit antennas for each user, defaults to 8
        :type num_tx: int, optional
        :param num_rx: Number of receive antennas at legitimate Rx, defaults to 16
        :type num_rx: int, optional
        :param num_tx_jammer: Number of transmit antennas at jammer, defaults to 64
        :type num_tx_jammer: int, optional
        :param bfra_algo: Beamforming, power allocation and scheduling strategy, defaults to None
        :type bfra_algo: Optional[Callable], optional
        :param rx_algo: Receive strategy, defaults to None
        :type rx_algo: Optional[Callable], optional
        :param power_jammer_db: Jamming power in dB, defaults to 30
        :type power_jammer_db: float, optional
        :param num_subcarriers: Number of subcarriers per slot, defaults to 64
        :type num_subcarriers: int, optional
        :param num_symbols: Number of symbols per slot, defaults to 14
        :type num_symbols: int, optional
        :param eta: Resilience hyperparameter, defaults to 1.
        :type eta: float, optional
        :param power_constraints_db: Power constraints for each user, defaults to None.
        If None, then all users will have a power constraint of 5 dBm.
        :type power_constraints_db: Optional[list  |  np.ndarray], optional
        :param num_rbs_per_user: _description_, defaults to None.
        If None, all users will be allocated equal resources.
        :type num_rbs_per_user: Optional[list  |  np.ndarray], optional
        :param noise_power_db: _description_, defaults to -3.
        :type noise_power_db: float, optional
        :param center_aoa: Central angle of legitimate signals, defaults to 0.
        :type center_aoa: float, optional
        :param aoa_spread: Angle spread of legitimate signals, defaults to 10.
        :type aoa_spread: float, optional
        :param aoa_spacing: Angle spacing between legitimate users, defaults to 10.
        :type aoa_spacing: float, optional
        :param center_aoa_jammer: Central direction of jamming signals, defaults to 20.
        :type center_aoa_jammer: float, optional
        :param aoa_spread_jammer: Angle spread for jamming signals, defaults to 5.
        :type aoa_spread_jammer: float, optional
        :param path_loss: Path loss in dB, defaults to 10.
        :type path_loss: float, optional
        :param num_paths: Number of propagation paths, defaults to 128
        :type num_paths: int, optional
        :param fc: Carrier frequency, defaults to 2.4e9
        :type fc: float, optional
        """        
        self.scenario = scenario
        # general numerology
        self.num_users = num_users
        self.num_tx = num_tx
        self.num_rx = num_rx
        self.num_subcarriers = num_subcarriers
        self.num_symbols = num_symbols
        self.num_rbs = self.num_subcarriers * self.num_symbols
        if power_constraints_db is None:
            self.power_constraints = db2lin(5.) * np.ones((self.num_users,)) 
        else:
            self.power_constraints = db2lin(np.asarray(power_constraints_db))
        
        if num_rbs_per_user is None:
            self.num_rbs_per_user = np.floor(self.num_rbs / self.num_users * np.ones((self.num_users,))).astype(np.uint64)
        
        # jammer settings
        self.num_tx_jammer = num_tx_jammer
        self.power_jammer = db2lin(power_jammer_db)
        # tx and rx strategies
        if bfra_algo is None:
            self.bfra_algo = iterative_waterfilling_rb_allocation
        else:    
            self.bfra_algo = bfra_algo

        if rx_algo is None:
            self.rx_algo = max_sinr_receiver
        else:
            self.rx_algo = rx_algo
        self.eta = eta
        # channel settings
        self.noise_power = db2lin(noise_power_db)
        self.center_aoa = center_aoa
        self.aoa_spread = aoa_spread
        self.aoa_spacing = aoa_spacing
        self.center_aoa_jammer = center_aoa_jammer
        self.aoa_spread_jammer = aoa_spread_jammer
        self.path_loss = path_loss
        self.num_paths = num_paths
        self.fc = fc

    def run(self) -> Dict[str, np.ndarray]:
        """Runs the wireless simulation and returns the communication MSEs for each user
        in the protected, unprotected and jammer-less cases.

        :return: Communication MSEs as :class`np.ndarray` with shape (num_users,).
        :rtype: np.ndarray
        """        
        channels, channel_jammer, _, rx_arr, _, mpcc_jammer = get_channels(
            num_users=self.num_users, num_rx=self.num_rx, num_tx=self.num_tx,
            num_tx_jammer=self.num_tx_jammer, aoa_spread=self.aoa_spread,
            aoa_spacing=self.aoa_spacing, center_aoa=self.center_aoa,
            center_aoa_jammer=self.center_aoa_jammer, aoa_spread_jammer=self.aoa_spread_jammer,
            path_loss=self.path_loss, fc=self.fc,
            num_sc=self.num_subcarriers, num_syms=self.num_symbols,
            num_paths=self.num_paths 
        )
        noise_covariance = (self.noise_power / self.num_rx)*np.eye(self.num_rx)
        noise_covariance = noise_covariance.reshape(1, 1, *noise_covariance.shape)

        aoas = mpcc_jammer.aoas
        steering_mat = rx_arr.steering_matrix(grid=aoas, axis=1)
        tmp = steering_mat @ steering_mat.conj().T
        cov_mat_aoa = self.eta * tmp.reshape(1, 1, *tmp.shape) + noise_covariance
    
        allocs, pows, precoders, receivers, mses_no_jammer = \
            self.run_algorithm(channels=channels, noise_covariance=noise_covariance)

        opt_jammer_cov = opt_jammer(channel_jammer=channel_jammer, 
                                    p_j=self.power_jammer, 
                                    channels=channels, 
                                    pows=pows, 
                                    allocs=allocs)
        cov_mat_opt = channel_jammer @ opt_jammer_cov @ channel_jammer.transpose((0, 2, 1)).conj() + noise_covariance
        mses_no_protection = rx_performance_metric(
            channels=channels, noise_covariance=cov_mat_opt, 
            receivers=receivers, precoders=precoders, powers=pows, allocs=allocs
        )[-1]

        allocs_p, pows_p, precoders_p, receivers_p, _ = \
            self.run_algorithm(channels=channels, noise_covariance=cov_mat_aoa)
        mses_protection = rx_performance_metric(
            channels=channels, noise_covariance=cov_mat_opt, 
            receivers=receivers_p, precoders=precoders_p, powers=pows_p, allocs=allocs_p
        )[-1]

        return {"no_jammer": mses_no_jammer, "no_protection": mses_no_protection, "w_protection": mses_protection}
    
    def __call__(self) -> list[float]:
        """Wrapper around :class:`self.run()` which returns the value for the correct scenario.

        :return: MSEs for the scenario defined by :class:`self.scenario` as a list of floats
        with length :class:`self.num_users`.
        :rtype: list[float]
        """        
        res = self.run()
        return res[self.scenario].tolist()

    def run_algorithm(self, channels: np.ndarray, noise_covariance: np.ndarray, **kwargs):
        """Runs tx and receive strategies using the provided channels and noise covariance.

        :param channels: Channel matrices for each user and RB as tensor (num_users, num_rbs, num_rx, num_tx)
        :type channels: np.ndarray
        :param noise_covariance: Noise covariances in the same format (num_users, num_rbs, num_rx, num_rx)
        :type noise_covariance: np.ndarray
        :return: Per user MSEs
        :rtype: np.ndarray
        """        
        allocs, pows, precoders, _ = self.bfra_algo(
            channels=channels, 
            noise_covariance=noise_covariance, 
            power_constraints=self.power_constraints, 
            num_rbs_per_user=self.num_rbs_per_user, 
            verbose=False,
            **kwargs)
        receivers = self.rx_algo(
            channels=channels, 
            precoders=precoders, 
            powers=pows, 
            allocs=allocs, 
            noise_covariance=noise_covariance, 
            **kwargs)
        mses = rx_performance_metric(
            channels=channels, 
            receivers=receivers, 
            precoders=precoders, 
            powers=pows, 
            allocs=allocs, 
            noise_covariance=noise_covariance
        )[-1]
        return allocs, pows, precoders, receivers, mses    