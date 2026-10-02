import numpy as np
from isac.antenna_arrays import UniformLinearArray
from isac.channel import AWGNChannel, MultiPathChannelConfig, OFDMBeamSpaceChannel
from isac.module import OFDMConfig
from isac.utils import db2lin
from torch.utils.data import IterableDataset


class SimulationDataset(IterableDataset):
    def __init__(
        self,
        num_ant_tx: int = 4,
        num_ant_rx: int = 16,
        num_carriers: int = 64,
        num_symbols: int = 14,
        scs: float = 15e3,
        snr: float = 0.0,
        path_loss: float = 0.0,
        num_paths: int = 3,
        center_dod: float | list[float] = 0.0,
        center_doa: float | list[float] = 0.0,
        dod_spread: float = 2.0,
        doa_spread: float = 5.0,
    ) -> None:
        """Implements a dataset where each batch is generated on-the-fly, rather than loaded from
        memory.

        For all the parameters, please consult the :attr:`isac` package.
        """
        super().__init__()

        self.tx_array = UniformLinearArray(num_antennas=num_ant_tx)
        self.rx_array = UniformLinearArray(num_antennas=num_ant_rx)

        self.ofdm_config = OFDMConfig(
            subcarrier_spacing=scs, num_guard_carriers=(0, 0), Nfft=num_carriers
        )
        self.num_symbols = num_symbols
        self.symbol_duration = self.ofdm_config.symbol_time

        self.snr = db2lin(snr)
        self.path_loss = db2lin(path_loss)

        self.center_dod = np.asarray(center_dod)
        self.center_doa = np.asarray(center_doa)
        self.dod_spread = dod_spread
        self.doa_spread = doa_spread
        self.num_paths = num_paths

        self.awgn = AWGNChannel(snr_db=self.snr, sigpow_db="measured")
        self.beamspace_channel = OFDMBeamSpaceChannel(
            mpc_configs=MultiPathChannelConfig.random_generate(num_paths=self.num_paths),
            ofdm_config=self.ofdm_config,
            tx_array=self.tx_array,
            rx_array=self.rx_array,
        )

    def __iter__(self):
        return self.simulation_loop()

    def simulation_loop(self):
        """Returns a generator, which produces one batch of simulated data.

        :yield: Tuple containing a QPSK modulated signal and the multipath configuration file
        :rtype: Generator[Tuple[Any, Dict[str, Any]]]
        """
        tx_shape = (self.tx_array.num_antennas, self.ofdm_config.Nfft, self.num_symbols)
        tx_signal = np.ones(shape=tx_shape) + 1j * np.ones(shape=tx_shape)

        while True:
            path_gains = (
                1
                / np.sqrt(2 * self.path_loss)
                * (
                    np.random.normal(size=self.num_paths)
                    + 1j * np.random.normal(size=self.num_paths)
                )
            )
            path_delays = np.random.uniform(0, self.symbol_duration / 3 * 2, size=self.num_paths)
            doppler_shifts = np.random.uniform(
                -self.ofdm_config.subcarrier_spacing / 3,
                self.ofdm_config.subcarrier_spacing / 3,
                size=self.num_paths,
            )
            doas = np.random.uniform(
                low=self.center_doa - self.doa_spread / 2,
                high=self.center_doa + self.doa_spread / 2,
                size=(self.num_paths,),
            )

            dods = np.random.uniform(
                low=self.center_dod - self.dod_spread / 2,
                high=self.center_dod + self.dod_spread / 2,
                size=(self.num_paths,),
            )

            self.beamspace_channel.mpc_configs = MultiPathChannelConfig(
                path_delays=path_delays,
                path_gains=path_gains,
                doppler_shifts=doppler_shifts,
                aoas=np.deg2rad(doas),
                aods=np.deg2rad(dods),
            )

            rx_signal = self.awgn.apply(self.beamspace_channel.apply(tx_signal))

            # convert the mpc config to a dict and get rid of the leading underscore in the attribute names
            ground_truth = {
                key[1:]: val for (key, val) in self.beamspace_channel.mpc_configs.__dict__.items()
            }
            yield rx_signal, ground_truth
