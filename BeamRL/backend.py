from abc import ABC, abstractmethod
from itertools import chain  
from typing import Optional
import cheetah
import numpy as np  
import ocelot as oc  
import torch  
from gymnasium import spaces
import lattice as clapa

device = torch.device('cpu')

class TransverseTuningBaseBackend(ABC):
    """Abstract class for a backend implementation of clapa"""

    def is_beam_on_screen(self) -> bool:
        """
        Returns: 'True' when beam is on screen and 'Falsa' when it isn't
        """
        pass

    def setup(self) -> None:
        """
        Prepare the accelerator for use with the environment. Should mostly be used for setting up simulations.

        Override with backend-specific implementation. Optional
        """
        pass

    @abstractmethod
    def get_magnets(self) -> np.ndarray:  
        """
        Return the magnet values as a NumPy array in order as the magnets appear in the
        accelerator.

        Override with backend-specific implementation. Must be implemented!
        """
        raise NotImplementedError

    @abstractmethod
    def set_magnets(self, values: np.ndarray) -> None:  
        """
        Set the magnets to the given values.

        The argument `magnets` will be passed as a NumPy array in the order the magnets
        appear in the accelerator.

        When applicable, this method should block until the magnet values are acutally
        set!

        Override with backend-specific imlementation. Must be implemented!
        """
        raise NotImplementedError

    def reset(self, seed=None, options=None) -> None:
        """
        Code that should set the accelerator up for a new episode. Run when the `reset`
        is called.

        Mostly meant for simulations to switch to a new incoming beam / misalignments or
        simular things.

        Override with backend-specific implementation. Optional.
        """
        pass

    def update(self) -> None:
        """
        Update accelerator metrics for later use. Use this to run the simulation or
        cache the beam image.

        Override with backend-specific imlementation. Optional.
        """
        pass

    @abstractmethod
    def get_beam_parameters(self) -> np.ndarray:  
        """
        Get the beam parameters measured on the diagnostic screen as NumPy array grouped
        by dimension (e.g. mu_x, sigma_x, mu_y, sigma_y).
        (BPM1, ion1) -- (mu_x1, mu_y1,mu_x2, mu_y2)

        Override with backend-specific implementation. Must be implemented!
        """
        raise NotImplementedError

    def get_incoming_parameters(self) -> np.ndarray:
        """
        Get all physical beam parameters of the incoming beam as NumPy array in order
        energy, mu_x, mu_xp, mu_y, mu_yp, sigma_x, sigma_xp, sigma_y, sigma_yp, sigma_s,
        sigma_p.

        Override with backend-specific implementation. Optional.
        """
        raise NotImplementedError

    def get_misalignments(self) -> np.ndarray:
        """
        Get misalignments of the quadrupoles and the diagnostic screen as NumPy array in
        order AREAMQZM1.misalignment.x, AREAMQZM1.misalignment.y,
        AREAMQZM2.misalignment.x, AREAMQZM2.misalignment.y, AREAMQZM3.misalignment.x,
        AREAMQZM3.misalignment.y, AREABSCR1.misalignment.x, AREABSCR1.misalignment.y.

        Override with backend-specific imlementation. Optional.
        """
        raise NotImplementedError

    def get_screen_image(self) -> np.ndarray:
        """
        Retreive the beam image as a 2-dimensional NumPy array.

        Note that if reading the beam image is expensive, it is best to cache the image
        in the `update_accelerator` method and the read the cached variable here.

        Ideally, the pixel values should look somewhat similar to the 12-bit values from
        the real screen camera.

        Override with backend-specific implementation. Optional.
        """
        raise NotImplementedError
    
    def get_binning(self) -> np.ndarray:
        """
        Return binning currently set on the screen camera as NumPy array [x, y].

        Override with backend-specific implementation. Must be implemented!
        """
        raise NotImplementedError

    def get_screen_resolution(self) -> np.ndarray:
        """
        Return (binned) resolution of the screen camera as NumPy array [x, y].

        Override with backend-specific implementation. Must be implemented!
        """
        raise NotImplementedError

    def get_pixel_size(self) -> np.ndarray:
        """
        Return the (binned) size of the area on the diagnostic screen covered by one
        pixel as NumPy array [x, y].

        Override with backend-specific implementation. Must be implemented!
        """
        raise NotImplementedError

    def get_info(self) -> dict:
        """
        Return a dictionary of aditional info from the accelerator backend, e.g.
        incoming beam and misalignments in simulation.

        Override with backend-specific implementation. Optional.
        """
        return {},

class CheetahBackend(TransverseTuningBaseBackend):  
    """"""

    def __init__(
            self,
            ocelot_cell: list[oc.Element],
            BPM_names: list[str],
            ION_names: list[str],  
            magnet_names: list[str],  
            incoming_mode: str = "random",
            incoming_values: Optional[np.ndarray] = None,
            max_misalignment: float = 5e-4,
            misalignment_mode: str = "random",
            misalignment_values: Optional[np.ndarray] = None,
            simulate_finite_screen: bool = False,
    ) -> None:
        self.BPM_name = BPM_names
        self.ION_names = ION_names
        self.magnet_names = magnet_names
        self.incoming_mode = incoming_mode
        self.incoming_values = incoming_values
        self.max_misalignment = max_misalignment
        self.misalignment_mode = misalignment_mode
        self.misalignment_values = misalignment_values
        self.simulate_finite_screen = simulate_finite_screen
        self.property_names = [
            self.get_property_name(magnet_name) for magnet_name in self.magnet_names
        ]

        solenoid_names = [name for name in self.magnet_names if name[0] == "S"]
        quadrupole_names = [name for name in self.magnet_names if name[0] == "Q"]  
        hcor_names = [name for name in self.magnet_names if name[0] == "H"]
        n_misalignments = 2 * (len(solenoid_names) + len(quadrupole_names))
        
        self.incoming_beam_space = spaces.Box(
            low=np.array(
                [
                    5e6,   
                    -1e-3,
                    -1e-4,
                    -1e-3,
                    -1e-4,
                    1e-5,
                    1e-6,
                    1e-5,
                    1e-6,
                    1e-6,
                    1e-4,],
                dtype=np.float32
            ),
            high=np.array(
                [20e6, 1e-3, 1e-4, 1e-3, 1e-4, 5e-4, 5e-5, 5e-4, 5e-5, 5e-5, 1e-3],
                dtype=np.float32,
            ),
        )

        self.misalignment_space = spaces.Box(
            low=-self.max_misalignment,
            high=self.max_misalignment,
            shape=(n_misalignments, )
        )

        self.segment = cheetah.Segment.from_ocelot(
            ocelot_cell, warnings=False, 
        )  

        self.solenoids = [getattr(self.segment, name) for name in solenoid_names]
        self.quadrupoles = [getattr(self.segment, name) for name in quadrupole_names]
        self.hcors = [getattr(self.segment, name) for name in hcor_names]
        self.bpm = [getattr(self.segment, name) for name in BPM_names]
        self.ion = [getattr(self.segment, name) for name in ION_names]

        for i in range(2):
            self.ion[i].binning = 1
            self.bpm[i].is_active = True
            self.ion[i].is_active = True     

    def is_beam_on_screen(self) -> list[bool]:
        beam_position_1 = np.array(
            [self.ion[0].get_read_beam.mu_x, self.ion[0].get_read_beam.mu_y]
        )
        beam_position_2 = np.array(
            [self.ion[1].get_read_beam.mu_x, self.ion[1].get_read_beam.mu_y]
        )
        
        limits = np.array(self.ion[0].resolution) / 2 * np.array(self.ion[0].pixel_size)
        return [np.all(np.abs(beam_position_1) < limits), np.all(np.abs(beam_position_2) < limits)]

    def get_magnets(self) -> np.ndarray:
        return np.array(
            [
                getattr(getattr(self.segment, magnet_name), property_name).cpu()
                for magnet_name, property_name in zip(
                    self.magnet_names, self.property_names
                )
            ],
            dtype=np.float32
        )

    def set_magnets(self, values: np.ndarray) -> None:
        for magnet_name, property_name, value in zip(
            self.magnet_names, self.property_names, values
        ):
            magnet = getattr(self.segment, magnet_name)
            setattr(magnet, property_name, torch.tensor(value, device=device, dtype=torch.float32))

    def reset(self, seed=None, options=None) -> None:
        
        if self.incoming_mode == "constant":
            incoming_parameters = self.incoming_values
        elif self.incoming_mode == "random":
            incoming_parameters = self.incoming_beam_space.sample()
        else:
            raise ValueError(f"Invalid value '{self.incoming_mode}' for incoming mode")
        self.incoming = cheetah.ParameterBeam.from_parameters(
            energy=torch.tensor(incoming_parameters[0], device=device),
            mu_x=torch.tensor(incoming_parameters[1], device=device),
            mu_px=torch.tensor(incoming_parameters[2], device=device),
            mu_y=torch.tensor(incoming_parameters[3], device=device),
            mu_py=torch.tensor(incoming_parameters[4], device=device),
            sigma_x=torch.tensor(incoming_parameters[5], device=device),
            sigma_px=torch.tensor(incoming_parameters[6], device=device),
            sigma_y=torch.tensor(incoming_parameters[7], device=device),
            sigma_py=torch.tensor(incoming_parameters[8], device=device),
            sigma_tau=torch.tensor(incoming_parameters[9], device=device),
            sigma_p=torch.tensor(incoming_parameters[10], device=device),
            device=device,
            dtype=torch.float32
        )

        if self.misalignment_mode == "constant":
            misalignments = self.misalignment_values
        elif self.misalignment_mode == "random":
            misalignments = self.misalignment_space.sample()
        else:
            raise ValueError(
                f'Invalid value "{self.misalignment_mode}" for misalignment_mode'
            )

        for i, solenoid in enumerate(self.solenoids):
            solenoid.misalignment = torch.tensor(misalignments[2 * i: 2 * i + 2], device=device, dtype=torch.float32)

        for i, quadrupole in enumerate(self.quadrupoles):
            quadrupole.misalignment = torch.tensor(misalignments[2 * (i + 3): 2 * (i + 3) + 2], device=device, dtype=torch.float32)

    def update(self) -> None:
        self.segment(self.incoming)  

    def get_beam_parameters(self) -> np.ndarray:
        
        return np.array(
            [
                self.ion[0].get_read_beam().sigma_x.cpu(),
                self.ion[0].get_read_beam().sigma_y.cpu(),
                self.bpm[0].reading[0].cpu(),
                self.bpm[0].reading[1].cpu(),
                self.bpm[1].reading[0].cpu(),
                self.bpm[1].reading[1].cpu(),
                self.ion[1].get_read_beam().sigma_x.cpu(),
                self.ion[1].get_read_beam().sigma_y.cpu(),
            ],
            dtype=np.float32
        )

    def get_incoming_parameters(self) -> np.ndarray:
        return np.array(
            [
                self.incoming.energy.cpu(),
                self.incoming.mu_x.cpu(),
                self.incoming.mu_px.cpu(),
                self.incoming.mu_y.cpu(),
                self.incoming.mu_py.cpu(),
                self.incoming.sigma_x.cpu(),
                self.incoming.sigma_px.cpu(),
                self.incoming.sigma_y.cpu(),
                self.incoming.sigma_py.cpu(),
                self.incoming.sigma_tau.cpu(),
                self.incoming.sigma_p.cpu(),
            ]
        )

    def get_misalignments(self) -> np.ndarray:
        quadrupole_misalignments = chain.from_iterable(
            [quadrupole.misalignment for quadrupole in self.quadrupoles]
        )
        solenoid_misalignment = chain.from_iterable(
            [solenoid.misalignment for solenoid in self.solenoids]
        )
        all_misalignments = chain.from_iterable(
            [solenoid_misalignment, quadrupole_misalignments]
        )
        
        return np.array([t.cpu().numpy() for t in all_misalignments], dtype=np.float32)  

    def get_binning(self) -> np.ndarray:
        return np.array(self.ion[0].binning)

    def get_info(self) -> dict:
        return {
            "incoming_beam": self.get_incoming_parameters(),
            "misalignments": self.get_misalignments(),
        }

    def get_property_name(self, magnet_name: str) -> str:
        """
        Figure out the correct property name depending on the magnet type, inferring the
        latter from its name according to DOOCS conventions.
        """
        type_indicator = magnet_name[0]
        if type_indicator == "Q":
            return "k1"
        elif type_indicator == "H":
            return "angle"
        elif type_indicator == "S":
            return "k"
        else:
            raise ValueError(f"Cannot determine property for magnet {magnet_name}")
class DOOCSBackend(TransverseTuningBaseBackend, ABC):
    """
    the real backend by pydoocs
    """
    pass

class EACheetahBackend(CheetahBackend):
    """Cheetah simulation backend to the ARES Experimental Area."""
    def __init__(
            self,
            incoming_mode: str = "random",
            incoming_values: Optional[np.array] = None,
            max_misalignment: float = 5e-4,
            misalignment_mode: str = "random",
            misalignment_values: Optional[np.ndarray] = None,
            simulate_finite_screen: bool = False,
    ) -> None:
        super().__init__(
            ocelot_cell=clapa.cell,

            magnet_names=[
                "S1",
                "S2",
                "S3",
                "Q1",
                "Q2",
                "H1",
                
                "Q3",
                "Q4",
                "H2",
                
                "Q5",
                "Q6",
                "H3",
                
                "Q7",
                "Q8",
                
                "Qj6",
                "Qj7",
                "Qj8"
            ],
            incoming_mode=incoming_mode,
            incoming_values=incoming_values,
            max_misalignment=max_misalignment,
            misalignment_mode=misalignment_mode,
            misalignment_values=misalignment_values,
            BPM_names=[
                "BPM1",
                "BPM2",
            ],
            ION_names=[
                "BSC1",
                "BSC2"
            ],
            simulate_finite_screen=simulate_finite_screen,
        )

class EAOcelotBackend(TransverseTuningBaseBackend):
    """Backend simulating the ARES EA in Ocelot."""
    pass

class EADOOCSBackend(DOOCSBackend):
    """
    Backend for the ARES EA to communicate with the real accelerator through the DOOCS
    control system.
    """
    def __init__(self) -> None:
        super().__init__(
            screen_name="AR.EA.BSC.R.1",
            magnet_names=[
                "AREAMQZM1",
                "AREAMQZM2",
                "AREAMCVM1",
                "AREAMQZM3",
                "AREAMCHM1",
            ],
        )