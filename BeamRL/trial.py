from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
import torch

"""
funtions for load trial 
"""


@dataclass
class Trial:
    """Reperesents conditions for a trial run in the ARES EA."""

    target_beam: torch.Tensor
    incoming_beam: torch.Tensor
    misalignments: torch.Tensor
    initial_magnets: torch.Tensor


def load_trials(filepath: Path) -> list[Trial]:
    """Load a set of trials from a `.yaml` file."""
    with open(filepath, "r") as f:
        raw = yaml.full_load(f.read())

    converted = []
    for i in sorted(raw.keys()):
        raw_trial = raw[i]

        target_beam = torch.tensor(target_beam_from_dictionary(raw_trial["target"]))
        incoming_beam = torch.tensor(incoming_beam_from_dictionary(raw_trial["incoming"]))
        misalignments = torch.tensor(misalignments_from_dictionary(raw_trial["misalignments"]))
        inital_magnets = torch.tensor(initial_magnets_from_dictionary(raw_trial["initial"]))

        converted_trial = Trial(
            target_beam, incoming_beam, misalignments, inital_magnets
        )
        converted.append(converted_trial)

    return converted


def target_beam_from_dictionary(raw: dict) -> np.ndarray:
    """
    Read a dictionary describing a target beam to a correctly arranged `np.ndarray`.
    """
    return np.array(
        [
            raw["sigma_x"],
            raw["sigma_y"],
            raw["mu_x"],
            raw["mu_y"],
            raw["mu_x"],
            raw["mu_y"],
            raw["sigma_x"],
            raw["sigma_y"],
        ]
    )


def incoming_beam_from_dictionary(raw: dict) -> np.ndarray:
    """
    Read a dictionary describing an incoming beam to a correctly arranged `np.ndarray`.
    """
    return np.array(
        [
            raw["energy"],
            raw["mu_x"],
            raw["mu_px"],
            raw["mu_y"],
            raw["mu_py"],
            raw["sigma_x"],
            raw["sigma_px"],
            raw["sigma_y"],
            raw["sigma_py"],
            raw["sigma_tau"],
            raw["sigma_p"],
        ]
    ).astype(np.float32)


def misalignments_from_dictionary(raw: dict) -> np.ndarray:
    """
    Read a dictionary describing misalignments to a correctly arranged `np.ndarray`.
    """
    return np.array(
        [
            raw["S1_x"],
            raw["S1_y"],
            raw["S2_x"],
            raw["S2_y"],
            raw["S3_x"],
            raw["S3_y"],
            raw["Q1_x"],
            raw["Q1_y"],
            raw["Q2_x"],
            raw["Q2_y"],
            raw["Q3_x"],
            raw["Q3_y"],
            raw["Q4_x"],
            raw["Q4_y"],
            raw["Q5_x"],
            raw["Q5_y"],
            raw["Q6_x"],
            raw["Q6_y"],
            raw["Q7_x"],
            raw["Q7_y"],
            raw["Q8_x"],
            raw["Q8_y"],
            raw["Qj6_x"],
            raw["Qj6_y"],
            raw["Qj7_x"],
            raw["Qj7_y"],
            raw["Qj8_x"],
            raw["Qj8_y"],
        ]
    )


def initial_magnets_from_dictionary(raw: dict) -> np.ndarray:
    """
    Read a dictionary describing initial magnet settings to a correctly arranged
    `np.ndarray`.
    """
    return np.array(
        [
            raw["S1"],
            raw["S2"],
            raw["S3"],
            raw["Q1"],
            raw["Q2"],
            raw["H1"],
            raw["Q3"],
            raw["Q4"],
            raw["H2"],
            raw["Q5"],
            raw["Q6"],
            raw["H3"],
            raw["Q7"],
            raw["Q8"],
            raw["Qj6"],
            raw["Qj7"],
            raw["Qj8"],
        ]
    ).astype(np.float32)