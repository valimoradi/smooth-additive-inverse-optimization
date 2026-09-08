"""
data/test_data.py
=================
Generate small synthetic patient data that matches the PortPy structure.
Used to test the full pipeline without loading real patients (no RAM needed).

Each test patient has:
  - A small random sparse D matrix (~500 voxels, ~50 beamlets)
  - Organ voxel indices for PTV + 5 OARs (~30 voxels each)
  - Same dict structure as load_patient() returns

The forward problem is solvable on these, so the full inverse pipeline
can be verified end-to-end.
"""

import numpy as np
import scipy.sparse as sp


def generate_test_patients(
    n_patients: int = 5,
    n_voxels: int = 500,
    n_beamlets: int = 50,
    n_ptv: int = 80,
    n_oar: int = 30,
    seed: int = 42,
) -> list[dict]:
    """
    Generate n_patients synthetic patient dicts matching PortPy structure.

    The D matrix is sparse random with structure that makes the forward
    problem feasible: PTV voxels get higher dose coefficients.
    """
    rng = np.random.default_rng(seed)
    patients = []

    for i in range(n_patients):
        # Random sparse D matrix
        # PTV voxels get stronger beamlet influence
        density = 0.3
        D_dense = rng.exponential(scale=0.5, size=(n_voxels, n_beamlets))
        mask = rng.random((n_voxels, n_beamlets)) > density
        D_dense[mask] = 0.0

        # Assign non-overlapping voxel regions
        all_idx = rng.permutation(n_voxels)
        offset = 0
        idx_ptv = np.sort(all_idx[offset:offset + n_ptv]); offset += n_ptv
        idx_bladder = np.sort(all_idx[offset:offset + n_oar]); offset += n_oar
        idx_rectum = np.sort(all_idx[offset:offset + n_oar]); offset += n_oar
        idx_femur_l = np.sort(all_idx[offset:offset + n_oar]); offset += n_oar
        idx_femur_r = np.sort(all_idx[offset:offset + n_oar]); offset += n_oar
        idx_skin = np.sort(all_idx[offset:offset + n_oar]); offset += n_oar

        # Boost PTV rows so the prescribed dose is reachable
        D_dense[idx_ptv, :] *= 5.0

        # Add some dose to OAR rows (they're near PTV, get scatter)
        for oar_idx in [idx_bladder, idx_rectum, idx_femur_l, idx_femur_r, idx_skin]:
            D_dense[oar_idx, :] *= (1.0 + rng.uniform(0.5, 2.0))

        D_sparse = sp.csr_matrix(D_dense)

        organ_indices = {
            'ptv': idx_ptv,
            'bladder': idx_bladder,
            'rectum': idx_rectum,
            'femur_l': idx_femur_l,
            'femur_r': idx_femur_r,
            'skin': idx_skin,
        }

        patients.append({
            'patient_id': f'Test_Patient_{i+1}',
            'D': D_sparse,
            'organ_indices': organ_indices,
        })

    return patients
