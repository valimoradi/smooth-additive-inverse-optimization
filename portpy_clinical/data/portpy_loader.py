"""
data/portpy_loader.py
=====================
Load real PortPy prostate patient data: D matrices, voxel indices, and
clinical criteria. Uses PortPy's native spatial downsampling to control
problem size while preserving all voxel information via weighted aggregation.
"""

import os
import numpy as np
import scipy.sparse as sp

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gc

from config import (
    PORTPY_DATA_DIR, PROTOCOL_NAME, STRUCTURE_NAMES,
    OAR_KEYS, OPT_VOX_XYZ_RES_MM,
)


def list_available_patients(data_dir: str = PORTPY_DATA_DIR) -> list[str]:
    """Return numerically sorted list of Prostate_Patient_* IDs on disk."""
    patients = []
    for name in os.listdir(data_dir):
        if name.startswith("Prostate_Patient_") and os.path.isdir(os.path.join(data_dir, name)):
            patients.append(name)
    # Sort numerically by patient number
    patients.sort(key=lambda x: int(x.split("_")[-1]))
    return patients


def load_patient(patient_id: str, data_dir: str = PORTPY_DATA_DIR,
                 opt_vox_xyz_res_mm: list = None) -> dict:
    """
    Load a single patient's data via PortPy.

    Parameters
    ----------
    opt_vox_xyz_res_mm : list of 3 floats, optional
        Target optimization voxel resolution [x, y, z] in mm.
        Passed directly to PortPy's InfluenceMatrix for native downsampling.
        None = use native CT resolution.

    Returns dict with:
      'patient_id': str
      'D': dense dose-influence matrix (n_voxels, n_beamlets)
      'organ_indices': dict mapping organ key -> voxel index array
    """
    import portpy.photon as pp

    data = pp.DataExplorer(data_dir=data_dir)
    data.patient_id = patient_id

    ct      = pp.CT(data)
    structs = pp.Structures(data)
    beams   = pp.Beams(data)

    # Load clinical criteria for structure creation
    clinical_criteria = pp.ClinicalCriteria(data, protocol_name=PROTOCOL_NAME)
    opt_params = data.load_config_opt_params(protocol_name=PROTOCOL_NAME)
    structs.create_opt_structures(opt_params=opt_params,
                                  clinical_criteria=clinical_criteria)

    # Load at native resolution first
    inf_matrix = pp.InfluenceMatrix(ct=ct, structs=structs, beams=beams)

    # Downsample via create_down_sample with overwrite=True to avoid deepcopy memory spike.
    # Fix PortPy bug: with overwrite=True, beamlets_dict gets replaced before
    # 'beamlet_idx_2d_finest_grid' is copied. Pre-copy it into _beams to work around.
    if opt_vox_xyz_res_mm is not None:
        for i in range(len(beams.beams_dict['beamlets'])):
            beams.beams_dict['beamlets'][i]['beamlet_idx_2d_finest_grid'] = \
                inf_matrix.beamlets_dict[i]['beamlet_idx_2d_finest_grid']
        inf_matrix = inf_matrix.create_down_sample(
            opt_vox_xyz_res_mm=opt_vox_xyz_res_mm, overwrite=True
        )

    # Extract voxel indices for each structure
    organ_indices = {}
    all_struct_names = structs.structures_dict['name']

    for key, pp_name in STRUCTURE_NAMES.items():
        if pp_name in all_struct_names:
            try:
                idx = inf_matrix.get_opt_voxels_idx(pp_name)
                organ_indices[key] = np.array(idx)
            except Exception:
                organ_indices[key] = np.array([], dtype=int)
        else:
            organ_indices[key] = np.array([], dtype=int)

    # Get D matrix — keep sparse for memory efficiency
    D = inf_matrix.A
    if not sp.issparse(D):
        D = sp.csr_matrix(D)

    return {
        'patient_id': patient_id,
        'D': D,
        'organ_indices': organ_indices,
    }


def load_patients(
    patient_ids: list[str],
    data_dir: str = PORTPY_DATA_DIR,
) -> list[dict]:
    """
    Load multiple patients. Returns list of patient data dicts.
    """
    patients = []
    for i, pid in enumerate(patient_ids):
        print(f"  [{i+1}/{len(patient_ids)}] Loading {pid} ...", end=" ", flush=True)
        try:
            pdata = load_patient(pid, data_dir,
                                 opt_vox_xyz_res_mm=OPT_VOX_XYZ_RES_MM)
            # Report sizes
            n_v, n_b = pdata['D'].shape
            oar_sizes = {k: len(v) for k, v in pdata['organ_indices'].items()}
            print(f"OK  D=({n_v},{n_b})  voxels={oar_sizes}")
            patients.append(pdata)
            gc.collect()
        except Exception as exc:
            print(f"FAILED: {exc}")
    return patients
