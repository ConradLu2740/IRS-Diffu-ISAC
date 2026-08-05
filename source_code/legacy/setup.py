# setup.py
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

Tau = 8
Time_slot = 1
P_SNR = 20
Power_sigma = 0.01

ROI_Length = 16
ROI_Number = ROI_Length ** 3
start = np.array([0.1, 0.1, 0.1])
end   = np.array([1.6, 1.6, 1.6])

pBS_0  = np.array([-4, 0.5, 2])
pROI_0 = np.array([0.5, 0.5, 0.5])
pUE_0  = np.array([2, 0.5, 0])

pUE = np.array([[2, 1.25, 0], [2, -0.25, 0], [2, 0.25, 0], [2, 0.75, 0]])
pBS = np.array([[-4, 0.5, 2], [-4, 0.55, 2], [-4, 0.6, 2], [-4, 0.65, 2]])

UE_Number, _ = pUE.shape
BS_Number, _ = pBS.shape

N_IRS = 4
pIRS_Interval = 0.1
pIRS_P1 = np.array([1.0, 2.0, 1.0])
pIRS_P2 = np.array([-1.0, -1.0, 1.0])

IRS_Number = N_IRS * N_IRS
IRS_total  = 2 * IRS_Number

def Power_SNR(sigma, snr):
    Power = 10 ** (snr / 10) * sigma * 64
    return Power

def get_Channel(a, b):
    if a.ndim >= 2:
        distance = np.linalg.norm(a - b, axis=-1)
    else:
        distance = np.linalg.norm(a - b)
    phase = 2 * np.pi * distance / 0.01
    H = np.sqrt(0.1) * np.exp(1j * phase) / distance
    return H

def make_irs_rotated():
    pIRS_1 = np.array([[pIRS_Interval * m, 0.0, pIRS_Interval * n] for m in range(N_IRS) for n in range(N_IRS)]) + pIRS_P1
    pIRS_2 = np.array([[pIRS_Interval * m, 0.0, pIRS_Interval * n] for m in range(N_IRS) for n in range(N_IRS)]) + pIRS_P2

    line_vector = np.array([1.0, 1.0, 1.0])
    plane_normal = np.array([0.0, 1.0, 0.0])
    axis = np.cross(plane_normal, line_vector)
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(np.dot(plane_normal, line_vector) / (np.linalg.norm(plane_normal) * np.linalg.norm(line_vector)))
    rotation = R.from_rotvec(axis * angle)
    pIRS_1_rotated = rotation.apply(pIRS_1 - pIRS_P1) + pIRS_P1

    line_vector_new = np.array([1.0, 1.0, -1.0])
    plane_normal_2 = np.array([0.0, 1.0, 0.0])
    axis_2 = np.cross(plane_normal_2, line_vector_new)
    axis_2 = axis_2 / np.linalg.norm(axis_2)
    angle_2 = np.arccos(np.dot(plane_normal_2, line_vector_new) / (np.linalg.norm(plane_normal_2) * np.linalg.norm(line_vector_new)))
    rotation_2 = R.from_rotvec(axis_2 * angle_2)
    pIRS_2_rotated = rotation_2.apply(pIRS_2 - pIRS_P2) + pIRS_P2

    return pIRS_1_rotated, pIRS_2_rotated

def make_roi_grid():
    x = np.linspace(start[0], end[0], ROI_Length)
    y = np.linspace(start[1], end[1], ROI_Length)
    z = np.linspace(start[2], end[2], ROI_Length)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="xy")
    pROI = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
    return pROI

def precompute_channels(device):
    pIRS_1_rotated, pIRS_2_rotated = make_irs_rotated()
    pROI = make_roi_grid()

    H0_1 = get_Channel(pBS_0, pROI_0)
    H0_2 = get_Channel(pROI_0, pUE_0)
    H2 = np.abs((H0_1 * H0_2)) ** 2
    Power = Power_SNR(Power_sigma, P_SNR)
    a = np.sqrt(10 ** (P_SNR / 10) * Power_sigma / H2 / 10)
    tensor_a = torch.tensor(a).to(device)

    UE_3 = pUE[np.newaxis, :, :]
    BS_2 = pBS[:, np.newaxis, :]
    ROI_2 = pROI[:, np.newaxis, :]
    ROI_3 = pROI[np.newaxis, :, :]

    IRS1_2 = pIRS_1_rotated[:, np.newaxis, :]
    IRS1_3 = pIRS_1_rotated[np.newaxis, :, :]
    IRS2_2 = pIRS_2_rotated[:, np.newaxis, :]
    IRS2_3 = pIRS_2_rotated[np.newaxis, :, :]

    H_ROI_UE = get_Channel(UE_3, ROI_2)
    H_IRS1_UE = get_Channel(UE_3, IRS1_2)
    H_IRS2_UE = get_Channel(UE_3, IRS2_2)
    H_ROI_IRS1 = get_Channel(IRS1_3, ROI_2)
    H_ROI_IRS2 = get_Channel(IRS2_3, ROI_2)
    H_IRS1_ROI = get_Channel(ROI_3, IRS1_2)
    H_IRS2_ROI = get_Channel(ROI_3, IRS2_2)
    H_BS_ROI = get_Channel(ROI_3, BS_2)
    H_ROI_BS = get_Channel(BS_2, ROI_3)
    H_BS_IRS1 = get_Channel(IRS1_3, BS_2)
    H_BS_IRS2 = get_Channel(IRS2_3, BS_2)

    def to_c64(x): return torch.from_numpy(x.astype(np.complex64)).to(device)

    H_dict = {
        "tensor_a": tensor_a,
        "H_ROI_UE": to_c64(H_ROI_UE),
        "H_IRS1_UE": to_c64(H_IRS1_UE),
        "H_IRS2_UE": to_c64(H_IRS2_UE),
        "H_ROI_IRS1": to_c64(H_ROI_IRS1),
        "H_ROI_IRS2": to_c64(H_ROI_IRS2),
        "H_IRS1_ROI": to_c64(H_IRS1_ROI),
        "H_IRS2_ROI": to_c64(H_IRS2_ROI),
        "H_BS_ROI": to_c64(H_BS_ROI),
        "H_ROI_BS": to_c64(H_ROI_BS),
        "H_BS_IRS1": to_c64(H_BS_IRS1),
        "H_BS_IRS2": to_c64(H_BS_IRS2),
    }
    return H_dict