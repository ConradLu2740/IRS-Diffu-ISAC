import torch
import math


class PhaseOptimizer:
    def __init__(self, H_dict, device="cuda"):
        self.H_dict = H_dict
        self.device = device

    @torch.no_grad()
    def optimize(self, ROI_voxel, X, n_irs_elements=32, n_iter=10, lr=0.1):
        c64 = torch.complex64
        H_ROI_UE = self.H_dict["H_ROI_UE"]
        H_IRS1_UE = self.H_dict["H_IRS1_UE"]
        H_IRS2_UE = self.H_dict["H_IRS2_UE"]
        H_ROI_IRS1 = self.H_dict["H_ROI_IRS1"]
        H_ROI_IRS2 = self.H_dict["H_ROI_IRS2"]
        H_IRS1_ROI = self.H_dict["H_IRS1_ROI"]
        H_IRS2_ROI = self.H_dict["H_IRS2_ROI"]
        H_BS_ROI = self.H_dict["H_BS_ROI"]
        H_BS_IRS1 = self.H_dict["H_BS_IRS1"]
        H_BS_IRS2 = self.H_dict["H_BS_IRS2"]

        S = ROI_voxel.reshape(-1).to(self.device)
        S_c = torch.complex(S.float(), torch.zeros_like(S.float()))

        IRS_half = n_irs_elements // 2

        phase = torch.zeros(n_irs_elements, device=self.device)
        phase_best = phase.clone()
        best_power = -float("inf")

        for _ in range(n_iter):
            for i in range(n_irs_elements):
                phase_i = phase.clone()
                phase_i[i] = phase_i[i] + 0.1
                p1 = self._compute_power(S_c, phase_i, IRS_half, X, H_ROI_UE, H_IRS1_UE,
                                         H_IRS2_UE, H_ROI_IRS1, H_ROI_IRS2,
                                         H_IRS1_ROI, H_IRS2_ROI, H_BS_ROI,
                                         H_BS_IRS1, H_BS_IRS2)

                phase_i[i] = phase_i[i] - 0.2
                p2 = self._compute_power(S_c, phase_i, IRS_half, X, H_ROI_UE, H_IRS1_UE,
                                         H_IRS2_UE, H_ROI_IRS1, H_ROI_IRS2,
                                         H_IRS1_ROI, H_IRS2_ROI, H_BS_ROI,
                                         H_BS_IRS1, H_BS_IRS2)

                if p1 > p2:
                    phase[i] = phase[i] + 0.1
                elif p2 > best_power and p2 > p1:
                    phase[i] = phase[i] - 0.1

                if max(p1, p2) > best_power:
                    best_power = max(p1, p2)
                    phase_best = phase.clone()

        return phase_best

    def _compute_power(self, S_c, phase, IRS_half, X, H_ROI_UE, H_IRS1_UE,
                       H_IRS2_UE, H_ROI_IRS1, H_ROI_IRS2,
                       H_IRS1_ROI, H_IRS2_ROI, H_BS_ROI,
                       H_BS_IRS1, H_BS_IRS2):
        v1 = torch.exp(1j * phase[:IRS_half]).to(torch.complex64)
        v2 = torch.exp(1j * phase[IRS_half:]).to(torch.complex64)

        Bmat = S_c[:, None] * H_ROI_UE
        H_BS_ROI_UE = H_BS_ROI.matmul(Bmat)

        C = H_BS_ROI * S_c[None, :]
        C1 = C.matmul(H_ROI_IRS1)
        C2 = C.matmul(H_ROI_IRS2)
        H_BS_ROI_IRS1_UE = (C1 * v1[None, :]).matmul(H_IRS1_UE)
        H_BS_ROI_IRS2_UE = (C2 * v2[None, :]).matmul(H_IRS2_UE)

        IR1 = H_IRS1_ROI * S_c[None, :]
        IR2 = H_IRS2_ROI * S_c[None, :]
        B1 = IR1.matmul(H_ROI_UE)
        B2 = IR2.matmul(H_ROI_UE)
        H_BS_IRS1_ROI_UE = (H_BS_IRS1 * v1[None, :]).matmul(B1)
        H_BS_IRS2_ROI_UE = (H_BS_IRS2 * v2[None, :]).matmul(B2)

        H_total = H_BS_ROI_UE + H_BS_ROI_IRS1_UE + H_BS_ROI_IRS2_UE + H_BS_IRS1_ROI_UE + H_BS_IRS2_ROI_UE
        Y = H_total.matmul(X)
        return torch.sum(torch.abs(Y) ** 2).item()


@torch.no_grad()
def compute_received_signal_irs(ROI_voxel, phase, X, H_dict, power_sigma, device="cuda"):
    H_ROI_UE = H_dict["H_ROI_UE"]
    H_IRS1_UE = H_dict["H_IRS1_UE"]
    H_IRS2_UE = H_dict["H_IRS2_UE"]
    H_ROI_IRS1 = H_dict["H_ROI_IRS1"]
    H_ROI_IRS2 = H_dict["H_ROI_IRS2"]
    H_IRS1_ROI = H_dict["H_IRS1_ROI"]
    H_IRS2_ROI = H_dict["H_IRS2_ROI"]
    H_BS_ROI = H_dict["H_BS_ROI"]
    H_BS_IRS1 = H_dict["H_BS_IRS1"]
    H_BS_IRS2 = H_dict["H_BS_IRS2"]

    IRS_half = phase.shape[0] // 2
    S = ROI_voxel.reshape(-1).to(device)
    S_c = torch.complex(S.float(), torch.zeros_like(S.float()))

    v1 = torch.exp(1j * phase[:IRS_half]).to(torch.complex64)
    v2 = torch.exp(1j * phase[IRS_half:]).to(torch.complex64)

    Bmat = S_c[:, None] * H_ROI_UE
    H_BS_ROI_UE = H_BS_ROI.matmul(Bmat)

    C = H_BS_ROI * S_c[None, :]
    C1 = C.matmul(H_ROI_IRS1)
    C2 = C.matmul(H_ROI_IRS2)
    H_BS_ROI_IRS1_UE = (C1 * v1[None, :]).matmul(H_IRS1_UE)
    H_BS_ROI_IRS2_UE = (C2 * v2[None, :]).matmul(H_IRS2_UE)

    IR1 = H_IRS1_ROI * S_c[None, :]
    IR2 = H_IRS2_ROI * S_c[None, :]
    B1 = IR1.matmul(H_ROI_UE)
    B2 = IR2.matmul(H_ROI_UE)
    H_BS_IRS1_ROI_UE = (H_BS_IRS1 * v1[None, :]).matmul(B1)
    H_BS_IRS2_ROI_UE = (H_BS_IRS2 * v2[None, :]).matmul(B2)

    H_total = H_BS_ROI_UE + H_BS_ROI_IRS1_UE + H_BS_ROI_IRS2_UE + H_BS_IRS1_ROI_UE + H_BS_IRS2_ROI_UE
    Y = H_total.matmul(X)

    std = math.sqrt(power_sigma)
    noise = torch.complex(torch.randn_like(Y.real) * std, torch.randn_like(Y.imag) * std)
    return Y + noise