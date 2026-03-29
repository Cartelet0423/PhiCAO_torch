import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def fringe_to_nm(j):
    if j < 1:
        raise ValueError("Fringe Zernike index j must be >= 1")
    d = math.ceil(math.sqrt(j)) - 1
    j_start = d**2 + 1
    k = j - j_start
    q = k // 2
    n = d + q
    m_abs = d - q
    m = 0 if m_abs == 0 else (m_abs if k % 2 == 0 else -m_abs)
    return int(n), int(m)

def get_zernike_basis(j, rho, theta):
    n, m = fringe_to_nm(j)
    R = torch.zeros_like(rho)
    m_abs = abs(m)
    for k in range((n - m_abs) // 2 + 1):
        num = ((-1)**k * math.factorial(n - k))
        den = (math.factorial(k) * math.factorial((n + m_abs) // 2 - k) * math.factorial((n - m_abs) // 2 - k))
        R += (num / den) * rho**(n - 2 * k)
    
    mask = rho <= 1.0
    if m > 0:
        Z = R * torch.cos(m_abs * theta)
    elif m < 0:
        Z = R * torch.sin(m_abs * theta)
    else:
        Z = R
    return Z * mask

class DifferentiablePhiCAO(nn.Module):
    def __init__(self, image_stack, dz, dy, dx, NA, lambda_em, n_imm, target_modes, w_meas=0.22, gamma=10, device='cuda'):
        super().__init__()
        self.device = device
        self.Nz, self.Ny, self.Nx = image_stack.shape
        self.lambda_em = lambda_em
        self.k0 = 2 * math.pi / lambda_em
        self.w_meas = w_meas
        self.gamma = gamma
        
        u = torch.fft.fftfreq(self.Nx, d=dx, device=device)
        v = torch.fft.fftfreq(self.Ny, d=dy, device=device)
        U, V = torch.meshgrid(u, v, indexing='xy')
        
        rho_freq = torch.sqrt(U**2 + V**2)
        rho_norm = rho_freq * lambda_em / NA
        theta = torch.atan2(V, U)
        
        self.P = (rho_norm <= 1.0).to(torch.complex64)
        self.Wd_base = torch.sqrt(torch.clamp(n_imm**2 - (lambda_em * rho_freq)**2, min=0.0))
        self.z_arr = torch.fft.fftfreq(self.Nz, d=1.0/self.Nz, device=device) * dz
        
        basis_list = [get_zernike_basis(m, rho_norm, theta) for m in target_modes]
        self.basis = torch.stack(basis_list).to(device)
        self.alphas = nn.Parameter(torch.zeros(len(target_modes), device=device))
        
        img_tensor = torch.tensor(image_stack, dtype=torch.float32, device=device)
        self.D = torch.fft.fftn(torch.fft.fftshift(img_tensor))
        
        kx = torch.fft.fftfreq(self.Nx, d=dx, device=device)
        ky = torch.fft.fftfreq(self.Ny, d=dy, device=device)
        kz = torch.fft.fftfreq(self.Nz, d=dz, device=device)
        Kz, Ky, Kx = torch.meshgrid(kz, ky, kx, indexing='ij')
        
        kl = torch.sqrt(Kx**2 + Ky**2)
        ks = Kz
        chi = 1.8
        theta_angle = math.asin(NA / n_imm)
        
        LHS = 2 * (2 * math.pi / (chi * lambda_em)) * (torch.abs(kl) * math.sin(theta_angle) - torch.abs(ks) * math.cos(theta_angle))
        RHS = torch.sqrt(kl**2 + ks**2)
        self.C_filter = (LHS >= RHS).float()
        
        P_ideal = self.P.unsqueeze(0) * torch.exp(1j * self.k0 * (self.z_arr.view(-1, 1, 1) * self.Wd_base.unsqueeze(0)))
        psf_ideal = torch.abs(torch.fft.ifft2(P_ideal))**2
        psf_ideal /= torch.sum(psf_ideal)
        
        A_ideal = torch.abs(torch.fft.fftn(psf_ideal))
        self.k_mask = A_ideal > (1e-6 * torch.max(A_ideal))

    def forward(self):
        Wa = torch.sum(self.alphas.view(-1, 1, 1) * self.basis, dim=0)
        phase_prime = self.k0 * (Wa.unsqueeze(0) + self.z_arr.view(-1, 1, 1) * self.Wd_base.unsqueeze(0))
        P_prime = self.P.unsqueeze(0) * torch.exp(1j * phase_prime)
        
        psf_3d = torch.abs(torch.fft.ifft2(P_prime))**2
        psf_3d /= torch.sum(psf_3d)
        
        O_prime = torch.fft.fftn(psf_3d)
        phi_prime = torch.angle(O_prime)
        
        denominator = torch.exp(1j * phi_prime) + self.w_meas**2
        D_prime = torch.zeros_like(self.D)
        D_prime[self.k_mask] = self.D[self.k_mask] / denominator[self.k_mask]
        
        d_prime = torch.real(torch.fft.ifftn(D_prime * self.C_filter))
        pos_d = torch.clamp(d_prime, min=0.0)
        neg_d = torch.clamp(d_prime, max=0.0)
        metric = torch.var(pos_d**2) - self.gamma * torch.var(neg_d**2)
        
        return metric, psf_3d, O_prime, Wa

def run_phicao(image_stack, dz, dy, dx, NA, lambda_em, n_imm, target_modes=[11, 12, 13], w_meas=0.22, w_final=0.01, gamma=10.0, epochs=50, lr=0.01):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DifferentiablePhiCAO(image_stack, dz, dy, dx, NA, lambda_em, n_imm, target_modes, w_meas, gamma, device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        metric, _, _, _ = model()
        loss = -metric
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:03d}/{epochs} | Loss: {loss.item():.4e}")

    with torch.no_grad():
        _, psf_final, O_final, Wa = model()
        denominator_final = O_final + w_final**2
        D_double_prime = torch.zeros_like(model.D)
        
        P_ideal = model.P.unsqueeze(0) * torch.exp(1j * model.k0 * (model.z_arr.view(-1, 1, 1) * model.Wd_base.unsqueeze(0)))
        psf_ideal = torch.abs(torch.fft.ifft2(P_ideal))**2
        psf_ideal /= torch.sum(psf_ideal)
        A = torch.abs(torch.fft.fftn(psf_ideal))
        
        D_double_prime[model.k_mask] = A[model.k_mask] * model.D[model.k_mask] / denominator_final[model.k_mask]
        corrected_img = torch.real(torch.fft.ifftn(D_double_prime))
        alphas_dict = {m: a.item() for m, a in zip(target_modes, model.alphas)}
        
    return (
        alphas_dict,
        torch.fft.fftshift(psf_final).cpu().numpy(),
        torch.fft.fftshift(corrected_img).cpu().numpy(),
        Wa.cpu().numpy()
    )

def run_phicao_subregion(image_stack, dz, dy, dx, NA, lambda_em, n_imm, target_modes=[11, 12, 13], w_meas=0.22, w_final=0.01, gamma=10.0, epochs=50, lr=0.01, grid_y=2, grid_x=2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Z, Y, X = image_stack.shape
    step_y = Y // grid_y
    step_x = X // grid_x

    patch_alphas, patch_centers, patch_results = {}, {}, {}

    for i in range(grid_y):
        for j in range(grid_x):
            print(f"Estimating Grid ({i+1}/{grid_y}, {j+1}/{grid_x})")
            y_start = i * step_y
            y_end = Y if i == grid_y - 1 else (i + 1) * step_y
            x_start = j * step_x
            x_end = X if j == grid_x - 1 else (j + 1) * step_x
            
            cy = y_start + (y_end - y_start) / 2.0
            cx = x_start + (x_end - x_start) / 2.0
            patch_centers[(i, j)] = (cy, cx)
            
            patch_stack = image_stack[:, y_start:y_end, x_start:x_end]
            model = DifferentiablePhiCAO(patch_stack, dz, dy, dx, NA, lambda_em, n_imm, target_modes, w_meas, gamma, device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            
            for epoch in range(epochs):
                optimizer.zero_grad()
                metric, _, _, _ = model()
                loss = -metric
                loss.backward()
                optimizer.step()
                
            patch_alphas[(i, j)] = model.alphas.detach().clone()
            with torch.no_grad():
                _, psf_final, O_final, Wa = model()
                patch_results[(i, j)] = {
                    'alphas': {m: a.item() for m, a in zip(target_modes, model.alphas)},
                    'psf_final': torch.fft.fftshift(psf_final).cpu().numpy(),
                    'wavefront': Wa.cpu().numpy(),
                    'bounds': (y_start, y_end, x_start, x_end)
                }

    print("Performing Subregion Deconvolution")
    full_model = DifferentiablePhiCAO(image_stack, dz, dy, dx, NA, lambda_em, n_imm, target_modes, w_meas, gamma, device)
    
    final_corrected = torch.zeros((Z, Y, X), dtype=torch.float32, device=device)
    weight_sum = torch.zeros((Y, X), dtype=torch.float32, device=device)
    
    y_indices = torch.arange(Y, device=device).view(-1, 1)
    x_indices = torch.arange(X, device=device).view(1, -1)

    for i in range(grid_y):
        for j in range(grid_x):
            with torch.no_grad():
                full_model.alphas.copy_(patch_alphas[(i, j)])
                _, _, O_final, _ = full_model()
                
                denominator_final = O_final + w_final**2
                D_double_prime = torch.zeros_like(full_model.D)
                
                P_ideal = full_model.P.unsqueeze(0) * torch.exp(1j * full_model.k0 * (full_model.z_arr.view(-1, 1, 1) * full_model.Wd_base.unsqueeze(0)))
                psf_ideal = torch.abs(torch.fft.ifft2(P_ideal))**2
                psf_ideal /= torch.sum(psf_ideal)
                A_ft = torch.abs(torch.fft.fftn(psf_ideal))
                
                D_double_prime[full_model.k_mask] = A_ft[full_model.k_mask] * full_model.D[full_model.k_mask] / denominator_final[full_model.k_mask]
                corrected_full = torch.fft.fftshift(torch.real(torch.fft.ifftn(D_double_prime)))
                
            cy, cx = patch_centers[(i, j)]
            wy = torch.clamp(1.0 - torch.abs(y_indices - cy) / step_y, min=0.0)
            wx = torch.clamp(1.0 - torch.abs(x_indices - cx) / step_x, min=0.0)
            
            if i == 0: wy[y_indices < cy] = 1.0
            if i == grid_y - 1: wy[y_indices > cy] = 1.0
            if j == 0: wx[x_indices < cx] = 1.0
            if j == grid_x - 1: wx[x_indices > cx] = 1.0
            
            weight_map = (wy * wx).unsqueeze(0)
            final_corrected += corrected_full * weight_map
            weight_sum += weight_map.squeeze(0)

    final_corrected /= (weight_sum.unsqueeze(0) + 1e-8)
    return final_corrected.cpu().numpy(), patch_results