import torch
import torch.nn as nn
import torch.nn.functional as F

class SplitComplexModReLU(nn.Module):
    """Split-complex modReLU activation function preserving phase while scaling magnitude."""
    def __init__(self, threshold: float = 0.1):
        super().__init__()
        self.b = nn.Parameter(torch.tensor(threshold))

    def forward(self, z: torch.complex64) -> torch.complex64:
        mag = torch.abs(z)
        activation = F.relu(mag + self.b)
        phase = torch.angle(z)
        return activation * torch.exp(1j * phase)

class IotaGatewayAttentionLayer(nn.Module):
    """Complex-Valued Iota-Gateway Attention Layer operating in Hilbert Space C^n."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.W_real = nn.Parameter(torch.randn(dim, dim) / (dim ** 0.5))
        self.W_imag = nn.Parameter(torch.randn(dim, dim) / (dim ** 0.5))
        self.mod_relu = SplitComplexModReLU(threshold=0.1)

    def forward(self, x_complex: torch.complex64, intent_phase: torch.Tensor) -> tuple[torch.complex64, torch.Tensor]:
        # Form complex weight matrix W = A + iB with Hermitian-adjusted coupling
        W_c = torch.complex(self.W_real, -self.W_imag)
        
        # Complex matrix multiplication
        z = torch.matmul(x_complex, W_c.t())
        
        # Apply complex activation
        z_out = self.mod_relu(z)
        
        # Calculate Phase Synchronization Index (sigma) and Spectral Alignment Friction (Xi)
        agent_phase = torch.angle(z_out)
        phase_diff = intent_phase - agent_phase
        sigma = torch.mean(torch.abs(torch.mean(torch.exp(1j * phase_diff), dim=-1)))
        xi = 1.0 - sigma
        
        return z_out, xi

# --- Execution Test ---
if __name__ == "__main__":
    dim = 64
    batch_size = 4

    # Generate synthetic complex inputs and intent phases
    x_complex = torch.randn(batch_size, dim, dtype=torch.complex64)
    intent_phase = torch.randn(batch_size, dim)

    # Initialize layer
    layer = IotaGatewayAttentionLayer(dim=dim)

    # Run forward pass
    z_out, xi = layer(x_complex, intent_phase)

    print("Output shape (z_out):", z_out.shape)
    print("Spectral Alignment Friction (Xi):", xi.item())
