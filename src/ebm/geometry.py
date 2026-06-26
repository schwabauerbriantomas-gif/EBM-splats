import torch
import torch.nn.functional as F

def exp_map(p: torch.Tensor, v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Projects tangent vector 'v' back to the hypersphere at base point 'p'."""
    norm_v = torch.norm(v, dim=-1, keepdim=True).clamp(min=eps)
    return torch.cos(norm_v) * p + torch.sin(norm_v) * v / norm_v

def log_map(p: torch.Tensor, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Projects point 'x' on the sphere to the tangent space at 'p'."""
    cos_theta = (p * x).sum(dim=-1, keepdim=True).clamp(-1 + eps, 1 - eps)
    theta = torch.acos(cos_theta)
    sin_theta = torch.sin(theta).clamp(min=eps)
    return theta / sin_theta * (x - cos_theta * p)

def project_to_tangent(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Projects arbitrary vector 'v' onto the tangent space of sphere at 'x'."""
    return v - (v * x).sum(dim=-1, keepdim=True) * x

def geodesic_distance(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Calculates angular distance between x and y on the hypersphere."""
    cos_theta = (x * y).sum(dim=-1).clamp(-1 + eps, 1 - eps)
    return torch.acos(cos_theta)

def normalize_sphere(x: torch.Tensor) -> torch.Tensor:
    """L2 normalizes 'x' onto the hypersphere."""
    return F.normalize(x, dim=-1)
