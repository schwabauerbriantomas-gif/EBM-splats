import torch
import math
from geometry import exp_map, log_map, project_to_tangent, geodesic_distance, normalize_sphere

def test_geometry_l2_normalization():
    # Setup base vectors
    p = normalize_sphere(torch.randn(10, 640))
    v_raw = torch.randn(10, 640)
    
    # Project v to tangent space at p
    v_tangent = project_to_tangent(p, v_raw)
    
    # 1. Assert orthogonality in tangent space
    inner_prod = (p * v_tangent).sum(dim=-1)
    assert torch.allclose(inner_prod, torch.zeros_like(inner_prod), atol=1e-6), "Tangent projection failed orthogonality."

    # 2. Assert exp_map stays on the sphere exactly (L2 norm = 1)
    mapped_sphere = exp_map(p, v_tangent)
    norms = torch.norm(mapped_sphere, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6), "Exp map drifted off the hypersphere."

    # 3. Assert distance is angular
    dist = geodesic_distance(p, mapped_sphere)
    assert torch.all(dist >= 0) and torch.all(dist <= math.pi), "Geodesic distances bounded poorly."
    print("All geometry assertions passed.")

if __name__ == "__main__":
    test_geometry_l2_normalization()
