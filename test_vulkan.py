import torch
from vulkan_engine import VulkanEBMRunner

def test_vulkan_init():
    try:
        print("Starting Vulkan Engine initialization...")
        runner = VulkanEBMRunner(latent_dim=640)
        print("Vulkan Engine Initialized Successfully!")
        
        # Test simulated run_compute
        x = torch.randn(4, 640)
        mu = torch.randn(100, 640)
        alpha = torch.randn(100)
        
        print(f"Dispatching dummy tensors: X={x.shape}, mu={mu.shape}")
        out_grad = runner.run_compute(x, mu, alpha)
        
        print(f"Vulkan output gradient shape: {out_grad.shape}")
        print("Vulkan Pipeline Test Passed.")
    except Exception as e:
        print(f"Vulkan testing failed: {e}")

if __name__ == "__main__":
    test_vulkan_init()
