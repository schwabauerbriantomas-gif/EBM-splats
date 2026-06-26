import torch
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB')

# Test PGLF imports
print('\n--- Testing PGLF imports ---')
from pglf import (
    TextEncoder, OmnimodalEncoder,
    HypersphereFlowMatching, FlowMatchingLoss,
    HypersphereContrastiveLoss, ParetoFilter,
    PGLFTrainer,
)
print('All PGLF imports OK')

# Test model instantiation
print('\n--- Testing model creation ---')
dim = 640
device = 'cuda' if torch.cuda.is_available() else 'cpu'

encoder = TextEncoder(dim=dim).to(device)
n_params = sum(p.numel() for p in encoder.parameters())
print(f'TextEncoder: {n_params/1e6:.1f}M params')

flow = HypersphereFlowMatching(dim=dim).to(device)
n_params = sum(p.numel() for p in flow.parameters())
print(f'FlowMatching: {n_params/1e6:.1f}M params')

# Test forward pass
print('\n--- Testing forward pass ---')
B = 4
input_ids = torch.randint(0, 50257, (B, 32)).to(device)
attn_mask = torch.ones(B, 32, dtype=torch.long).to(device)

# Test TextEncoder directly
with torch.no_grad():
    emb = encoder(input_ids, attn_mask)
    print(f'TextEncoder output shape: {emb.shape}')
    print(f'Norms: {emb.norm(dim=-1).tolist()}')

# Test OmnimodalEncoder
omni = OmnimodalEncoder(dim=dim).to(device)
with torch.no_grad():
    emb, mod = omni.encode_text(input_ids, attn_mask)
    print(f'Embedding shape: {emb.shape}')
    print(f'Norms: {emb.norm(dim=-1).tolist()}')
    print(f'Modalities: {mod.tolist()}')

# Test flow matching loss
print('\n--- Testing flow matching ---')
fm_loss = FlowMatchingLoss().to(device)
condition = emb
modality = mod

with torch.no_grad():
    loss = fm_loss(flow, emb, condition, modality)
    print(f'FM Loss: {loss.item():.4f}')

# Test flow sampling
print('\n--- Testing flow sampling ---')
with torch.no_grad():
    samples = flow.sample(condition=condition, modality=modality, n_steps=5)
    print(f'Sampled shape: {samples.shape}')
    print(f'Sampled norms: {samples.norm(dim=-1).tolist()}')

# Test Pareto filter
print('\n--- Testing Pareto filter ---')
pf = ParetoFilter(n_keep=3)
selected, objectives, fronts = pf.filter(emb)
print(f'Pareto: {selected.shape[0]}/{emb.shape[0]} selected')
print(f'Fronts: {[len(f) for f in fronts]}')

# VRAM usage
if torch.cuda.is_available():
    alloc = torch.cuda.memory_allocated(0) / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9
    print(f'\nVRAM: {alloc:.2f} GB allocated, {reserved:.2f} GB reserved')

print('\n=== ALL TESTS PASSED ===')
