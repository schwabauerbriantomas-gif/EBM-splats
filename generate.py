import torch
import torch.nn.functional as F

def generate(model, tokenizer, prompt: str, max_tokens: int = 100, temperature: float = 1.0):
    model.eval()
    
    # Tokenize input context
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    
    # Fast path: context tracks history natively. For generation we just
    # accumulate state dynamically.
    
    for _ in range(max_tokens):
        # Obtain current context representation
        active_device = "cpu" if model.config.device == "vulkan" else model.config.device
        input_ids = torch.tensor([tokens], device=active_device)
        x_ctx = model.embed(input_ids)
        x_ctx = x_ctx.mean(dim=1)
        x_ctx = F.normalize(x_ctx, dim=-1)  # Context point on sphere
        
        # Sample new latent state from context via Langevin Dynamics
        x_new = model.sample(n_samples=1, context=x_ctx)[0].unsqueeze(0)
        
        # Decode state back into vocab dimensional representations
        logits = model.decode(x_new, context=x_ctx)
        logits = logits / temperature
        
        # Sample next token
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()
        
        tokens.append(next_token)
        
        # Note: Depending on the tokenizer, there may be a defined eos_token_id.
        if hasattr(tokenizer, 'eos_token_id') and next_token == tokenizer.eos_token_id:
            break
            
    return tokenizer.decode(tokens)
