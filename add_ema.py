import re

c = open('autoresearch_train.py').read()

# 1. Add EMA class before class Embedder
ema_class = '''
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.named_parameters() if v.requires_grad}
    def update(self, model):
        for k, v in model.named_parameters():
            if v.requires_grad:
                self.shadow[k].mul_(self.decay).add_(v.data, alpha=1 - self.decay)
    def apply(self, model):
        backup = {k: v.clone() for k, v in model.named_parameters() if v.requires_grad}
        for k, v in model.named_parameters():
            if v.requires_grad:
                v.data.copy_(self.shadow[k])
        return backup
    def restore(self, model, backup):
        for k, v in model.named_parameters():
            if v.requires_grad:
                v.data.copy_(backup[k])

'''

c = c.replace('\nclass Embedder', ema_class + 'class Embedder')

# 2. Add ema creation
c = c.replace(
    '    embedder = Embedder(cfg).to(device)\n    n_params',
    '    embedder = Embedder(cfg).to(device)\n    ema = EMA(score_net, decay=0.999)\n    n_params'
)

# 3. Add ema.update after both optimizer.step() calls
c = c.replace(
    '                scaler.step(optimizer)\n                scaler.update()',
    '                scaler.step(optimizer)\n                scaler.update()\n                ema.update(score_net)'
)

c = c.replace(
    '                optimizer.step()\n            epoch_loss',
    '                optimizer.step()\n                ema.update(score_net)\n            epoch_loss'
)

# 4. Apply EMA before evaluate, restore after
# Replace the evaluation block
old_eval = '''            val_loss = evaluate_loss(score_net, embedder, val_loader, device, cfg.noise_levels)
            print(f"E{epoch+1} | val_loss: {val_loss:.4f}")
            if val_loss < best_val_loss: best_val_loss = val_loss'''

new_eval = '''            backup = ema.apply(score_net)
            val_loss = evaluate_loss(score_net, embedder, val_loader, device, cfg.noise_levels)
            ema.restore(score_net, backup)
            print(f"E{epoch+1} | val_loss: {val_loss:.4f}")
            if val_loss < best_val_loss: best_val_loss = val_loss'''

c = c.replace(old_eval, new_eval)

# Also the final eval
c = c.replace(
    '    val_loss = evaluate_loss(score_net, embedder, val_loader, device, cfg.noise_levels)\n    print(f"---',
    '    backup = ema.apply(score_net)\n    val_loss = evaluate_loss(score_net, embedder, val_loader, device, cfg.noise_levels)\n    ema.restore(score_net, backup)\n    print(f"---'
)

open('autoresearch_train.py', 'w').write(c)
print('OK')
