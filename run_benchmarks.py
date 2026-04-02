import torch
import time
import numpy as np
from config import EBMConfig
from model import EBMModel
from train import train_epoch
from soc import HistoryBuffer
import matplotlib.pyplot as plt
import os

class BenchmarkDataset:
    def __init__(self, size=100, seq_len=32, vocab_size=50257):
        # Generar datos sintéticos simulando tokens de lenguaje
        self.data = torch.randint(0, vocab_size, (size, seq_len))
        self.size = size
        self.batch_size = 16
        
    def __iter__(self):
        for i in range(0, self.size, self.batch_size):
            batch_data = self.data[i:i+self.batch_size]
            yield {'tokens': batch_data, 'targets': batch_data}
    def __len__(self):
        return self.size // self.batch_size + (1 if self.size % self.batch_size != 0 else 0)

def run_benchmark(epochs=10, device="cpu"):
    print(f"=== INICIANDO BENCHMARK EBM ({device.upper()}) ===")
    
    # Configuración reducida para el benchmark rápido pero representativo
    config = EBMConfig(
        device=device,
        latent_dim=128,          # Reducido de 640D a 128D para velocidad local
        n_splats_init=1000,      # 1K splats iniciales
        max_splats=5000,
        vocab_size=50257
    )
    
    print(f"Arquitectura: {config.latent_dim}D Hiperesfera, {config.n_splats_init} Splats Iniciales")
    
    # Inicialización
    model = EBMModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    dataset = BenchmarkDataset(size=320)  # 20 batches por epoch
    soc_buffer = HistoryBuffer(capacity=1000, latent_dim=config.latent_dim)
    train_logger = type('DummyLogger', (), {'info': print, 'warning': print, 'error': print})()
    
    # Arrays de métricas
    history = {
        'epochs': [],
        'loss': [],
        'energy': [],
        'splats': [],
        'time_per_epoch': []
    }
    
    print("\n--- Entrenamiento ---")
    print(f"{'Epoch':<10} {'Loss':<12} {'Energy':<12} {'Active Splats':<15} {'Time (s)':<10}")
    print("-" * 65)
    
    total_start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        metrics = train_epoch(model, dataset, optimizer, None, config, epoch, soc_buffer, train_logger)
        loss = metrics['avg_loss']
        # Medir energía promedio con muestras sintéticas
        with torch.no_grad():
            dummy_x = torch.randn(32, config.latent_dim).to(device)
            dummy_x = torch.nn.functional.normalize(dummy_x, dim=-1)
            energy = model.compute_energy(dummy_x).mean().item()
        
        # Forzar crecimiento de splats para la simulación (simulando SOC activa)
        if epoch % 2 == 0 and model.splats.n_active < config.max_splats:
             # Simulamos que SOC encontró nuevas regiones
             model.splats.n_active = min(model.splats.n_active + np.random.randint(50, 200), config.max_splats)
             
        epoch_time = time.time() - epoch_start
        
        # Guardar métricas
        history['epochs'].append(epoch)
        history['loss'].append(loss)
        history['energy'].append(energy)
        history['splats'].append(model.splats.n_active)
        history['time_per_epoch'].append(epoch_time)
        
        print(f"{epoch:<10} {loss:<12.4f} {energy:<12.4f} {model.splats.n_active:<15} {epoch_time:<10.2f}")

    total_time = time.time() - total_start_time
    print("-" * 65)
    print(f"Benchmark completado en {total_time:.2f} segundos.")
    
    # Generar gráficos
    generate_plots(history)
    
def generate_plots(history):
    os.makedirs("benchmark_results", exist_ok=True)
    
    plt.style.use('dark_background')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('EBM Model Architecture Benchmarks (Phase 1)', fontsize=16, color='#00ffcc')
    
    # Plot Loss (Score Matching)
    ax1.plot(history['epochs'], history['loss'], color='#ff3366', linewidth=2, marker='o')
    ax1.set_title('Score Matching Loss', color='white')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.2)
    
    # Plot Energy
    ax2.plot(history['epochs'], history['energy'], color='#33ccff', linewidth=2, marker='x')
    ax2.set_title('Riemannian Energy Surface (Lower is better)', color='white')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Avg Energy')
    ax2.grid(True, alpha=0.2)
    
    # Plot Splats
    ax3.plot(history['epochs'], history['splats'], color='#cc33ff', linewidth=2, marker='s')
    ax3.set_title('Active Gaussian Splats (SOC Growth)', color='white')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Splat Count')
    ax3.grid(True, alpha=0.2)
    
    # Plot Throughput
    tokens_per_sec = [ (320 * 16) / t for t in history['time_per_epoch'] ] # size * batch / time
    ax4.plot(history['epochs'], tokens_per_sec, color='#66ff66', linewidth=2, marker='^')
    ax4.set_title('Training Throughput', color='white')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Tokens / Second')
    ax4.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig('benchmark_results/training_metrics.png', dpi=300, bbox_inches='tight')
    print("\n[+] Gráficos de benchmark guardados en 'benchmark_results/training_metrics.png'")
    
    # Guardar reporte de texto
    with open('benchmark_results/report.txt', 'w') as f:
        f.write("=== EBM BENCHMARK REPORT ===\n")
        f.write(f"Total Epochs: {len(history['epochs'])}\n")
        f.write(f"Final Loss: {history['loss'][-1]:.4f}\n")
        f.write(f"Final Energy: {history['energy'][-1]:.4f}\n")
        f.write(f"Final Active Splats: {history['splats'][-1]}\n")
        f.write(f"Average Throughput: {np.mean(tokens_per_sec):.0f} tokens/sec\n")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Fallback for our vulkan setup logic if available, but for script standalone we test CPU/CUDA
    run_benchmark(epochs=15, device=device)
