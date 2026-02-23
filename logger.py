import time
import json
from datetime import datetime


class TrainingLogger:
    def __init__(self, output_dir='logs'):
        self.output_dir = output_dir
        self.current_file = None
        self.log_data = {
            'start_time': time.time(),
            'epochs': [],
            'current_epoch': {}
        }
    
    def start_epoch(self, epoch):
        epoch_start = time.time()
        self.log_data['current_epoch'] = {
            'epoch': epoch,
            'start_time': epoch_start,
            'batches': []
        }
        self.info(f"Epoch {epoch} started")
    
    def log_batch(self, batch_idx, loss, energy, splat_stats, learning_rate):
        batch_info = {
            'batch': batch_idx,
            'loss': loss,
            'energy': energy,
            'splat_stats': splat_stats,
            'timestamp': time.time() - self.log_data['start_time'],
            'learning_rate': learning_rate
        }
        self.log_data['current_epoch']['batches'].append(batch_info)
        
        # Detailed logging every 10 batches
        if batch_idx % 10 == 0:
            self.debug(f"Batch {batch_idx}: Loss={loss:.4f}, Energy={energy:.4f}, Splats={splat_stats}")
    
    def end_epoch(self, epoch, model, optimizer, scheduler, metrics):
        epoch_end = time.time()
        epoch_duration = epoch_end - self.log_data['current_epoch']['start_time']
        
        epoch_summary = {
            'epoch': epoch,
            'duration_seconds': epoch_duration,
            'metrics': metrics,
            'num_batches': len(self.log_data['current_epoch']['batches']),
            'avg_loss': sum(b['loss'] for b in self.log_data['current_epoch']['batches']) / len(self.log_data['current_epoch']['batches']),
            'avg_energy': sum(b['energy'] for b in self.log_data['current_epoch']['batches']) / len(self.log_data['current_epoch']['batches']),
            'splat_stats': self._aggregate_splat_stats(epoch),
        }
        
        self.log_data['epochs'].append(epoch_summary)
        self.log_data['current_epoch'] = {}
        
        self.info(f"Epoch {epoch} completed in {epoch_duration:.2f}s: {metrics}")
    
    def _aggregate_splat_stats(self, epoch):
        all_stats = [b['splats_stats'] for b in self.log_data['current_epoch']['batches']]
        if not all_stats:
            return {}
        
        # Aggregate splat statistics
        return {
            'total_splats': all_stats[-1]['n_active'],
            'avg_frequency': all_stats[-1]['mean_frequency'],
            'avg_kappa': all_stats[-1]['mean_kappa'],
            'splats_added': all_stats[-1]['splats_added'] if 'splats_added' in all_stats[-1] else 0
        }
    
    def info(self, message):
        print(f"[INFO] {message}")
    
    def warning(self, message):
        print(f"[WARNING] {message}")
    
    def error(self, message):
        print(f"[ERROR] {message}")
    
    def debug(self, message):
        print(f"[DEBUG] {message}")
    
    def save_log(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = f"{self.output_dir}/training_log_{timestamp}.json"
        
        with open(log_filename, 'w') as f:
            json.dump(self.log_data, f, indent=2)
        
        self.info(f"Training log saved to {log_filename}")
    
    def get_metrics_summary(self):
        if not self.log_data['epochs']:
            return {}
        
        all_losses = [e['metrics']['avg_loss'] for e in self.log_data['epochs']]
        all_energies = [e['metrics']['avg_energy'] for e in self.log_data['epochs']]
        
        return {
            'total_epochs': len(self.log_data['epochs']),
            'avg_loss': sum(all_losses) / len(all_losses),
            'best_loss': min(all_losses),
            'worst_loss': max(all_losses),
            'avg_energy': sum(all_energies) / len(all_energies),
            'best_energy': min(all_energies),
            'worst_energy': max(all_energies),
            'total_time': time.time() - self.log_data['start_time']
        }
