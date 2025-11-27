import torch
import psutil
import os

class GPUMonitor:
    """Monitor GPU memory usage and system resources"""
    @staticmethod
    def get_gpu_memory_usage():
        """Return GPU memory usage in MB for all devices"""
        if torch.cuda.is_available():
            return {
                f'GPU {i}': f"{torch.cuda.memory_allocated(i)/1024**2:.2f} MB / {torch.cuda.memory_reserved(i)/1024**2:.2f} MB"
                for i in range(torch.cuda.device_count())
            }
        return {"Status": "No GPU available"}

    @staticmethod
    def get_system_memory_usage():
        """Return system RAM usage"""
        mem = psutil.virtual_memory()
        return f"RAM: {mem.used/1024**2:.2f} MB / {mem.total/1024**2:.2f} MB ({mem.percent}%)"

