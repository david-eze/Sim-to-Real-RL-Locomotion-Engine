import os
import logging
from typing import Dict, Any

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RL_Robotics")


class MetricsLogger:
    def __init__(self, log_dir: str = "./logs/tb_logs", use_tensorboard: bool = True):
        self.log_dir = log_dir
        self.use_tb = use_tensorboard and TENSORBOARD_AVAILABLE
        self.writer = None
        
        if self.use_tb:
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)
            logger.info(f"[MetricsLogger] TensorBoard active. Logging to: {log_dir}")
        else:
            logger.info("[MetricsLogger] Console logging active (TensorBoard unavailable/disabled).")

    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int) -> None:
        if self.writer:
            for tag, value in tag_scalar_dict.items():
                self.writer.add_scalar(f"{main_tag}/{tag}", value, step)
                
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in tag_scalar_dict.items()])
        logger.info(f"[Step {step:7d}] [{main_tag}] {metrics_str}")

    def close(self) -> None:
        if self.writer:
            self.writer.flush()
            self.writer.close()
