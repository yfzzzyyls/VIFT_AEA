"""Dummy metrics calculator for basic training"""
from src.metrics.base_metrics_calculator import BaseMetricsCalculator
import numpy as np

class DummyMetricsCalculator(BaseMetricsCalculator):
    """A dummy metrics calculator that returns basic metrics"""
    
    def __init__(self):
        pass
    
    def calculate_metrics(self, predictions, ground_truth):
        """Calculate dummy metrics"""
        return {
            'mae': 0.0,
            'mse': 0.0,
            'rmse': 0.0
        }
    
    def calculate_loss(self, predictions, ground_truth):
        """Return dummy loss"""
        return 0.0