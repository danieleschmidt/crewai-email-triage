#!/usr/bin/env python3
"""
Predictive auto-scaling for email triage service using ML-based forecasting.
Anticipates email volume patterns and pre-scales infrastructure.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

@dataclass
class ScalingMetrics:
    """Metrics used for predictive scaling decisions."""
    timestamp: datetime
    email_volume: int
    processing_latency_p95: float
    cpu_utilization: float
    memory_utilization: float
    queue_depth: int
    cost_per_email: float

class PredictiveAutoScaler:
    """ML-powered predictive auto-scaling for email processing workloads."""
    
    def __init__(self, prediction_horizon_minutes: int = 30):
        self.prediction_horizon = prediction_horizon_minutes
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.historical_metrics: List[ScalingMetrics] = []
        self.is_trained = False
        
    async def collect_metrics(self) -> ScalingMetrics:
        """Collect current system metrics for scaling decisions."""
        # Implementation would integrate with monitoring stack
        current_time = datetime.now()
        
        # Mock implementation - replace with actual metrics collection
        return ScalingMetrics(
            timestamp=current_time,
            email_volume=self._get_current_email_volume(),
            processing_latency_p95=self._get_processing_latency(),
            cpu_utilization=self._get_cpu_utilization(),
            memory_utilization=self._get_memory_utilization(),
            queue_depth=self._get_queue_depth(),
            cost_per_email=self._get_cost_per_email()
        )
    
    def _extract_features(self, metrics: List[ScalingMetrics]) -> np.ndarray:
        """Extract ML features from historical metrics."""
        features = []
        
        for metric in metrics:
            # Time-based features
            hour_of_day = metric.timestamp.hour
            day_of_week = metric.timestamp.weekday()
            is_business_hour = 9 <= hour_of_day <= 17 and day_of_week < 5
            
            # Trend features (requires at least 2 data points)
            if len(self.historical_metrics) >= 2:
                prev_metric = self.historical_metrics[-2]
                email_volume_trend = metric.email_volume - prev_metric.email_volume
                latency_trend = metric.processing_latency_p95 - prev_metric.processing_latency_p95
            else:
                email_volume_trend = 0
                latency_trend = 0
            
            feature_vector = [
                hour_of_day,
                day_of_week,
                int(is_business_hour),
                metric.email_volume,
                metric.processing_latency_p95,
                metric.cpu_utilization,
                metric.memory_utilization,
                metric.queue_depth,
                metric.cost_per_email,
                email_volume_trend,
                latency_trend
            ]
            features.append(feature_vector)
            
        return np.array(features)
    
    def train_model(self) -> bool:
        """Train the predictive model on historical data."""
        if len(self.historical_metrics) < 100:  # Need sufficient training data
            logging.warning("Insufficient training data for predictive scaling")
            return False
            
        try:
            # Prepare training data
            features = self._extract_features(self.historical_metrics[:-1])
            targets = np.array([m.email_volume for m in self.historical_metrics[1:]])
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Train model
            self.model.fit(features_scaled, targets)
            self.is_trained = True
            
            logging.info("Predictive scaling model trained successfully")
            return True
            
        except Exception as e:
            logging.error(f"Failed to train predictive model: {e}")
            return False
    
    async def predict_scaling_needs(self) -> Tuple[int, float]:
        """Predict required replicas and confidence score."""
        if not self.is_trained:
            if not self.train_model():
                return self._fallback_scaling_decision()
        
        try:
            current_metrics = await self.collect_metrics()
            features = self._extract_features([current_metrics])
            features_scaled = self.scaler.transform(features)
            
            # Predict email volume for next prediction horizon
            predicted_volume = self.model.predict(features_scaled)[0]
            
            # Calculate required replicas based on predicted volume
            emails_per_replica_per_minute = 20  # Based on performance testing
            required_replicas = max(
                2,  # Minimum replicas
                int(np.ceil(predicted_volume / emails_per_replica_per_minute))
            )
            
            # Calculate confidence based on feature importances and historical accuracy
            confidence = self._calculate_prediction_confidence(features_scaled)
            
            logging.info(
                f"Predicted scaling: {required_replicas} replicas "
                f"for {predicted_volume:.0f} emails (confidence: {confidence:.2%})"
            )
            
            return required_replicas, confidence
            
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            return self._fallback_scaling_decision()
    
    def _fallback_scaling_decision(self) -> Tuple[int, float]:
        """Fallback scaling based on current metrics when ML prediction fails."""
        if not self.historical_metrics:
            return 2, 0.5  # Default conservative scaling
            
        current_metrics = self.historical_metrics[-1]
        
        # Simple rule-based scaling
        if current_metrics.queue_depth > 100:
            return 8, 0.7
        elif current_metrics.queue_depth > 50:
            return 5, 0.6
        elif current_metrics.queue_depth > 20:
            return 3, 0.5
        else:
            return 2, 0.5
    
    def _calculate_prediction_confidence(self, features: np.ndarray) -> float:
        """Calculate confidence score for the prediction."""
        # Simplified confidence calculation based on feature variance
        feature_variance = np.var(features)
        confidence = max(0.3, min(0.95, 1.0 - feature_variance / 10.0))
        return confidence
    
    # Mock methods - replace with actual monitoring integrations
    def _get_current_email_volume(self) -> int:
        return 45  # emails per minute
    
    def _get_processing_latency(self) -> float:
        return 180.5  # milliseconds
    
    def _get_cpu_utilization(self) -> float:
        return 65.0  # percentage
    
    def _get_memory_utilization(self) -> float:
        return 70.0  # percentage
    
    def _get_queue_depth(self) -> int:
        return 25  # pending emails
    
    def _get_cost_per_email(self) -> float:
        return 0.05  # USD per email processed