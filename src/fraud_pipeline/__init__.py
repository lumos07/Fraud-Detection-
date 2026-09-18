"""Fraud intelligence pipeline package."""

__all__ = ["FraudPipeline", "PipelineConfig"]


def __getattr__(name: str):
    if name in {"FraudPipeline", "PipelineConfig"}:
        from fraud_pipeline.pipeline import FraudPipeline
        from fraud_pipeline.config import PipelineConfig

        mapping = {
            "FraudPipeline": FraudPipeline,
            "PipelineConfig": PipelineConfig,
        }
        return mapping[name]
    raise AttributeError(name)
