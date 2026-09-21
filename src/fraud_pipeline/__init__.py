"""Fraud intelligence pipeline package."""

__all__ = ["FraudPipeline", "KagglePipeline", "PipelineConfig"]


def __getattr__(name: str):
    if name == "KagglePipeline":
        from fraud_pipeline.kaggle_pipeline import KagglePipeline
        return KagglePipeline
    if name in {"FraudPipeline", "PipelineConfig"}:
        from fraud_pipeline.pipeline import FraudPipeline
        from fraud_pipeline.config import PipelineConfig

        mapping = {
            "FraudPipeline": FraudPipeline,
            "PipelineConfig": PipelineConfig,
        }
        return mapping[name]
    raise AttributeError(name)
