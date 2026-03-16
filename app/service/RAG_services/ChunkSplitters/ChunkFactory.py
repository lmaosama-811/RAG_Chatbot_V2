class ChunkerRegistry:
    registry={}

    @classmethod
    def register(cls,strategy):
        def decorator(processor):
            cls.registry[strategy] = processor()
            return processor
        return decorator
    @classmethod
    def get_registry(cls,strategy):
        processor = cls.registry.get(strategy)
        if processor is None:
            raise ValueError(f"Chunking strategy '{strategy}' not found. Available strategies: {list(cls.registry.keys())}")
        return processor