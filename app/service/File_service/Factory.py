import os 

class FileProcessorRegistry:
    registry={}

    @classmethod
    def register(cls,ext):
        def decorator(processor):
            cls.registry[ext] = processor()
            return processor
        return decorator
    @classmethod
    def get_registry(cls,file_name = None,extension=None):
        ext = os.path.splitext(file_name)[1].lower() if file_name is not None else extension
        processor = cls.registry.get(ext)
        if not processor:
            raise #Error
        return processor