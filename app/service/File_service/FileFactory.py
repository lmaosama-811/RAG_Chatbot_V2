import os
from fastapi import HTTPException

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
        if processor is None:
            return cls.registry.get("rest")
        return processor
    
    @classmethod
    def inspect_file_and_routing(cls,file_id,processor=None,file_name=None, extension=None,upload_file_path=None): #classify whether file is simple or complicated structured
        processor = (cls.get_registry(file_name,extension) if processor is None else processor)
        doc = processor.get_file(file_id,upload_file_path)
        return processor.is_complicated_file(doc)