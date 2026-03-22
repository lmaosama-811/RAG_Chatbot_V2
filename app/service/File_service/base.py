from abc import ABC, abstractmethod 

class FileProcessor(ABC):
    @abstractmethod
    def save_file(self,file_bytes,file_name):
        pass
    @abstractmethod
    def get_file_path(self,folder,file_id):
        pass
    @abstractmethod
    def get_file(self,file_id,upload_file_path=None):
        pass
    @abstractmethod
    def process_file(self,file_id,upload_file_path=None):
        pass
    @abstractmethod
    def get_list_file(self):
        pass
    @abstractmethod
    def is_complicated_file(self,doc):
        pass 