from sqlmodel import select,update, delete,desc
import re 
from ..schemas.request_model import ConversationHistory, Summary, FileStatus, ParentStore

class DBService:
    def clean_text(self, text: str) -> str:
        if text is None:
            return ""
        text = text.replace("\x00", "")
        text = text.replace("\n", " ")
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    
    def get_list_file(self,type_of_file,db):
        if type_of_file == "rest":
            cmd = select(FileStatus.file_name,FileStatus.file_id).where(FileStatus.type.notin_([".docx", ".pdf", ".txt"]))
        else:
            cmd = select(FileStatus.file_name,FileStatus.file_id).where(FileStatus.type==type_of_file)
        return db.exec(cmd).mappings().all() #List[Tuple]
    
    def get_conversation_history(self,session_id:int, db):
        cmd = select(ConversationHistory.role, ConversationHistory.content).where(ConversationHistory.session_id == session_id)
        return db.exec(cmd).all() #return List[tuple] as we selecting columns
    
    def get_last_dialog(self,session_id,db)-> ConversationHistory|None:
        cmd = select(ConversationHistory).where(ConversationHistory.session_id == session_id).order_by(desc(ConversationHistory.id)).limit(1)
        return db.exec(cmd).first()
    
    def create_dialog(self, session_id, session_name, role, content,db):
        clean_content = self.clean_text(content)
        new_diaglog = ConversationHistory(session_id=session_id,session_name=session_name,role=role,content=clean_content)
        db.add(new_diaglog)
        db.commit()

    def update_session_name(self,session_id,new_name,db):
        cmd = update(ConversationHistory).where(ConversationHistory.session_id==session_id).values(session_name=new_name)
        db.exec(cmd)
        db.commit()

    def delete_conversation(self,session_id,db):
        cmd= delete(ConversationHistory).where(ConversationHistory.session_id == session_id)
        db.exec(cmd)
        db.commit()

    def get_list_conversation(self,db):
        cmd = select(ConversationHistory.session_id,ConversationHistory.session_name,ConversationHistory.id).order_by(desc(ConversationHistory.id))
        all_dialogs = db.exec(cmd).all()
        # Get unique sessions and session_name
        seen = set()
        result = []
        for session_id, session_name, _ in all_dialogs:
            if session_id not in seen:
                seen.add(session_id)
                result.append({"session_id": session_id, "session_name": session_name})
        return result
    
    def get_conversation(self,session_id,db):
        cmd = select(ConversationHistory).where(ConversationHistory.session_id==session_id)
        return db.exec(cmd).first() #ConversationHistory|None 
    
    def create_summary(self,session_id:str, covered_until_message_id,content,db): #summary table
        clean_content = self.clean_text(content)
        new_summary = Summary(session_id=session_id, covered_until_message_id=covered_until_message_id,content=clean_content)
        db.add(new_summary)
        db.commit()

    def get_last_summary(self, session_id, db):
        cmd = select(Summary).where(Summary.session_id == session_id)\
                            .order_by(desc(Summary.id))\
                            .limit(1)
        return db.exec(cmd).first()
    
    def get_file_status(self,file_id,db):
        cmd = select(FileStatus.status).where(FileStatus.file_id==file_id)
        return db.exec(cmd).first()
    
    def get_file_type(self,file_id,db):
        cmd = select(FileStatus.type).where(FileStatus.file_id==file_id)
        return db.exec(cmd).first()
    
    def create_file(self, db, file_id, file_name, type, status="PENDING"):
        file = FileStatus(file_id=file_id, file_name=file_name, type=type, status=status)
        db.add(file)
        db.commit()

    def check_file_exists_by_name(self, file_name: str, db) -> bool:
        stmt = select(FileStatus).where(FileStatus.file_name == file_name)
        result = db.exec(stmt).first()
        return result is not None

    def update_file_status(self,file_id,status,db):
        cmd=update(FileStatus).where(FileStatus.file_id==file_id).values(status=status)
        db.exec(cmd)
        db.commit()

    def delete_file(self,file_id,db):
        cmd= delete(FileStatus).where(FileStatus.file_id==file_id)
        db.exec(cmd)
        db.commit()

    def create_parent_chunk(self,parent_id,file_id,context,db):
        new_parent_chunk = ParentStore(parent_id=parent_id,file_id=file_id,context=context)
        db.add(new_parent_chunk)
        db.commit()

    def get_parent_chunk(self,parent_id,db):
        cmd = select(ParentStore).where(ParentStore.parent_id == parent_id)
        return db.exec(cmd).first() #ParentStore

db_service = DBService()
