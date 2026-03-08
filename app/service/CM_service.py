from .LLM_service import llm_service
from .DB_service import db_service
from tiktoken import get_encoding
import uuid 

class ConversationManagement:
    def __init__(self):
        self.summarization = {"role":"system"}
        self.token_threshold = 4000
        self.compress_threshold = 5 #24 messages -> compress 
        self.compress_batch = 3 # 5 message each compress 
        self.encoder = get_encoding("cl100k_base")
    def format_history(self,dialog:tuple): #return conversation history 
        return {"role":dialog[0],"content":dialog[1]}
    def load_conversation_history_and_update_summarization(self,session_id,db,llm,index=None,old_summary=None): #lấy history, nếu vượt threshold thì update summary vào old summary
        all_dialogs = db_service.get_conversation_history(session_id,db) #get all dialogs
        dialogs = (all_dialogs if index is None else all_dialogs[index-1:index+5])
        count = 0
        recent_dialogs = []
        for i in range(len(dialogs)-1,-1,-1):
            if (count + len(self.encoder.encode(dialogs[i][1]))+3) <= self.token_threshold:
                recent_dialogs.append(self.format_history(dialogs[i]))
                count += (len(self.encoder.encode(dialogs[i][1])) + 3)
            else:
                sum_list = []
                for j in range(i+1):
                    sum_list.append(self.format_history(dialogs[j]))
                if len(sum_list) !=0:
                    user_content = llm_service.format_user_content(task="summarization",conversation_history=sum_list,old_summary=old_summary)
                    summary = llm_service.ask_model(llm,"summarization",user_content)
                    db_service.create_summary(session_id,i+1,summary.content,db) #base 1 according to table 
                    return [{**self.summarization,**{"content":summary.content}}] + recent_dialogs # list[role,content]: đoạn được summary và recent_dialogs 
        return recent_dialogs #list(role,content)
    
    def analyze_conversation_history(self,session_id,db,llm): #return history conversation list[dict(role,content)]
        old_summary = db_service.get_last_summary(session_id,db) #Summary
        if old_summary is None: #không có old summary -> chưa tóm tắt lần nào -> load history như bình thường, nếu quá threshold thì tự thêm summary và trả về 
            return self.load_conversation_history_and_update_summarization(session_id,db,llm)
        elif (db_service.get_last_dialog(session_id,db).id - old_summary.covered_until_message_id +1) <= self.compress_threshold: #nhỏ hơn threshold thì trả recent_dialogs + old_summary 
            return [{"role":"system","content":old_summary.content}] + self.load_conversation_history_and_update_summarization(session_id,db,llm,old_summary.covered_until_message_id)
        else: #lấy 5 messages + old summary = new summary, cho vào table và trả về history 
            dialogs_for_summary = db_service.get_conversation_history(session_id,db)[old_summary.covered_until_message_id -1:old_summary.covered_until_message_id +5]
            dialogs_for_summary = [self.format_history(d) for d in dialogs_for_summary]
            user_content = llm_service.format_user_content(task="summarization",conversation_history=dialogs_for_summary,old_summary=old_summary)
            summary = llm_service.ask_model(llm,"summarization",user_content)
            db_service.create_summary(session_id,old_summary.covered_until_message_id+1,summary.content,db) #base 1 according to table
            return [{"role":"system","content":summary.content}] + self.load_conversation_history_and_update_summarization(session_id,db,llm,old_summary.covered_until_message_id)

             
    def generate_session_id(self):
        return uuid.uuid4().hex[:16]
    
CM_service = ConversationManagement()
        

            
