from fastapi import HTTPException 
import logging 

logger = logging.getLogger(__name__)

class LLMService: 
    def __init__(self): 
        self.system_message = {"summarization":{"role":"system","content":"You are a summarization AI assistant"}, 
                               "question_answer":{"role":"system","content":"You are a retrieval-augmented AI assistant"}} 
    def format_user_content(self,task,context=None,question=None,conversation_history = None,old_summary=None):
            if task not in self.system_message: 
                raise HTTPException(400,"Chatbot doesn't support this task") 
            with open(f"app/prompt/{task}.txt", "r") as f: 
                raw_content = f.read() 
            if task == "question_answer": 
                return (raw_content.replace("<<CONTEXT>>",context).replace("<<QUESTION>>",question))
            history_text = ""
            if conversation_history:
                history_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in conversation_history])
            old_summary_text = (old_summary.content if old_summary is not None else "None")
            return (raw_content.replace("<<MESSAGES>>",history_text).replace("<<OLDSUMMARY>>",old_summary_text))
        
    def ask_model(self,llm,task,user_content,conversation_history=None):
            if conversation_history is None:
                 conversation_history = []
            prompt = [self.system_message[task]] + conversation_history + [{"role":"user","content":user_content}]
            try: 
                logger.info("Calling LLM")
                response = llm.invoke(prompt)
                logger.info("LLM generated")
                return response
            except TimeoutError: 
                raise HTTPException(504,"LLM timeout") 
    def stream_model(self, llm, task, user_content, conversation_history=None):

        if conversation_history is None:
            conversation_history = []

        prompt = [self.system_message[task]] + conversation_history + [{"role": "user", "content": user_content}]

        for chunk in llm.stream(prompt):
            if chunk.content:
                yield chunk.content
            
llm_service = LLMService()
