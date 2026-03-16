from fastapi import HTTPException 
import logging,json 

logger = logging.getLogger(__name__)

class LLMService: 
    def __init__(self): 
        self.system_message = {"summarization":{"role":"system","content":"You are a summarization AI assistant"}, 
                               "question_answer":{"role":"system","content":"You are a retrieval-augmented AI assistant"},
                               "synthesize_context":{"role":"system","content":"You are generating short contextual descriptions for document chunks to improve retrieval in a RAG system."},
                               "analyze_query":{"role":"system","content":"You are a professional query analysis AI assistant."},
                               "hallucination_check":{"role":"system","content":"You are a highly skeptical data auditor."},
                               "direct_answer":{"role":"system","content":"You are a document assistant. Your role is to help users retrieve and understand information from their uploaded documents."}} 
    def format_user_content(self,task,context=None,question=None,conversation_history = None,old_summary=None,chunk=None,answer=None):
            if task not in self.system_message: 
                raise HTTPException(400,"Chatbot doesn't support this task") 
            with open(f"app/prompt/{task}.txt", "r") as f: 
                raw_content = f.read() 
            if task == "question_answer": 
                return (raw_content.replace("<<CONTEXT>>",context).replace("<<QUESTION>>",question))
            elif task == "synthesize_context":
                return raw_content.replace("<<CHUNK>>",chunk)
            elif task == "analyze_query" or task == "direct_answer":
                 return raw_content.replace("<<QUESTION>>",question)
            elif task == "hallucination_check":
                 return (raw_content.replace("<<CONTEXT>>",context).replace("<<QUESTION>>",question).replace("<<ANSWER>>",answer))
            elif task == "summarization":
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
                logger.info(f"Calling LLM for {task} task")
                response = llm.invoke(prompt)
                logger.info(f"LLM generated answer for {task} task")
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
    def analyze_query(self, llm,question: str) -> dict:
        user_content = self.format_user_content("analyze_query", question=question)
        model_output = self.ask_model(llm, "analyze_query", user_content)

        try:
            result = json.loads(model_output.content.strip())
            if result.get("type") not in ("chit_chat", "simple", "complex"):
                raise ValueError
            return result
        except Exception:
            # Fallback if LLM doesn't return correct JSON format as "simple" is the least harmful way when LLM predicts incorrectness
            return {"type": "simple", "round1": [question], "round2":[]}
        
    def hallucination_check(self,llm, question:str,context:str, answer: str):
        user_content = self.format_user_content("hallucination_check",context=context, question=question, answer=answer)
        model_output = self.ask_model(llm, "hallucination_check", user_content)

        try:
            result = json.loads(model_output.content.strip())
            confidence = float(result.get("confidence", 1.0))
        except Exception:
            confidence = 1.0

        return confidence
llm_service = LLMService()
