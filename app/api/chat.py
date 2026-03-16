from fastapi import APIRouter, Depends,Request
from fastapi.responses import StreamingResponse
from sqlmodel import Session
import logging

from ..schemas.request_model import ChatbotRequest
from ..schemas.response_model import Error, ChatBotResponse

from ..model import llm
from ..service.RAG_services.RAG_service import rag_service
from ..service.LLM_service import llm_service
from ..service.CM_service import CM_service
from ..service.DB_service import db_service
from ..deps import check_file_available,check_session_id_available,validate_file_status
from ..db import get_session
from ..core.limiter import limiter
from ..core.env_config import settings 

router = APIRouter(prefix="/chat", tags=["Chat"])
logger =logging.getLogger(__name__)

@router.post("",response_model=ChatBotResponse)
@limiter.limit("10/minute") #10 request each minute each IP
def chat(request: Request,query: ChatbotRequest, db: Session = Depends(get_session)):
    """Select the file you would like the chatbot to retrieve information from (by entering file ID), and enter the question you wish to ask.\n
    You can check whether the file exists and file ID by Get Files\n
    Session name is automatically set to session ID.\n
    You can change it later"""
    logger.info("Chat request received",extra={"file_id": query.file_id})

    #-----Guard rail----------
    if not check_file_available(query.file_id,db):
        logger.warning("File not found", extra={"file_id": query.file_id})
        return Error(code=404,error="File not Found")
    if not validate_file_status(query.file_id,db):
        logger.warning("File is not ready",extra={"file_id":query.file_id})
        return Error(code=400,error="File is not ready!")
    if query.session_id is not None and not check_session_id_available(query.session_id,db):
        logger.warning("Session ID not found",extra={"session_id":query.session_id})
        return Error(code=404,error="Session ID not Found")
    session_id = (CM_service.generate_session_id() if query.session_id is None else query.session_id)


    #------LLM Call 1: Analyze query--------
    analysis = llm_service.analyze_query(llm,query.question)
    q_type   = analysis["type"]
    logger.info(f"Query analyzed. Type: {q_type}")


    #----Chit-chat: answer directly------
    if q_type == "chit_chat":
        user_content = llm_service.format_user_content("direct_answer", question=query.question)
        answer = llm_service.ask_model(llm, "direct_answer", user_content)
        try: 
            db_service.create_dialog(session_id,session_id,"user",user_content,db)
            db_service.create_dialog(session_id,session_id,"assistant",answer.content,db)
        except Exception as e:
            db.rollback()
            logger.error(f"{e}")
            raise
        return ChatBotResponse(model_name=llm.model_name,session_id=session_id,session_name=session_id,answer=answer.content,confidence=1.0)
    
    #----Get k chunks. If None => lack of context
    context = rag_service.load_FAISS_and_retrieve(query.file_id,query.question,analysis,settings.strategy,db)
    if context is None:
        return ChatBotResponse(model_name=llm.model_name,session_id=session_id,session_name=session_id,answer="Lack of context. Can not answer this question. Please try another question", confidence=0.0) 

    #Load conversation history 
    conversation_history = CM_service.analyze_conversation_history(session_id,db,llm)

    #create user content 
    user_content = llm_service.format_user_content("question_answer",context,query.question)

    #----LLM Call 2: Answer question--------------
    model_output = llm_service.ask_model(llm,"question_answer",user_content,conversation_history)
    try: 
        db_service.create_dialog(session_id,session_id,"user",user_content,db)
        db_service.create_dialog(session_id,session_id,"assistant",model_output.content,db)
    except Exception as e:
        db.rollback()
        logger.error(f"{e}")
        raise
    def event_generator():
        try:
            full_response = ""
            for token in llm_service.stream_model(llm,"question_answer",user_content,conversation_history):
                full_response += token
                yield token
        finally:
            if full_response:
                try: 
                    db_service.create_dialog(session_id,session_id,"user",user_content,db)
                    db_service.create_dialog(session_id,session_id,"assistant",full_response,db)
                except Exception as e:
                    db.rollback()
                    logger.error(f"{e}")
                    raise
    logger.info("Chat response generated")

    # ── LLM Call 3: Hallucination check ──────────────────
    confidence = llm_service.hallucination_check(llm,query.question, context, model_output.content)

    disclaimer = ""
    if confidence < settings.confidence_threshold:
        disclaimer = "\n\n The answer may not be accurate based on the context provided."
    logger.info(f"Check hallucination successfully. Confidence: {confidence}")
    # return StreamingResponse(event_generator(),media_type="text/plain") #Use this if we want streaming response
    return ChatBotResponse(model_name=llm.model_name,session_id=session_id,session_name=session_id,answer=model_output.content + disclaimer, confidence=confidence)