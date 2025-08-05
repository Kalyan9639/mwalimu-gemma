# # updated_backend.py

import os
import tempfile
import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Tuple

# --- Imports for Google, Chroma Cloud, and Environment Variables ---
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import chromadb

# --- Standard LangChain Imports ---
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.schema import Document
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.tools import TavilySearchResults
from langchain_community.document_loaders import UnstructuredImageLoader, UnstructuredPDFLoader, WebBaseLoader


# --- GLOBAL SETUP: GOOGLE GENERATIVE AI ---
try:
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY not found in .env file.")
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3,
        max_output_tokens=2048,
        convert_system_message_to_human=True
    )
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
except Exception as e:
    print(f"[FATAL ERROR] Could not initialize Google Generative AI. Error: {e}")
    exit()


# --- ⭐️ DATABASE SETUP: CORRECTED FOR CHROMA CLOUD ⭐️ ---
try:
    # Load Chroma Cloud credentials from the .env file
    CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
    CHROMA_TENANT = os.getenv("CHROMA_TENANT")
    CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")

    if not all([CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE]):
        raise ValueError("CHROMA_API_KEY, CHROMA_TENANT, and CHROMA_DATABASE must be set in .env file.")

    # Initialize the ChromaDB CloudClient as per the official documentation
    # This connects to your hosted database on the cloud.
    client = chromadb.CloudClient(
        tenant=CHROMA_TENANT,
        database=CHROMA_DATABASE,
        api_key=CHROMA_API_KEY
    )
    print("[INFO] Successfully connected to ChromaDB Cloud.")

    # The LangChain Chroma wrapper now uses the CloudClient.
    # We are NO LONGER using `persist_directory`.
    user_vector_store = Chroma(
        client=client,
        collection_name="user_profiles",
        embedding_function=embeddings,
    )
    content_vector_store = Chroma(
        client=client,
        collection_name="learning_content",
        embedding_function=embeddings,
    )
except Exception as e:
    print(f"[FATAL ERROR] Could not connect to ChromaDB Cloud. Error: {e}")
    exit()


# --- AGENT CLASSES (Unchanged) ---

class UtilityAgent:
    def extract_text_from_file(self, file_path: str) -> str | None:
        try:
            loader = UnstructuredImageLoader(file_path) if file_path.lower().endswith(('.png', '.jpg', '.jpeg')) else UnstructuredPDFLoader(file_path)
            return "\n".join([doc.page_content for doc in loader.load()])
        except Exception: return None

    def save_user_profile(self, email: str, profile_data: Dict):
        doc = Document(page_content=json.dumps(profile_data), metadata={"email": email})
        user_vector_store.add_documents([doc], ids=[email])

    def load_user_profile(self, email: str) -> Dict | None:
        try:
            retrieved_doc = user_vector_store.get(ids=[email], include=["metadatas", "documents"])
            return json.loads(retrieved_doc['documents'][0]) if retrieved_doc and retrieved_doc['documents'] else None
        except Exception: return None

class TopicSequencingAgent:
    def run(self, syllabus_text: str, completed_topics: str) -> List[str] | None:
        prompt = ChatPromptTemplate.from_template(
            """You are an expert curriculum planner. Analyze the syllabus and completed topics.
            CRITICAL RULE: Your response MUST NOT include any topic already listed under 'Completed Topics'.
            Your task is to identify the next 3-4 entirely new, granular topics that logically follow the learning progression.
            Your final answer must be a valid JSON list of strings. Do not add any other text.
            Syllabus: {syllabus}
            Completed Topics: {completed}"""
        )
        chain = prompt | llm | JsonOutputParser()
        try:
            return chain.invoke({"syllabus": syllabus_text, "completed": completed_topics})
        except Exception as e:
            print(f"[DEBUG] Error in TopicSequencingAgent: {e}")
            return None

class ContentIngestionAgent:
    def run(self, topic: str, subject: str) -> bool:
        if content_vector_store.get(where={"topic": topic})['documents']: return True
        search_query = f"In-depth explanation and examples of '{topic}' in {subject}"
        try:
            search_tool = TavilySearchResults(max_results=3)
            urls = [result['url'] for result in search_tool.invoke(search_query)]
            if not urls: return False
            loader = WebBaseLoader(urls, continue_on_failure=True)
            docs = loader.load()
            if not docs: return False
            splits = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200).split_documents(docs)
            for doc in splits: doc.metadata["topic"] = topic
            content_vector_store.add_documents(splits)
            return True
        except Exception: return False

class ResourceGenerationAgent:
    def __init__(self, topics_list: list):
        self.topics = topics_list
        self.retriever = content_vector_store.as_retriever(search_kwargs={'k': 12, 'filter': {'$or': [{'topic': t} for t in topics_list]}})

    def generate_lesson_content(self) -> str:
        prompt_template = ChatPromptTemplate.from_template(
            "You are a subject matter expert preparing teaching materials. "
            "Your task is to provide the core content and explanations for the topics: **{topics}**. "
            "This is NOT a plan of activities; it is the actual knowledge the teacher needs. "
            "For EACH topic, provide a clear explanation, key definitions, and examples based ONLY on the provided context.\n\n"
            "CONTEXT:\n{context}"
        )
        rag_chain = ({"context": self.retriever, "topics": RunnablePassthrough()} | prompt_template | llm | StrOutputParser())
        return rag_chain.invoke(", ".join(self.topics))

    def get_chat_chain(self) -> Any:
        system_prompt = """You are Mwalimu, an expert AI mentor for teachers, specializing in explaining educational topics clearly and comprehensively.
        You have been provided with specific CONTEXT from lesson materials. Your primary goal is to answer the user's question by seamlessly blending the information from the CONTEXT with your own general knowledge to provide the most helpful response possible.
        - Prioritize the CONTEXT: First, use the provided context as the foundation of your answer.
        - Enhance with your knowledge: Augment the information from the context with your broader understanding of the topic to provide a richer, more complete explanation.
        - Be an expert partner: Act as a helpful mentor who can go beyond the provided text to give the user the best possible answer.
        CONTEXT:
        {context}
        """
        prompt = ChatPromptTemplate.from_messages([("system", system_prompt), MessagesPlaceholder(variable_name="chat_history"), ("human", "{question}")])
        chain = RunnablePassthrough.assign(context=lambda x: "\n\n".join(doc.page_content for doc in self.retriever.get_relevant_documents(x["question"]))) | prompt | llm | StrOutputParser()
        return chain

class MwalimuOrchestrator:
    def __init__(self):
        self.util_agent = UtilityAgent()
        self.topic_agent = TopicSequencingAgent()
        self.ingestion_agent = ContentIngestionAgent()

    def prepare_daily_lesson(self, profile: Dict) -> Tuple[str | None, Any | None]:
        topics = self.topic_agent.run(profile['syllabus_text'], profile['completed_topics'])
        if not topics: return None, None
        for topic in topics: self.ingestion_agent.run(topic, profile['subject'])
        resource_generator = ResourceGenerationAgent(topics)
        lesson_content = resource_generator.generate_lesson_content()
        return lesson_content, resource_generator

    def handle_chat(self, prompt: str, resource_generator: Any) -> str:
        if not resource_generator: return "No active lesson. Cannot chat."
        chat_chain = resource_generator.get_chat_chain()
        return chat_chain.invoke({"question": prompt, "chat_history": []})

# --- API SETUP & ENDPOINTS (Unchanged) ---
app = FastAPI(title="Mwalimu AI Mentor API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
orchestrator = MwalimuOrchestrator()
sessions: Dict[str, Dict] = {}

class OnboardResponse(BaseModel): message: str; email: str; name: str
class LessonResponse(BaseModel): lesson_plan: str; topics: List[str]
class ChatRequest(BaseModel): email: str; prompt: str; topics: List[str] | None = None
class ChatResponse(BaseModel): response: str

@app.post("/onboard", response_model=OnboardResponse)
async def onboard_teacher(email: str=Form(...), name: str=Form(...), subject: str=Form(...), grade: str=Form(...), completed_topics: str=Form(...), syllabus_file: UploadFile=File(...)):
    if orchestrator.util_agent.load_user_profile(email):
        return {"message": "Login successful", "email": email, "name": name}
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(syllabus_file.filename)[1]) as tmp:
            tmp.write(await syllabus_file.read())
            tmp_path = tmp.name
        syllabus_text = orchestrator.util_agent.extract_text_from_file(tmp_path)
        os.remove(tmp_path)
        if not syllabus_text: raise HTTPException(400, "Could not extract text from file.")
        profile_data = {"email": email, "name": name, "subject": subject, "grade": grade, "syllabus_text": syllabus_text, "completed_topics": completed_topics}
        orchestrator.util_agent.save_user_profile(email, profile_data)
        return {"message": "Onboarding successful", "email": email, "name": name}
    except Exception as e: raise HTTPException(500, f"An unexpected error occurred: {e}")

@app.get("/start_lesson/{email}", response_model=LessonResponse)
def start_lesson(email: str):
    if email in sessions and sessions[email]:
        return sessions[email]
    profile = orchestrator.util_agent.load_user_profile(email)
    if not profile: raise HTTPException(404, "User not found.")
    lesson_content, resource_generator = orchestrator.prepare_daily_lesson(profile)
    if not lesson_content: raise HTTPException(500, "Could not generate lesson content.")
    lesson_data = {"lesson_plan": lesson_content, "topics": resource_generator.topics}
    sessions[email] = lesson_data
    return lesson_data

@app.post("/mark_and_get_next_lesson/{email}", response_model=LessonResponse)
def mark_and_get_next_lesson(email: str):
    profile = orchestrator.util_agent.load_user_profile(email)
    if not profile: raise HTTPException(404, "User not found.")
    if email in sessions and sessions[email]:
        completed_topics = sessions[email].get("topics", [])
        profile['completed_topics'] += "\n" + "\n".join(completed_topics)
        orchestrator.util_agent.save_user_profile(email, profile)
    lesson_content, resource_generator = orchestrator.prepare_daily_lesson(profile)
    if not lesson_content: raise HTTPException(500, "Could not generate the next lesson.")
    lesson_data = {"lesson_plan": lesson_content, "topics": resource_generator.topics}
    sessions[email] = lesson_data
    return lesson_data

@app.post("/chat", response_model=ChatResponse)
def chat_with_mwalimu(request: ChatRequest):
    current_topics = []
    if request.email in sessions and sessions[request.email]:
        current_topics = sessions[request.email].get("topics", [])
    elif request.topics:
        current_topics = request.topics
    if not current_topics:
        raise HTTPException(status_code=404, detail="No active session found. Please go to the dashboard to start a lesson.")
    resource_generator = ResourceGenerationAgent(current_topics)
    response_text = orchestrator.handle_chat(request.prompt, resource_generator)
    return {"response": response_text}
