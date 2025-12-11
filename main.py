from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI
from agents import set_tracing_disabled, function_tool, SQLiteSession
import os
from dotenv import load_dotenv
from agents import enable_verbose_stdout_logging, set_tracing_disabled
import cohere
from qdrant_client import QdrantClient
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

set_tracing_disabled(disabled=True)
load_dotenv()

# Load secrets from environment
GEMINI_API_KEY = "AIzaSyC_ZPnndrJ3pDf7Yt0Fw7jQb8aXZdbEXQE"
COHERE_API_KEY="H29FlEmZ77wCWpmN1xazfRdV6pH8xoy1l6wEFIm2"
QDRANT_URL="https://76b54101-b9f2-41af-b61f-64238bb0c66d.us-west-1-0.aws.cloud.qdrant.io"
QDRANT_API_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0._VIZLuGpWM1uCdUkgJkmuqgqPWOC21_hwToLxmCbJwE"

if GEMINI_API_KEY is None:
    raise RuntimeError("Missing GEMINI_API_KEY environment variable")
if COHERE_API_KEY is None:
    raise RuntimeError("Missing COHERE_API_KEY environment variable")
if QDRANT_URL is None or QDRANT_API_KEY is None:
    raise RuntimeError("Missing QDRANT_URL or QDRANT_API_KEY environment variables")

# Initialize FastAPI app
app = FastAPI(title="AI Textbook Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

provider = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=provider
)

# Initialize Cohere client
cohere_client = cohere.Client(COHERE_API_KEY)

# Connect to Qdrant
qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

def get_embedding(text):
    """Get embedding vector from Cohere Embed v3"""
    response = cohere_client.embed(
        model="embed-english-v3.0",
        input_type="search_query",  # Use search_query for queries
        texts=[text],
    )
    return response.embeddings[0]


@function_tool
def retrieve(query):
    embedding = get_embedding(query)
    result = qdrant.query_points(
        collection_name="AI-Native-Technical-Textbook",
        query=embedding,
        limit=5
    )
    return [point.payload["text"] for point in result.points]


agent = Agent(
    name="Assistant",
    instructions="""
    # Role

You are an expert AI tutor specializing in Physical AI and Humanoid Robotics. Your role is to provide clear, accurate explanations grounded exclusively in the course material. You demonstrate deep knowledge of the textbook content while maintaining pedagogical clarity and patience with learners at all levels.

# Task

Answer student questions about Physical AI and Humanoid Robotics by retrieving relevant textbook content and explaining it in an accessible way. Provide only information that exists in the retrieved material, and explicitly state when information is unavailable.

# Context

Students use this tutoring service to deepen their understanding of Physical AI and Humanoid Robotics concepts. By anchoring all responses in the textbook material, you ensure consistency, accuracy, and alignment with course learning objectives. This approach builds student confidence that the information is authoritative and directly supports their studies.

# Instructions

The assistant should:

1. Call the `retrieve` tool with the user's question to access relevant textbook content before formulating any response.

2. Base all explanations exclusively on the content returned by the `retrieve` tool—do not supplement with external knowledge or assumptions about what the textbook might contain.

3. Present information clearly by explaining concepts in simple terms, using examples from the retrieved content when available, and breaking complex ideas into logical steps.

4. If the `retrieve` tool fails to return information, encounters an error, or cannot access the requested content, respond with a user-friendly message such as "We're currently building out this section of the course material" or "This content is being prepared for you" instead of displaying technical errors or error messages.

5. Maintain an encouraging, supportive tone that invites follow-up questions and helps students feel comfortable asking for clarification.
""",
    model=model,
    tools=[retrieve]
)

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    response: str


@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Create session based on session_id
        user_session = SQLiteSession(f"conversation_{request.session_id}")

        result = await Runner.run(
            agent,
            input=request.message,
            session=user_session
        )

        return ChatResponse(response=result.final_output)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

