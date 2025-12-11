import cohere
from qdrant_client import QdrantClient

# Initialize Cohere client
cohere_client = cohere.Client("H29FlEmZ77wCWpmN1xazfRdV6pH8xoy1l6wEFIm2")

# Connect to Qdrant
qdrant = QdrantClient(
    url="https://76b54101-b9f2-41af-b61f-64238bb0c66d.us-west-1-0.aws.cloud.qdrant.io", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0._VIZLuGpWM1uCdUkgJkmuqgqPWOC21_hwToLxmCbJwE"
)

def get_embedding(text):
    """Get embedding vector from Cohere Embed v3"""
    response = cohere_client.embed(
        model="embed-english-v3.0",
        input_type="search_query",  # Use search_query for queries
        texts=[text],
    )
    return response.embeddings[0]  # Return the first embedding

def retrieve(query):
    embedding = get_embedding(query)
    result = qdrant.query_points(
        collection_name="AI-Native-Technical-Textbook",
        query=embedding,
        limit=5
    )
    return [point.payload["text"] for point in result.points]

# Test
print(retrieve("What data do you have?"))