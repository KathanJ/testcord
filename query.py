from fastapi import FastAPI, Query, APIRouter
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import numpy as np
from gensim.utils import simple_preprocess
from transformers import DistilBertTokenizer, DistilBertModel
import torch

# Load environment variables
load_dotenv()
ATLAS_URI = os.getenv('ATLAS_URI')

# Connect to MongoDB
client = MongoClient(ATLAS_URI)
database = client.get_database("ml-model-datasource-watchcord")
collection = database.get_collection("products-data")

# Initialize FastAPI and APIRouter
app = FastAPI()
query_router = APIRouter()

# Load DistilBERT model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")


def generate_embeddings(text_list):
    """
    Generate embeddings consistently for both storage and search.
    """
    if isinstance(text_list, str):
        text_list = [text_list]

    model.eval()  # Set the model to evaluation mode

    # Tokenize the input text list
    inputs = tokenizer(text_list, padding=True, truncation=True, return_tensors='pt', max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)

    # Get the full embeddings
    last_hidden_state = outputs.last_hidden_state

    # Directly slice to the first 256 dimensions
    truncated_embeddings = last_hidden_state[:, :, :256]  # Keep all tokens, slice first 256 dimensions

    # Convert to numpy and take the mean for each input
    embeddings = truncated_embeddings.mean(dim=1).numpy()  # Average to get one embedding per input

    return embeddings


@query_router.get("/search")
async def search(
    query: str = Query(..., description="Search query"),
    min_price: float = Query(None, description="Minimum price filter"),
    max_price: float = Query(None, description="Maximum price filter"),
    min_rating: float = Query(None, description="Minimum rating filter")
):
    # Preprocess the query
    processed_text = simple_preprocess(query)
    
    # Generate truncated embeddings for the processed query
    embeddings = generate_embeddings([" ".join(processed_text)])
    
    # Convert the numpy array to a regular Python list
    query_vector = embeddings.flatten().tolist()

    # Build the search pipeline with vector search and filtering criteria
    search_pipeline = [
        {
            "$vectorSearch": {
                "exact": False,
                "index": "vector_index",
                "path": "embeddings",
                "queryVector": query_vector,
                "numCandidates": 100,
                "limit": 10
            }
        }
    ]

    # Append filtering stages based on optional parameters
    match_stage = {}

    if min_price is not None:
        match_stage["mrp"] = {"$gte": min_price}

    if max_price is not None:
        match_stage["mrp"] = match_stage.get("mrp", {})
        match_stage["mrp"]["$lte"] = max_price

    if min_rating is not None:
        match_stage["rating"] = {"$gte": min_rating}

    if match_stage:
        search_pipeline.append({"$match": match_stage})

    # Projection to include specific fields
    search_pipeline.append({
        "$project": {
            "title": 1,   # Include the title field
            "output": 1,  # Include the output field
            "rating": 1,
            "categories": 1,
            "price": 1,
            "_id": 0      # Exclude the default _id field
        }
    })

    # Execute the search
    results = list(collection.aggregate(search_pipeline))

    return {"results": results}


# Add the router to the FastAPI app
app.include_router(query_router)
