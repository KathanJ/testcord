from fastapi import FastAPI, Query, APIRouter
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import numpy as np
from gensim.utils import simple_preprocess
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.decomposition import PCA
import torch

# Load environment variables
load_dotenv()
ATLAS_URI = os.getenv('ATLAS_URI')

# Connect to MongoDB
client = MongoClient(ATLAS_URI)
database = client.get_database("ml-model-datasource-watchcord")
collection = database.get_collection("products-data")

query_router = APIRouter()

# Load DistilBERT model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

# PCA model to reduce embeddings
pca = PCA(n_components=48)


# Function to generate embeddings using DistilBERT
def generate_embeddings(text_list):
    if isinstance(text_list, str):
        text_list = [text_list]
    inputs = tokenizer(text_list, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state
    mean_embeddings = embeddings.mean(dim=1)
    return mean_embeddings.numpy()


def apply_pca(embeddings, n_components=48):
    """
    Apply PCA to embeddings with proper handling of single samples
    """
    # If we have a single sample, create synthetic samples
    if embeddings.shape[0] == 1:
        # Create synthetic samples by adding small random noise
        n_synthetic = max(n_components + 1, 50)  # Ensure we have enough samples
        noise_scale = 0.000000001  # Small noise to maintain similarity
        
        # Generate synthetic samples
        synthetic_samples = np.repeat(embeddings, n_synthetic, axis=0)
        noise = np.random.normal(0, noise_scale, (n_synthetic, embeddings.shape[1]))
        synthetic_samples += noise
        
        # Fit PCA on synthetic samples
        pca = PCA(n_components=n_components)
        pca.fit(synthetic_samples)
        
        # Transform only the original embedding
        reduced_embedding = pca.transform(embeddings)
        return reduced_embedding
    else:
        # If we have multiple samples, apply PCA directly
        pca = PCA(n_components=n_components)
        return pca.fit_transform(embeddings)


@query_router.get("/search")
async def search(
    query: str = Query(..., description="Search query"),
    min_price: float = Query(None, description="Minimum price filter"),
    max_price: float = Query(None, description="Maximum price filter"),
    min_rating: float = Query(None, description="Minimum rating filter")
):
    # Preprocess the query
    processed_text = simple_preprocess(query)
    
    # Generate embeddings for the processed query
    embeddings = generate_embeddings([" ".join(processed_text)])
    
    # Apply PCA to reduce the dimensionality
    reduced_embeddings = apply_pca(embeddings)
    
    # Convert the numpy array to a regular Python list
    query_vector = reduced_embeddings.flatten().tolist()  # Convert to Python list

    print(query_vector)

    # Build the search pipeline with filtering criteria
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

    # Append filtering stage based on optional parameters
    match_stage = {}

    if min_price is not None:
        match_stage["mrp"] = {"$gte": min_price}

    if max_price is not None:
        if "mrp" not in match_stage:
            match_stage["mrp"] = {}  # Initialize if not already set
        match_stage["mrp"]["$lte"] = max_price  # Update the existing condition

    if min_rating is not None:
        match_stage["rating"] = {"$gte": min_rating}

    if match_stage:
        search_pipeline.append({"$match": match_stage})
    
    print(match_stage)

    # Execute the search
    results = list(collection.aggregate(search_pipeline))

    return {"results": results}

# Add the router to the FastAPI app
app = FastAPI()
app.include_router(query_router)