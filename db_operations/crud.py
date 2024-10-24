from fastapi import APIRouter, Request, Body
from pymongo import MongoClient
from dotenv import load_dotenv
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.decomposition import PCA
import os
import torch
import numpy as np

load_dotenv()

ATLAS_URI = os.getenv('ATLAS_URI')

crud_router = APIRouter()

# Load DistilBERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Function to generate embeddings
def generate_embeddings(text_list):
    if isinstance(text_list, str):
        text_list = [text_list]
    inputs = tokenizer(text_list, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state
    mean_embeddings = embeddings.mean(dim=1)
    return mean_embeddings.numpy()

# PCA reduction function
def apply_pca(embeddings, n_components):
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

@crud_router.post("/upload_new_data")
async def upload_new_data(data: dict = Body(...)):  
    try:
        print("Received new data.")

        if isinstance(data, dict):
            new_data = [data.copy()]
        elif isinstance(data, list):
            new_data = data.copy()
        else:
            print("Invalid input format.")
            return {"error": "Invalid input format. Expected a JSON object or list."}

        filtered_data_list = []
        categories_texts = []  # Collect all categories texts for batch processing
        titles_texts = []      # Collect all titles texts for batch processing

        for item in new_data:
            required_fields = ["title", "mrp", "product_id", "categories", "rating", "domain"]
            if all(field in item for field in required_fields):
                item['mrp'] = float(item['mrp'].replace("₹", "").replace(",", ""))
                filtered_data = {
                    "title": item["title"],
                    "mrp": item["mrp"],
                    "categories": item["categories"],
                    "rating": item["rating"],
                    "output": str((item["product_id"], item["domain"]))
                }

                categories_texts.append(' '.join(item['categories']) if isinstance(item['categories'], list) else item['categories'])
                titles_texts.append(item['title'])
                filtered_data_list.append(filtered_data)

        # Generate embeddings for both categories and titles in batches
        print("Generating category embeddings...")
        categories_embeddings = generate_embeddings(categories_texts)

        print("Generating title embeddings...")
        title_embeddings = generate_embeddings(titles_texts)

        # Concatenate the category and title embeddings along the second dimension
        combined_embeddings = np.concatenate([categories_embeddings, title_embeddings], axis=1)

        # Reduce the concatenated embeddings to 48 dimensions using PCA
        print("Reducing combined embeddings to 48 dimensions...")
        combined_embedding_reduced = apply_pca(combined_embeddings, n_components=48)  # Reduce to 48D

        # Assign the reduced embedding to the corresponding data entries
        for idx in range(len(filtered_data_list)):
            filtered_data_list[idx]['embeddings'] = combined_embedding_reduced[idx].tolist()

        print("Embeddings generated and reduced to 48 dimensions.")
    
        print("Connecting to MongoDB...")
        conn = MongoClient(ATLAS_URI)

        if filtered_data_list:
            conn.db.products.insert_many(filtered_data_list)
            return {"status": "Data uploaded successfully"}
        else:
            return {"status": "No valid data to upload"}

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return {"error": "Failed to process request", "details": str(e)}
