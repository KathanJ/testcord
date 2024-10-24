from fastapi import FastAPI
from db_operations.crud import crud_router  # Ensure correct import path
from query import query_router
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Include the CRUD router for upload endpoints
app.include_router(crud_router)

app.include_router(query_router)

# Root endpoint for basic server check
@app.get("/")
async def root():
    return {"message": "Server is running!"}
