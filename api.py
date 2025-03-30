from fastapi.middleware.cors import CORSMiddleware  # Import CORSMiddleware
from fastapi import FastAPI, Query, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os

import query_data as qd

# Initialize FastAPI app
app = FastAPI()

# Allow all origins for testing (you can change this to specific origins)
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1",
]

# Add CORSMiddleware to handle OPTIONS requests and CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # ✅ อนุญาตจากแหล่งที่มาที่กำหนด
    allow_credentials=True,
    allow_methods=["*"],   # ✅ อนุญาตทุก method (GET, POST, OPTIONS, ...)
    allow_headers=["*"],   # ✅ อนุญาตทุก header
)

# ✅ เปลี่ยนชื่อ QueryModel เพื่อไม่ให้ชนกับ fastapi.Query
class QueryModel(BaseModel):
    question: str  # Expects "question" as the field name

@app.post("/ask")
def receive_query(query: QueryModel):  # ใช้ชื่อ QueryModel
    return {"reply": f"{qd.query_rag(query.question)}"}

@app.options("/ask")
def options_route(response: Response):
    # Handle CORS preflight request (OPTIONS method)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

@app.get("/image")
def get_image(query: str = Query(...)):  # ใช้ Query จาก fastapi
    image_path = os.path.join("chroma", f"{query}.png")

    if not os.path.exists(image_path):
        return {"error": "Image not found"}

    return FileResponse(image_path, media_type="image/png")

# Run the API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
