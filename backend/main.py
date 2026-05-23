from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from modules.audio_routes import (
    router as audio_router
)

from core.db import (
    initialize_database
)

from modules.analytics_routes import (
    router as analytics_router
)

app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

initialize_database()

app.include_router(
    audio_router
)

app.include_router(
    analytics_router
)

@app.get("/")
def home():

    return {
        "project": "VoxGuard AI",
        "status": "running"
    }