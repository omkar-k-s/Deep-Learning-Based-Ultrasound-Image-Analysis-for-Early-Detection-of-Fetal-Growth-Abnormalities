import os
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
import gradio as gr

# Import our Flask app
from backend.app import app as flask_app

# Create FastAPI app
app = FastAPI()

# Mount the Flask app to the root
app.mount("/", WSGIMiddleware(flask_app))

# Gradio dummy so Hugging Face recognizes it as a Gradio Space (optional, but ensures healthchecks pass)
demo = gr.Interface(fn=lambda x: x, inputs="text", outputs="text")

if __name__ == "__main__":
    import uvicorn
    # Hugging Face runs on 7860
    uvicorn.run(app, host="0.0.0.0", port=7860)
