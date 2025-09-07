from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------
# Load environment variables
# ---------------------------
# Try to load .env from backend folder or project root
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

OPENAI_API_KEY = os.getenv("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")
if not OPENAI_API_KEY:
    raise RuntimeError(
        "Missing OPENAI_API_KEY in environment variables. "
        "Make sure you have a .env file with OPENAI_API_KEY=your_key_here"
    )

# ---------------------------
# OpenAI client
# ---------------------------
client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------------------
# Logging setup
# ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------
# FastAPI app setup
# ---------------------------
app = FastAPI(title="AI Story + Image Generator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Relaxed for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Models
# ---------------------------
class StoryRequest(BaseModel):
    idea: str
    genre: str = "fantasy"
    tone: str = "lighthearted"
    audience: str = "kids"
    art_style: str = "watercolor"
    scene_count: int = 4

class Scene(BaseModel):
    title: str
    narrative: str
    image_prompt: str
    image_url: Optional[str] = None
    image_error: Optional[str] = None

class StoryResponse(BaseModel):
    idea: str
    genre: str
    tone: str
    audience: str
    art_style: str
    scenes: List[Scene]

# ---------------------------
# Routes
# ---------------------------
@app.get("/")
def root():
    return {"message": "AI Story + Image backend is running."}

@app.post("/generate-story", response_model=StoryResponse)
def generate_story(req: StoryRequest):
    prompt = f"""
Expand the following idea into {req.scene_count} coherent scenes.
- Genre: {req.genre}
- Tone: {req.tone}
- Audience: {req.audience}
- Keep style consistent.
- Each scene should move the story forward.

Return ONLY valid JSON in this format:
{{
  "scenes": [
    {{
      "title": "Scene title",
      "narrative": "150-200 words of vivid, coherent narration",
      "image_prompt": "A concise visual description fit for illustration, in {req.art_style} style"
    }},
    ...
  ]
}}

Idea: "{req.idea}"
"""

    try:
        chat = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You write cohesive stories and output strictly valid JSON when asked."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
        )
        raw = chat.choices[0].message.content.strip()
        logger.info(f"Raw AI response: {raw}")

        parsed = json.loads(raw)
        scenes_data = parsed.get("scenes", [])
        scenes = []

        for scene in scenes_data:
            try:
                img = client.images.generate(
                    model="dall-e-3",
                    prompt=scene["image_prompt"],
                    size="1024x1024"
                )
                scene["image_url"] = img.data[0].url
                scene["image_error"] = None
            except Exception as e:
                scene["image_url"] = None
                scene["image_error"] = str(e)

            scenes.append(Scene(**scene))

        return StoryResponse(
            idea=req.idea,
            genre=req.genre,
            tone=req.tone,
            audience=req.audience,
            art_style=req.art_style,
            scenes=scenes
        )

    except json.JSONDecodeError:
        logger.error("Failed to parse AI response as JSON.")
        return StoryResponse(
            idea=req.idea,
            genre=req.genre,
            tone=req.tone,
            audience=req.audience,
            art_style=req.art_style,
            scenes=[
                Scene(
                    title="Parsing error",
                    narrative="The AI returned invalid JSON. Please try again or tweak your inputs.",
                    image_prompt="N/A",
                    image_url=None,
                    image_error="Invalid JSON from AI"
                )
            ]
        )
    except Exception as e:
        logger.exception("Unexpected error during story generation.")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/regenerate-image")
def regenerate_image(payload: dict):
    prompt = payload.get("image_prompt", "")
    size = payload.get("size", "1024x1024")
    try:
        img = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=size
        )
        return {"image_url": img.data[0].url}
    except Exception as e:
        return {"image_url": None, "error": str(e)}
