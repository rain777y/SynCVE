"""
Generate test face images using Gemini API for SynCVE integration testing.
Creates a set of test images with different emotions, lighting, angles.

Uses the unified google-genai SDK (not the deprecated google-generativeai).
"""
import os
from pathlib import Path

from google import genai
from google.genai import types

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# Use the stable image generation model
IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL_NAME", "gemini-2.5-flash-image")


def generate_test_faces():
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY environment variable is required.")
        return

    client = genai.Client(api_key=GEMINI_API_KEY)

    output_dir = Path(__file__).parent / "generated"
    output_dir.mkdir(exist_ok=True)

    # Use Gemini image generation to create test faces
    scenarios = [
        (
            "happy_face",
            "A photorealistic portrait of a person with a genuine happy smile, "
            "well-lit, facing camera directly, plain background",
        ),
        (
            "sad_face",
            "A photorealistic portrait of a person looking sad with downturned mouth, "
            "well-lit, facing camera, plain background",
        ),
        (
            "angry_face",
            "A photorealistic portrait of a person with an angry expression, "
            "furrowed brows, well-lit, facing camera, plain background",
        ),
        (
            "surprised_face",
            "A photorealistic portrait of a person with a surprised expression, "
            "wide eyes and open mouth, well-lit, facing camera, plain background",
        ),
        (
            "neutral_face",
            "A photorealistic portrait of a person with a neutral calm expression, "
            "well-lit, facing camera, plain background",
        ),
        (
            "low_light_face",
            "A photorealistic portrait of a person in dim low lighting conditions, "
            "facing camera",
        ),
        (
            "side_angle_face",
            "A photorealistic portrait of a person at a 30 degree side angle, well-lit",
        ),
        (
            "multiple_faces",
            "A photorealistic photo of three people standing together facing the camera, "
            "all with different expressions, well-lit",
        ),
        (
            "no_face",
            "A photorealistic photo of a landscape with mountains and trees, no people at all",
        ),
        (
            "small_face_far",
            "A photorealistic full body photo of a person standing far away, "
            "very small in the frame, in a large room",
        ),
    ]

    for filename, prompt in scenarios:
        print(f"Generating: {filename}...")
        image_path = output_dir / f"{filename}.jpg"
        try:
            response = client.models.generate_content(
                model=IMAGE_MODEL,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                ),
            )
            # Save the generated image
            saved = False
            for part in getattr(response, "parts", []) or []:
                if hasattr(part, "inline_data") and part.inline_data:
                    with open(image_path, "wb") as f:
                        f.write(part.inline_data.data)
                    print(f"  Saved: {image_path}")
                    saved = True
                    break
            if not saved:
                print(f"  No image data in response for {filename}")
                _create_fallback_image(image_path, filename)
        except Exception as e:
            print(f"  Failed to generate {filename}: {e}")
            # Create a simple colored rectangle as fallback
            _create_fallback_image(image_path, filename)

    print(f"\nGenerated {len(scenarios)} test images in {output_dir}")


def _create_fallback_image(path, name):
    """Create a simple test image using PIL if Gemini fails."""
    try:
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (640, 480), color=(128, 128, 128))
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), f"Test: {name}", fill=(255, 255, 255))
        # Draw a simple face-like oval
        draw.ellipse([220, 100, 420, 380], outline=(255, 255, 255), width=2)
        draw.ellipse(
            [270, 180, 310, 220], outline=(255, 255, 255), width=2
        )  # left eye
        draw.ellipse(
            [330, 180, 370, 220], outline=(255, 255, 255), width=2
        )  # right eye
        draw.arc(
            [280, 260, 360, 320], 0, 180, fill=(255, 255, 255), width=2
        )  # mouth
        img.save(path, "JPEG")
        print(f"  Fallback image saved: {path}")
    except ImportError:
        print(f"  PIL not available for fallback image: {name}")


if __name__ == "__main__":
    generate_test_faces()
