#!/usr/bin/env python3
"""
Simple test script for GPT-Image API.
Tests the OpenAI gpt-image-1.5 model directly.
"""

import os
import base64
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def main():
    # Initialize client (uses OPENAI_API_KEY from environment)
    client = OpenAI()
    
    prompt = "A cute robot toy with big eyes, standing on a white background"
    
    print(f"[INFO] Generating image with gpt-image-1.5...")
    print(f"[INFO] Prompt: {prompt}")
    
    # Call the API - following official docs
    result = client.images.generate(
        model="gpt-image-1.5",
        prompt=prompt,
    )
    
    # Get base64 image data
    image_base64 = result.data[0].b64_json
    image_bytes = base64.b64decode(image_base64)
    
    # Save the image
    output_path = "test_robot.png"
    with open(output_path, "wb") as f:
        f.write(image_bytes)
    
    print(f"[SUCCESS] Image saved to: {output_path}")
    print(f"[INFO] Image size: {len(image_bytes)} bytes")

if __name__ == "__main__":
    main()
