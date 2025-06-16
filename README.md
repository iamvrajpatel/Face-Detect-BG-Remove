# Face Detection & Background Removal API

## Overview
This API removes the background from an image and detects, crops, and resizes a face with margin using FastAPI. It utilizes:
- rembg for background removal
- retina-face for face detection

## Requirements
- Python 3.11.12
- Dependencies as listed in [requirements.txt](./requirements.txt)

## Installation

1. Install Python 3.11.12.
2. Clone the repository.
3. Navigate to the project directory:
   ```bash
   cd /Face-Detect-BG-Remove
   ```
4. (Optional) Create and activate a virtual environment:
   ```bash
   python3.11 -m venv env
   source env/bin/activate  # On Windows use: env\Scripts\activate
   ```
5. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the API
Start the server with:
```bash
uvicorn face_detect_bg_remove:app --host 0.0.0.0 --port 8000
```

## API Endpoint

### POST /process
- Accepts raw binary image data.
- Removes the background and detects a face.
- Crops the detected face with a margin and resizes it to 640x640.
- Returns the processed image in PNG format.

## Notes
- The application enforces a maximum of 4 concurrent processing threads.
- Adjust CORS settings as needed for production.
- Ensure all dependencies from [requirements.txt](./requirements.txt) are installed.

## Explanation of face_detect_bg_remove.py

This module implements the FastAPI application responsible for image processing by:
- Creating a FastAPI app with CORS enabled to allow cross-origin requests.
- Restricting concurrent processing to 4 threads using a semaphore to prevent overload.
- Defining the POST /process endpoint, which:
  - Accepts raw binary image data.
  - Uses the rembg library to remove the image background.
  - Detects faces in the background-removed image with RetinaFace.
  - Crops the detected face with a specified margin and resizes it to 640x640 pixels.
  - Returns the processed image as a PNG.
