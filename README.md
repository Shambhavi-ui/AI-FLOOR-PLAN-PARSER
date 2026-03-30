AI Floor Plan to 3D Converter (Web3 Enabled)

An AI-based system that reads a 2D floor plan, reconstructs it into a 3D model, and provides basic material cost insights — with an added layer of decentralized storage using IPFS.

---

Contents

- Overview
- Features
- Tech Stack
- How It Works
- Getting Started
- Project Structure
- Web3 Integration
- Limitations
- Future Work

---

Overview

The goal of this project is to simplify architectural visualization. Instead of manually creating models in CAD tools, users can upload a floor plan image and get a 3D representation automatically.

Along with reconstruction, the system also provides a basic cost breakdown and explanation of the structure.
As an extension, generated models are stored on IPFS to demonstrate decentralized storage.

---

Features

- Floor plan parsing using computer vision
- Automatic room and wall detection
- 2D → 3D model generation
- Basic material cost estimation
- Simple explanation of structure
- IPFS-based storage for generated models

---

Tech Stack

- Backend: Python (Flask)
- Computer Vision: OpenCV
- 3D Rendering: Three.js
- Storage: IPFS
- Frontend: HTML, CSS, JavaScript

---

How It Works

1. Upload a floor plan image
2. Detect walls and enclosed regions
3. Convert geometry into structured layout
4. Extrude layout into 3D model
5. Estimate material cost
6. Store model on IPFS and generate link

---

Getting Started

Prerequisites

- Python 3.x
- IPFS installed locally

---

Installation

pip install -r requirements.txt

---

Run IPFS

ipfs daemon

---

Start Application

python backend/app.py

Open in browser:

http://127.0.0.1:5000

---

Project Structure

backend/        → Flask backend + processing logic  
frontend/       → UI and viewer  
blockchain/     → smart contract placeholder  
static/         → generated models and assets  
uploads/        → input images  

---

Web3 Integration

This project uses IPFS for storing generated 3D models.

Each model:

- is uploaded to IPFS
- receives a unique content hash (CID)
- can be accessed via gateway

This ensures:

- decentralized storage
- tamper-resistant files
- easy sharing

Rendering is handled separately using Three.js inside the application.

---

Limitations

- Works best with clean, digital floor plans
- Room classification is basic
- No structural validation yet
- Web3 integration is limited to storage (no deployed contract)

---

Future Work

- Better room classification
- Improved geometry accuracy
- Structural validation rules
- Smart contract integration
- Multi-floor building support

---

Notes

This project was built during a hackathon, so the focus was on completing the full pipeline within limited time. The current version demonstrates the core idea and can be extended further.
