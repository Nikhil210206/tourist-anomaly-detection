## Tourist-Anomaly-Detection

# AI-Based Anomaly Detection Service

![Status](https://img.shields.io/badge/status-in_progress-yellow)

[cite_start]This module is the AI-Powered Safety Engine for the "Smart Tourist Safety Monitoring" application, part of our submission for the Smart India Hackathon 2025[cite: 1, 5]. Its purpose is to analyze real-time geospatial data from tourists to detect anomalous behavior that may indicate distress.

## Key Features

The model is being developed to detect and flag the following anomalies:
- Sudden location drop-offs
- Prolonged inactivity
- Deviation from planned routes
- Flagging missing, silent, or distress behavior for investigations

## Tech Stack

- **Backend API:** FastAPI 
- **Machine Learning:** TensorFlow, XGBoost 
- **Data Handling:** Pandas, NumPy
- **ML Utilities:** Scikit-learn

## Project Structure

```
tourist-anomaly-detection/
│
├── app/                  # FastAPI service code
│   ├── main.py           # API endpoints
│   ├── model.py          # ML model architecture
│   ├── preprocessing.py  # Data preprocessing functions
│   └── predictor.py        # Logic for loading model and predicting
│
├── data/                 # (Git Ignored) Local data storage
│   ├── raw/              # Original datasets
│   └── processed/        # Processed data for training
│
├── models/               # (Git Ignored) Saved model files
│
├── notebooks/            # Jupyter notebooks for R&D
│
├── scripts/              # Helper scripts
│
├── .gitignore            # Specifies files for Git to ignore
├── README.md             # This file
└── requirements.txt      # Python dependencies
```

## Getting Started

Follow these instructions to set up the development environment.

### 1. Prerequisites

- Python 3.9+
- Git

### 2. Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd tourist-anomaly-detection
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # Create the venv
    python -m venv venv

    # Activate on Windows
    venv\Scripts\activate

    # Activate on macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### 3. Data Setup (Important!)

This project does **not** track data files using Git. All datasets and trained models are managed via a shared cloud drive to keep the repository lightweight.

1.  Go to our shared Google Drive folder: `[Link to Your Shared Google Drive Folder]`
2.  Download the contents of the `data/` folder from the drive.
3.  Place them into your local `data/` directory. The structure should look like `data/raw/dataset_name.csv`.

## How to Run the Service

Once the setup is complete, you can run the FastAPI server locally.

1.  **Start the server:**
    ```bash
    uvicorn app.main:app --reload
    ```
    The `--reload` flag makes the server restart automatically after you make code changes.

2.  **Access the API docs:**
    Open your browser and navigate to `http://127.0.0.1:8000/docs`. You will see the auto-generated Swagger UI documentation for the API.

## API Usage

The service exposes the following endpoint for anomaly detection.

### `POST /detect-anomaly`

This endpoint accepts a sequence of recent GPS data points for a user and returns a decision on whether the sequence is anomalous.

**Request Body:**

```json
{
  "user_id": "user-123",
  "trajectory": [
    {"lat": 13.0827, "lon": 80.2707, "timestamp": "2025-09-05T10:00:00Z"},
    {"lat": 13.0830, "lon": 80.2710, "timestamp": "2025-09-05T10:00:10Z"},
    {"lat": 13.0833, "lon": 80.2713, "timestamp": "2025-09-05T10:00:20Z"}
  ]
}
```

**Success Response (Code 200):**

```json
{
  "user_id": "user-123",
  "anomaly": true,
  "anomaly_type": "PROLONGED_INACTIVITY",
  "confidence_score": 0.95
}
```

## Authors

- Vishaal Pillay 
- Nikhil Balamurugan 
