from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib

app = FastAPI()

# Allow frontend to communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load trained ML model
model = joblib.load("model.pkl")


@app.get("/")
def home():
    return {
        "message": "Student Placement Prediction API is running"
    }


@app.get("/predict")
def predict(
    cgpa: float,
    projects: int,
    internships: int,
    coding_score: int
):
    student = pd.DataFrame(
        [[cgpa, projects, internships, coding_score]],
        columns=[
            "CGPA",
            "Projects",
            "Internships",
            "CodingScore"
        ]
    )

    prediction = model.predict(student)

    if prediction[0] == 1:
        result = "Placed"
    else:
        result = "Not Placed"

    return {
        "prediction": result
    }
