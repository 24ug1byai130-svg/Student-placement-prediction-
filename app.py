from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()

model = joblib.load("model.pkl")


@app.get("/")
def home():
    return {"message": "Student Placement Prediction API is running"}


@app.get("/predict")
def predict(
    cgpa: float,
    projects: int,
    internships: int,
    coding_score: int
):
    student = pd.DataFrame(
        [[cgpa, projects, internships, coding_score]],
        columns=["CGPA", "Projects", "Internships", "CodingScore"]
    )

    prediction = model.predict(student)

    if prediction[0] == 1:
        result = "Placed"
    else:
        result = "Not Placed"

    return {"prediction": result}