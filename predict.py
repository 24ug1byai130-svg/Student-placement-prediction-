import pandas as pd
import joblib

model = joblib.load("model.pkl")

# Get student details
cgpa = float(input("Enter CGPA: "))
projects = int(input("Enter number of projects: "))
internships = int(input("Enter number of internships: "))
coding_score = int(input("Enter coding score: "))

# Create input data
student = pd.DataFrame(
    [[cgpa, projects, internships, coding_score]],
    columns=["CGPA", "Projects", "Internships", "CodingScore"]
)

# Make prediction
prediction = model.predict(student)

if prediction[0] == 1:
    print("\nStudent is likely to be PLACED")
else:
    print("\nStudent is likely to NOT be PLACED")