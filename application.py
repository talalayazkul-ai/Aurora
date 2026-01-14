from flask import Flask, request, render_template_string
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

application = Flask(__name__)
app = application

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Aurora - Student Performance Predictor</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 600px; margin: 50px auto; padding: 20px; }
        h1 { color: #333; }
        form { background: #f5f5f5; padding: 20px; border-radius: 8px; }
        label { display: block; margin-top: 10px; font-weight: bold; }
        select, input { width: 100%; padding: 8px; margin-top: 5px; border: 1px solid #ddd; border-radius: 4px; }
        button { margin-top: 20px; padding: 12px 24px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background: #0056b3; }
        .result { margin-top: 20px; padding: 20px; background: #d4edda; border-radius: 8px; text-align: center; }
        .result h2 { color: #155724; margin: 0; }
    </style>
</head>
<body>
    <h1>Student Math Score Predictor</h1>
    <form method="POST" action="/predict">
        <label>Gender</label>
        <select name="gender" required>
            <option value="male">Male</option>
            <option value="female">Female</option>
        </select>

        <label>Race/Ethnicity</label>
        <select name="race_ethnicity" required>
            <option value="group A">Group A</option>
            <option value="group B">Group B</option>
            <option value="group C">Group C</option>
            <option value="group D">Group D</option>
            <option value="group E">Group E</option>
        </select>

        <label>Parental Level of Education</label>
        <select name="parental_level_of_education" required>
            <option value="some high school">Some High School</option>
            <option value="high school">High School</option>
            <option value="some college">Some College</option>
            <option value="associate's degree">Associate's Degree</option>
            <option value="bachelor's degree">Bachelor's Degree</option>
            <option value="master's degree">Master's Degree</option>
        </select>

        <label>Lunch</label>
        <select name="lunch" required>
            <option value="standard">Standard</option>
            <option value="free/reduced">Free/Reduced</option>
        </select>

        <label>Test Preparation Course</label>
        <select name="test_preparation_course" required>
            <option value="none">None</option>
            <option value="completed">Completed</option>
        </select>

        <label>Reading Score (0-100)</label>
        <input type="number" name="reading_score" min="0" max="100" required>

        <label>Writing Score (0-100)</label>
        <input type="number" name="writing_score" min="0" max="100" required>

        <button type="submit">Predict Math Score</button>
    </form>

    {% if result is not none %}
    <div class="result">
        <h2>Predicted Math Score: {{ result }}</h2>
    </div>
    {% endif %}
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE, result=None)


@app.route("/predict", methods=["POST"])
def predict():
    data = CustomData(
        gender=request.form.get("gender"),
        race_ethnicity=request.form.get("race_ethnicity"),
        parental_level_of_education=request.form.get("parental_level_of_education"),
        lunch=request.form.get("lunch"),
        test_preparation_course=request.form.get("test_preparation_course"),
        reading_score=int(request.form.get("reading_score")),
        writing_score=int(request.form.get("writing_score")),
    )

    df = data.get_data_as_dataframe()
    pipeline = PredictPipeline()
    result = pipeline.predict(df)

    return render_template_string(HTML_TEMPLATE, result=round(result[0], 2))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
