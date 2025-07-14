from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("fake_review_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route('/')
def home():
    print("Home route triggered!")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    review_vector = vectorizer.transform([review])
    prediction = model.predict(review_vector)[0]  # prediction is 0 or 1

    if prediction == 1:
        result = "This review is genuine ✅"
    else:
        result = "This review is fake ❌"

    return render_template('index.html', prediction=result)


if __name__ == '__main__':
    app.run(debug=True)
