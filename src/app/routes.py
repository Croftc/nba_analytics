from flask import Blueprint, render_template, request, jsonify, redirect, url_for
from analytics.utils import predict_today
# Import your analysis functions
# from analysis.model import your_model_function

main = Blueprint('main', __name__)

@main.route('/')
def index():
    # Assuming 'data' and 'model' are available here. Adjust as needed.
    data = predict_today()

    # Assuming data['matchups'] is a list of matchups
    return render_template('index.html', matchups=data['matchups'])

@main.route('/predict', methods=['POST'])
def predict():
    # Example: Get data from request and use it in your model
    data = request.json
    result = your_model_function(data)
    return jsonify(result)

@main.route('/update', methods=['POST'])
def update():
    # Re-run the predict_today function
    data = predict_today()
    return render_template('index.html', html_content=data['matchups'])
