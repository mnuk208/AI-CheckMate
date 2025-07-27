# app.py (New Main File)
# This is now the main entry point for your application.
# To run your app, you will now execute: python app.py

from flask import Flask, render_template

# Import the blueprints from your route files
from detect import detect_api
from humanise import humanize_api

app = Flask(__name__)

# Register the blueprints to connect them to the main app
app.register_blueprint(detect_api)
app.register_blueprint(humanize_api)

@app.route('/')
def home():
    """Renders the main HTML page."""
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
