from flask import Flask, render_template, jsonify
import subprocess
import sys

app = Flask(__name__)

@app.route('/')
def index():
    # Serve the HTML page with the button
    return render_template('index.html')

@app.route('/start-applicationA', methods=['POST'])
def start_applicationA():
    # Start the application.py script
    python_executable = sys.executable  
    application_script = "applicationA.py"
    
    try:
        # Run application.py in the background
        subprocess.Popen([python_executable, application_script])
        return jsonify({"status": "Application started successfully."}), 200
    except Exception as e:
        return jsonify({"status": "Error starting application.", "error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
