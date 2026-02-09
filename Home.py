from flask import Flask, send_file, redirect
import os

app = Flask(__name__, static_folder='assets', static_url_path='/assets')

@app.route('/')
def landing_page():
    """Serve the HTML landing page"""
    return send_file('index.html')

@app.route('/optimizer')
def optimizer():
    """Redirect to Streamlit app"""
    return redirect('http://localhost:8501')

@app.route('/assets/<path:filename>')
def serve_assets(filename):
    """Serve static files like images/gifs"""
    return send_file(f'assets/{filename}')

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🎯 PORTFOLIO OPTIMIZER - LANDING PAGE")
    print("="*50)
    print("📍 Landing page: http://localhost:5000")
    print("📊 Streamlit app: http://localhost:8501")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=True)