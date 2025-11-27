"""
Simple Flask server to serve a webcam client page and accept base64 frames for prediction.
Client uses getUserMedia to capture frames and POST to /api/predict (base64 JSON).
"""
from flask import Flask, request, jsonify, render_template_string
import base64
import io
from PIL import Image
import os

from food_inference import FoodClassifier

app = Flask(__name__)

# Load classifier once
classifier = FoodClassifier(model_path='food_model_efficientnet.pth')

# Simple webcam client page (same-origin, posts to /api/predict)
WEBCAM_PAGE = '''
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>Webcam Food Classifier</title>
    <style>
      body { font-family: Arial, sans-serif; max-width: 900px; margin: 30px auto; }
      video { border: 1px solid #ccc; }
      #result { margin-top: 10px; }
    </style>
  </head>
  <body>
    <h2>Webcam Food Classifier</h2>
    <video id="video" width="480" height="360" autoplay playsinline></video>
    <div>
      <button id="start">Start</button>
      <button id="stop">Stop</button>
    </div>
    <div id="result">Result: <span id="label">-</span></div>

    <script>
      const video = document.getElementById('video');
      const startBtn = document.getElementById('start');
      const stopBtn = document.getElementById('stop');
      const labelSpan = document.getElementById('label');
      let stream = null;
      let capturing = false;

      async function startCamera() {
        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        video.srcObject = stream;
      }

      function stopCamera() {
        if (stream) {
          stream.getTracks().forEach(t => t.stop());
          stream = null;
        }
      }

      async function captureLoop() {
        capturing = true;
        const canvas = document.createElement('canvas');
        canvas.width = 224; canvas.height = 224;
        const ctx = canvas.getContext('2d');

        while (capturing) {
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          const dataUrl = canvas.toDataURL('image/jpeg');
          const base64 = dataUrl.split(',')[1];

          try {
            const resp = await fetch('/api/predict', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ image: base64 })
            });
            const data = await resp.json();
            if (data.error) {
              labelSpan.textContent = 'Error';
            } else {
              labelSpan.textContent = `${data.top_prediction} (${data.top_confidence})`;
            }
          } catch (err) {
            labelSpan.textContent = 'Request failed';
          }

          await new Promise(r => setTimeout(r, 800)); // ~1.25 FPS to reduce server load
        }
      }

      startBtn.onclick = async () => {
        await startCamera();
        captureLoop();
      };
      stopBtn.onclick = () => { capturing = false; stopCamera(); };
    </script>
  </body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(WEBCAM_PAGE)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        img_bytes = base64.b64decode(data['image'])
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        # Save temporarily to disk (re-use classifier API)
        tmp = 'tmp_webcam_client.jpg'
        img.save(tmp)

        result = classifier.predict(tmp, top_k=1)

        try:
            os.remove(tmp)
        except Exception:
            pass

        # Return top prediction and top confidence
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print('Starting Flask webcam server on http://0.0.0.0:5000')
    app.run(host='0.0.0.0', port=5000, debug=False)
