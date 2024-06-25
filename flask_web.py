from flask import Flask, render_template, Response, redirect, url_for
import subprocess
import cv2

app = Flask(__name__)
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        # Read the camera frame
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/run_script')
def run_script():
    # Replace this with the command you want to run
    # For example, running a simple shell command
    subprocess.Popen(['echo', 'Script is running'])
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(host='127.0.0.1', debug=True, port=5000)
