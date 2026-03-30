import os
import sys
import base64
import cv2
from flask import Flask, request, jsonify, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(BASE_DIR, 'project')
os.makedirs(os.path.join(PROJECT_DIR, 'static'), exist_ok=True)
sys.path.insert(0, PROJECT_DIR)

from parser import parse_floor_plan, find_rooms, find_doors, find_windows, get_edge_preview, find_furniture
from geometry import build_geometry
from model3d import generate_3d_model
from materials import recommend_materials, build_cost_report
from explain import generate_explanation

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.after_request
def add_no_cache_headers(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/')
def index():
    return send_from_directory(PROJECT_DIR, 'index.html')

@app.route('/process', methods=['POST'])
def process():
    upload = request.files.get('image') or request.files.get('file')
    if upload is None:
        return jsonify({'error': 'No image uploaded'}), 400

    input_path = os.path.join(PROJECT_DIR, 'input.png')
    upload.save(input_path)

    walls = parse_floor_plan(input_path)
    doors = find_doors(input_path)
    windows = find_windows(input_path)
    rooms = find_rooms(input_path)
    furniture = find_furniture(input_path)
    points, edges = build_geometry(walls)
    model_3d = generate_3d_model(edges)
    materials = recommend_materials(edges)
    cost_report = build_cost_report(materials)
    explanation = generate_explanation(materials)

    edge_preview = get_edge_preview(input_path)
    edge_preview_base64 = None
    edge_preview_path = None
    if edge_preview is not None:
        save_path = os.path.join(PROJECT_DIR, 'static', 'edges.png')
        cv2.imwrite(save_path, edge_preview)
        edge_preview_path = '/static/edges.png'
        _, buf = cv2.imencode('.png', edge_preview)
        edge_preview_base64 = base64.b64encode(buf).decode('ascii')

    return jsonify({
        'edgePreviewPath': edge_preview_path,
        'walls': walls,
        'doors': doors,
        'windows': windows,
        'rooms': rooms,
        'furniture': furniture,
        'model3D': model_3d,
        'materials': materials,
        'costReport': cost_report,
        'explanation': explanation,
        'edgePreview': f'data:image/png;base64,{edge_preview_base64}' if edge_preview_base64 else None
    })

@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory(PROJECT_DIR, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
