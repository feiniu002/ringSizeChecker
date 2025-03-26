import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import uuid
# 导入 size_check.py 中的 measure_finger_size 函数
from size_check import measure_finger_size


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# 确保上传文件夹存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/measure', methods=['POST'])
def measure():
    if 'image' not in request.files:
        return jsonify({'error': '没有上传文件'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    if file and allowed_file(file.filename):
        # 生成唯一文件名
        filename = secure_filename(str(uuid.uuid4()) + os.path.splitext(file.filename)[1])
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 读取图像
        image = cv2.imread(filepath)
        if image is None:
            return jsonify({'error': '无法读取上传的图像'}), 400
        
        # 调用修改后的 measure_finger_size 函数，现在它返回图像和测量结果
        result_image, width_mm, circumference_mm, ring_size_cn = measure_finger_size(image)
        
        if result_image is not None:
            # 保存结果图像
            result_filename = 'result_' + filename
            result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
            cv2.imwrite(result_path, result_image)
            
            # 准备响应数据
            response_data = {
                'success': True,
                'result_image': '/' + result_path
            }
            
            # 添加测量结果（如果有）
            if width_mm is not None:
                response_data['width'] = f"{width_mm:.2f}"
            if circumference_mm is not None:
                response_data['circumference'] = f"{circumference_mm:.2f}"
            if ring_size_cn is not None:
                response_data['ring_size'] = f"{ring_size_cn}"
            
            return jsonify(response_data)
        else:
            return jsonify({'error': '无法检测到戒指或手指'}), 400
    
    return jsonify({'error': '不支持的文件类型'}), 400

if __name__ == '__main__':
    app.run(debug=True)
