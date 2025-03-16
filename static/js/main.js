document.addEventListener('DOMContentLoaded', function() {
    const imageInput = document.getElementById('image-input');
    const measureBtn = document.getElementById('measure-btn');
    const originalImage = document.getElementById('original-image');
    const originalPlaceholder = document.querySelector('#original-image-container .placeholder');
    const resultImage = document.getElementById('result-image');
    const resultPlaceholder = document.querySelector('#result-image-container .placeholder');
    const measurementResult = document.getElementById('measurement-result');
    const loading = document.getElementById('loading');
    const errorMessage = document.getElementById('error-message');
    
    // 监听文件选择
    imageInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        
        if (file) {
            const reader = new FileReader();
            
            reader.onload = function(e) {
                originalImage.src = e.target.result;
                originalImage.style.display = 'block';
                originalPlaceholder.style.display = 'none';
                
                // 重置结果区域
                resultImage.style.display = 'none';
                resultPlaceholder.style.display = 'flex';
                measurementResult.textContent = '';
                errorMessage.textContent = '';
                
                // 启用测量按钮
                measureBtn.disabled = false;
            };
            
            reader.readAsDataURL(file);
        }
    });
    
    // 测量按钮点击事件
    measureBtn.addEventListener('click', function() {
        const file = imageInput.files[0];
        
        if (!file) {
            errorMessage.textContent = '请先选择图片';
            return;
        }
        
        // 显示加载动画
        loading.style.display = 'flex';
        errorMessage.textContent = '';
        
        // 创建表单数据
        const formData = new FormData();
        formData.append('image', file);
        
        // 发送请求
        fetch('/measure', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            loading.style.display = 'none';
            
            if (data.error) {
                errorMessage.textContent = data.error;
                return;
            }
            
            // 显示结果图片
            resultImage.src = data.result_image;
            resultImage.style.display = 'block';
            resultPlaceholder.style.display = 'none';
            
            // 构建测量结果HTML
            let resultHTML = '';
            
            // 添加宽度信息（如果有）
            if (data.width) {
                resultHTML += `<p>手指宽度: ${data.width} mm</p>`;
            }
            
            // 添加周长信息（如果有）
            if (data.circumference) {
                resultHTML += `<p>手指周长: ${data.circumference} mm</p>`;
            }
            
            // 添加戒指尺寸信息（如果有）
            if (data.ring_size) {
                resultHTML += `<p>戒指尺寸(中国标准): ${data.ring_size}</p>`;
            }
            
            // 如果没有测量结果
            if (resultHTML === '') {
                resultHTML = '<p>无法获取测量结果，请查看图像</p>';
            }
            
            // 更新测量结果区域
            measurementResult.innerHTML = resultHTML;
        })
        .catch(error => {
            loading.style.display = 'none';
            errorMessage.textContent = '处理请求时出错: ' + error.message;
            console.error('Error:', error);
        });
    });
});
