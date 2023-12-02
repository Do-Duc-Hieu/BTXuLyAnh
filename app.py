from flask import Flask, render_template, request
import cv2
import numpy as np
from io import BytesIO

app = Flask(__name__)

# Hàm thực hiện các biến đổi ảnh dựa trên loại biến đổi được chọn
def apply_transformation(image, transformation_type):
    if transformation_type == 'negative':
        # Biến đổi âm bản: Lấy ảnh đảo ngược (255 - pixel)
        return 255 - image
    elif transformation_type == 'threshold':
        # Phân ngưỡng: Chuyển đổi thành ảnh nhị phân
        _, thresholded_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
        return thresholded_image
    elif transformation_type == 'gray':
        # Biến đổi mức xám theo hàm tuyến tính
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            return image
    elif transformation_type == 'logarithm':
        # Thêm một hằng số như 1 để tránh logarithm của số âm
        return np.uint8(255 / np.log(1 + np.max(image)) * (np.log(np.float32(image) + 1)))
    elif transformation_type == 'power':
        # Biến đổi tuyến tính theo hàm mũ
        return np.power(image, 2)
    elif transformation_type == 'histogram_equalization':
        return apply_histogram_equalization(image)
    elif transformation_type == 'edge_detection':
        return apply_edge_detection(image)
    elif transformation_type == 'sobel':
        return apply_sobel(image)
    elif transformation_type == 'morphology':
        return apply_morphology(image)
    elif transformation_type == 'blur':
        return apply_blur(image)
    elif transformation_type == 'otsu':
        return apply_otsu(image)
    elif transformation_type == 'dilation':
        return apply_dilation(image)
    elif transformation_type == 'neighbor_operator':
        return apply_neighbor_operator(image)
    elif transformation_type == 'opening':
        return apply_opening(image)
    elif transformation_type == 'closing':
        return apply_closing(image)

def apply_opening(image):
    kernel = np.ones((5, 5), np.uint8)
    opened_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return opened_image

def apply_closing(image):
    kernel = np.ones((5, 5), np.uint8)
    closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return closed_image

def apply_neighbor_operator(image):
    # Kernel láng giềng
    kernel = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]])

    # Áp dụng toán tử láng giềng
    result_image = cv2.filter2D(image, -1, kernel)

    return result_image

def apply_dilation(image):
    kernel = np.ones((5, 5), np.uint8)
    dilated_image = cv2.dilate(image, kernel, iterations=1)
    return dilated_image

def apply_otsu(image):
    _, otsu_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu_image

def apply_blur(image):
    return cv2.GaussianBlur(image, (5, 5), 2)

def apply_morphology(image):
    # Áp dụng phép co với kernel 5x5
    kernel = np.ones((5, 5), np.uint8)
    morph_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    return morph_image

def apply_sobel(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Áp dụng bộ lọc Sobel
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Kết hợp độ dốc theo hướng x và y
    sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)

    # Chuyển đổi về ảnh uint8
    sobel_combined = np.uint8(sobel_combined)

    return sobel_combined

def apply_histogram_equalization(image):
    if len(image.shape) == 3:
        # Chuyển đổi sang ảnh xám nếu nó không phải là ảnh xám
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    # Cân bằng lược đồ xám
    equalized_image = cv2.equalizeHist(image)

    return equalized_image

def apply_edge_detection(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Áp dụng phát hiện biên Canny
    edges = cv2.Canny(image, 50, 150)

    return edges

# Định nghĩa route '/' cho phép nhận cả request GET và POST
@app.route('/', methods=['GET', 'POST'])
def index():
    original_image_path = None
    transformed_image_path = None

    # Xử lý request POST khi người dùng tải lên ảnh
    if request.method == 'POST':
        file = request.files['image']
        if file:
            print(file.filename)
            # Đọc dữ liệu ảnh từ request và chuyển thành mảng numpy
            image_bytes = BytesIO(file.read())
            image = cv2.imdecode(np.frombuffer(image_bytes.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
            
            # Lấy loại biến đổi từ form
            transformation_type = request.form.get('transformation_type')
            
            # Áp dụng biến đổi và nhận ảnh chuyển đổi
            transformed_image = apply_transformation(image, transformation_type)

            # Lưu ảnh gốc và ảnh chuyển đổi tạm thời để hiển thị trong HTML
            original_image_path = "static/"+file.filename
            transformed_image_path = "static/transformed_image.jpg"
            
            cv2.imwrite(transformed_image_path, transformed_image)

    # Hiển thị trang HTML với đường dẫn của ảnh gốc và ảnh chuyển đổi
    return render_template('index.html', original_image_path=original_image_path, transformed_image_path=transformed_image_path)

# Chạy ứng dụng trên máy chủ Flask khi script được chạy
if __name__ == '__main__':
    app.run(debug=True)
