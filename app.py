from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import cv2
import numpy as np
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Dify Vision Tools API")

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 定义前端/Dify 传过来的数据格式
class CropRequest(BaseModel):
    image_base64: str
    target_w: int
    target_h: int

# --- 核心算法区 ---
def compute_energy(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return cv2.magnitude(grad_x, grad_y)

def energy_aware_crop(img, target_w, target_h):
    h, w = img.shape[:2]
    if target_w >= w or target_h >= h:
        return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

    energy = compute_energy(img)
    integral_energy = cv2.integral(energy)
    
    max_energy = -1.0
    best_x, best_y = 0, 0
    
    for y in range(h - target_h + 1):
        for x in range(w - target_w + 1):
            total_energy = (integral_energy[y + target_h, x + target_w] 
                          + integral_energy[y, x] 
                          - integral_energy[y, x + target_w] 
                          - integral_energy[y + target_h, x])
            if total_energy > max_energy:
                max_energy = total_energy
                best_x = x
                best_y = y
                
    return img[best_y : best_y + target_h, best_x : best_x + target_w]

# --- API 路由区 ---
@app.post("/api/crop")
async def process_crop(request: CropRequest):
    try:
        # 1. 处理 Base64 字符串 (清洗可能带有的前缀)
        b64_string = request.image_base64
        if "base64," in b64_string:
            b64_string = b64_string.split("base64,")[1]

        # 2. Base64 解码为 cv2 图像
        img_data = base64.b64decode(b64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Image decoding failed")

        # 3. 执行智能裁剪
        result_img = energy_aware_crop(img, request.target_w, request.target_h)

        # 4. 将结果图像重新编码为 Base64
        _, buffer = cv2.imencode('.jpg', result_img)
        result_base64 = base64.b64encode(buffer).decode('utf-8')

        return {
            "status": "success",
            "base64_data": result_base64,
            "markdown": f"![cropped_image](data:image/jpeg;base64,{result_base64})"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Vision API is running. Go to /docs to test it."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)