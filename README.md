# PIP Image Captioning

一个集成图像处理、AI标注和标签管理的一站式解决方案，基于 Qwen2.5-VL-7B-Instruct 本地视觉模型。



<img width="1931" height="1889" alt="image" src="https://github.com/user-attachments/assets/3d4336f3-fa21-4134-987a-9d156c6c4b34" />
<img width="1932" height="1824" alt="image" src="https://github.com/user-attachments/assets/bc7d9b87-3223-4b58-a435-9d4eadd67f4e" />


## 更新日志

- **0830**：增加了对于自定义PE模板的支持

<img width="1926" height="1743" alt="a806fd34da082d1bd6fc4e50dc6bca67" src="https://github.com/user-attachments/assets/7be57bf4-9b56-40e7-a427-5b2d09268c18" />
<img width="1803" height="1770" alt="a0e4618aba0c1c3a3cc20db9a41fb3ed" src="https://github.com/user-attachments/assets/b4ecbe67-af80-440f-9b50-8a4057ecfb7a" />



## 功能特性

### 🖼️ 图像预处理
- **批量格式转换**：支持 JPG、PNG、WebP、BMP、GIF、TIFF 等多种格式互转
- **智能尺寸调整**：可配置最大尺寸限制，自动等比例缩放
- **批量重命名**：统一命名规则，便于管理
- **格式优化**：转换为训练友好的格式

### 🤖 AI 图像标注（四种模式）
- **自然语言推理**：生成详细的英文段落描述，涵盖主体、背景、构图、风格等全方位分析
- **标签推理**：输出简洁的逗号分隔标签，适合训练数据集
- **混合标签**：结合概述段落和关键标签，平衡详细度和简洁性


### 🌐 双语支持
- **English（默认）**：英文输出，适合国际化数据集
- **中文**：中文输出，适合本土化应用
- **灵活切换**：单张和批量处理都支持语言选择

### 🏷️ 标签管理
- **批量查看**：浏览文件夹中所有标注文件内容
- **文本替换**：批量查找替换特定词汇或短语
- **触发词添加**：为所有标注文件统一添加触发词
- **编码转换**：UTF-8 编码规范化处理

## 使用方法

### 单张图片标注
1. 点击"单张标注"选项卡
2. 选择图片文件（支持拖拽上传）
3. 选择标注模式和输出语言
4. 点击"开始标注"，实时查看结果

### 批量图片处理
1. 点击"单张标注"选项卡
2. 输入包含图片的文件夹路径
3. 选择标注模式和语言
4. 点击"开始标注"，查看实时进度

### 图像预处理
1. 点击"图像处理"选项卡
2. 输入源文件夹路径
3. 配置转换参数（格式、尺寸、重命名）
4. 点击"开始转换"

### 标签管理
1. 点击"标签管理"选项卡
2. 输入标注文件所在文件夹
3. 使用查看、替换、添加触发词等功能

## 安装步骤

### 1. 环境要求
- Python 3.8+
- CUDA 支持的 GPU（推荐）
- 8GB+ 显存（用于加载 7B 模型）

### 2. 克隆项目
```bash
git clone https://github.com/chenpipi0807/PIP-Image-Captioning.git
cd PIP-Image-Captioning
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 启动应用（更推荐使用venv）
```bash
python app.py
```

访问 `http://localhost:5000` 开始使用。

## 模型下载与配置

### 模型获取
本工具使用 **Qwen2.5-VL-7B-Instruct** 模型，可从以下渠道下载：

#### Hugging Face（推荐）
```bash
# 使用 git-lfs 下载
git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
```

#### ModelScope（国内用户）
```bash
# 使用 modelscope 库下载
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('qwen/Qwen2.5-VL-7B-Instruct', local_dir='./Qwen2.5-VL-7B-Instruct')"
```

### 目录结构
下载完成后，确保模型目录结构如下：
```
PIP-Image-Captioning/
├── Qwen2.5-VL-7B-Instruct/
│   ├── config.json
│   ├── model.safetensors.index.json
│   ├── model-00001-of-00005.safetensors
│   ├── model-00002-of-00005.safetensors
│   ├── model-00003-of-00005.safetensors
│   ├── model-00004-of-00005.safetensors
│   ├── model-00005-of-00005.safetensors
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── vocab.json
│   ├── merges.txt
│   ├── preprocessor.json
│   └── generation_config.json
├── app.py
├── requirements.txt
├── prompts.json
└── templates/
    └── index.html
```

### 自定义模型路径
如果模型放在其他位置，可设置环境变量：
```bash
export LOCAL_VLM_PATH="/path/to/your/Qwen2.5-VL-7B-Instruct"
python app.py
```

## 技术栈
- **后端**：Flask 2.3.3 + PyTorch + Transformers
- **前端**：Bootstrap 5.1.3 + Font Awesome
- **模型**：Qwen2.5-VL-7B-Instruct（本地推理）
- **加速**：Accelerate + GPU 优化

## 注意事项
- 首次启动需要加载 7B 模型，耗时约 20-30 秒
- 建议使用 GPU 进行推理，CPU 模式较慢
- 批量处理时会自动覆盖同名的标注文件
- 支持的图片格式：JPG、JPEG、PNG、WebP、BMP、GIF、TIFF、JFIF、HEIC、HEIF、AVIF

## 许可证
本项目遵循相应的开源许可证，具体请查看 LICENSE 文件。
