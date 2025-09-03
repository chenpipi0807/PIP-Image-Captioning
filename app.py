import os
import json
import logging
import time
from pathlib import Path
import shutil
from PIL import Image
import uuid
from flask import Flask, render_template, request, jsonify, send_from_directory
import threading
import requests
import random
import base64
import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
from datetime import datetime
from qwen_vl_utils import process_vision_info
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 全局模型实例（单例模式）
global_model_instance = None

def get_model_instance():
    global global_model_instance
    if global_model_instance is None:
        global_model_instance = LocalQwenVLTool()
    return global_model_instance

class LocalQwenVLTool:
    def __init__(self, model_path: str = None):
        self.questions = {}
        self.model = None
        self.tokenizer = None
        self.processor = None
        # Only two options: explicit env var or default folder 'Qwen2.5-VL-7B-Instruct'. No fallback.
        env_override = os.environ.get('LOCAL_VLM_PATH')
        default_model = Path(__file__).parent / 'Qwen2.5-VL-7B-Instruct'
        self.model_path = env_override or str(default_model)
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"模型目录不存在: {self.model_path}. 请设置 LOCAL_VLM_PATH 指向已下载的 Qwen2.5-VL-7B-Instruct 目录，"
                f"或将其放到 {default_model}。"
            )
        logger.info(f"Using local VLM: {self.model_path}")

        # Load prompts
        self.load_prompts()
        self.load_user_prompts()

        # Load model on GPU (Qwen2.5-VL 推理建议使用 GPU)
        self.load_model()


    def load_prompts(self):
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            prompts_path = os.path.join(script_dir, 'prompts.json')
            with open(prompts_path, 'r', encoding='utf-8') as f:
                self.prompts = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("prompts.json 未找到。请在应用根目录提供 prompts.json 以定义所有系统提示词与模式。")
        except Exception as e:
            raise RuntimeError(f"加载 prompts.json 出错: {e}")
    
    def load_user_prompts(self):
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            user_prompts_path = os.path.join(script_dir, 'user-prompts.json')
            with open(user_prompts_path, 'r', encoding='utf-8') as f:
                self.user_prompts = json.load(f)
        except FileNotFoundError:
            # Create default user prompts file if it doesn't exist
            self.user_prompts = {"custom_templates": []}
            self.save_user_prompts()
        except Exception as e:
            logger.error(f"加载 user-prompts.json 出错: {e}")
            self.user_prompts = {"custom_templates": []}
    
    def save_user_prompts(self):
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            user_prompts_path = os.path.join(script_dir, 'user-prompts.json')
            with open(user_prompts_path, 'w', encoding='utf-8') as f:
                json.dump(self.user_prompts, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存 user-prompts.json 出错: {e}")

    def load_model(self):
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("需要 GPU (CUDA) 才能高效运行 Qwen2.5-VL。本机未检测到可用 GPU。")

            logger.info("Loading Qwen2.5-VL-7B-Instruct model (GPU, Transformers)")
            logger.info(f"GPU信息: {torch.cuda.get_device_name(0)} - 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True, use_fast=False
            )
            self.processor = AutoProcessor.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            # 使用其他优化选项替代Flash Attention
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,  # 使用bfloat16可能更快
                trust_remote_code=True,
                low_cpu_mem_usage=True,  # 减少CPU内存使用
                use_cache=True  # 启用KV缓存
            )
            
            # 预热模型
            self.model.eval()
            torch.cuda.empty_cache()
            
            logger.info("Qwen2.5-VL-7B-Instruct loaded successfully on GPU.")
            logger.info(f"模型设备: {next(self.model.parameters()).device}")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            self.model = None
            self.tokenizer = None
            self.processor = None

    def is_loaded(self):
        """检查模型是否已成功加载"""
        return (self.model is not None and 
                self.tokenizer is not None and 
                self.processor is not None)

    def _build_inputs(self, image_path, prompt_text):
        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        return inputs

    def generate_response(self, image_path, prompt_text, max_tokens=512):
        if self.model is None or self.tokenizer is None or self.processor is None:
            return "错误：本地视觉模型未成功加载（需要 GPU 且模型路径需有效）。"

        try:
            start_time = time.time()
            
            inputs = self._build_inputs(image_path, prompt_text)
            
            # 确保所有输入都正确移动到GPU并且数据类型正确
            inputs = {k: v.to(self.model.device, non_blocking=True) for k, v in inputs.items()}
            
            # 检查输入是否有效
            if 'input_ids' in inputs and inputs['input_ids'].numel() == 0:
                return "错误：输入处理失败，请检查图片和提示词"
            
            prep_time = time.time() - start_time
            logger.info(f"输入预处理耗时: {prep_time:.2f}s")
            
            gen_start = time.time()
            with torch.inference_mode():
                # 清理显存
                torch.cuda.empty_cache()
                
                # 使用更优化的生成参数
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    use_cache=True,  # 启用KV缓存加速
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            gen_time = time.time() - gen_start
            logger.info(f"模型生成耗时: {gen_time:.2f}s")
            
            # 截取新生成部分
            new_tokens = output_ids[:, inputs["input_ids"].shape[1]:]
            text = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
            
            total_time = time.time() - start_time
            logger.info(f"总推理耗时: {total_time:.2f}s")
            
            return text.strip()
        except Exception as e:
            logger.error(f"推理失败: {e}")
            # 清理显存
            torch.cuda.empty_cache()
            return f"错误：模型推理失败 - {str(e)}"

    def annotate_image(self, image_path, selected_types, language: str = 'en'):
        try:
            # System prompt from prompts.json -> modes.natural.<lang>.system (fallbacks retained)
            system_prompt = (
                self.prompts.get('modes', {}).get('natural', {}).get(language, {}).get('system')
                or self.prompts.get('modes', {}).get('natural', {}).get('en', {}).get('system')
                or self.prompts.get('modes', {}).get('natural', {}).get('system')
                or self.prompts.get('natural_system', '')
            )
            # Single-paragraph natural description per system prompt only
            return self.generate_response(image_path, system_prompt, max_tokens=512)
        except Exception as e:
            logger.error(f"annotate_image 出错: {e}")
            return None

    def refine_prompt(self, annotation_text, image_path):
        try:
            # System prompt from prompts.json -> modes.natural.system (fallback to legacy key)
            system_prompt = (
                self.prompts.get('modes', {}).get('natural', {}).get('system')
                or self.prompts.get('natural_system', '')
            )
            # refine instruction optional; fallback to built-in default for compatibility
            refine_instruction = (
                self.prompts.get('modes', {}).get('natural', {}).get('refine_instruction')
                or self.prompts.get('refine_instruction',
                    "请将上述分析内容整理为一段自然语言中文描述，按照内容主体、外观细节、姿态动作、场景环境、氛围背景、色彩构图、机位与镜头、风格特点的顺序，以逗号分隔短语，流畅连贯，不包含分类信息。"
                )
            )
            full_prompt = f"{system_prompt}\n\n基于以下分析：{annotation_text}\n\n{refine_instruction}"
            return self.generate_response(image_path, full_prompt, max_tokens=512)
        except Exception as e:
            logger.error(f"refine_prompt 出错: {e}")
            return None

    def annotate_image_short(self, image_path, language: str = 'en'):
        try:
            # tags prompt from modes.tags.<lang>.system (fallbacks retained)
            tag_prompt = (
                self.prompts.get('modes', {}).get('tags', {}).get(language, {}).get('system')
                or self.prompts.get('modes', {}).get('tags', {}).get('en', {}).get('system')
                or self.prompts.get('modes', {}).get('tags', {}).get('zh', {}).get('system')
                or self.prompts.get('modes', {}).get('tags', {}).get('system')
                or self.prompts.get('tags')
            )
            return self.generate_response(image_path, tag_prompt or '', max_tokens=512)
        except Exception as e:
            logger.error(f"annotate_image_short 出错: {e}")
            return None

    def annotate_image_mixed(self, image_path, language: str = 'en'):
        try:
            # mixed now uses a unified system prompt at modes.mixed.<lang>.system
            mixed_prompt = (
                self.prompts.get('modes', {}).get('mixed', {}).get(language, {}).get('system')
                or self.prompts.get('modes', {}).get('mixed', {}).get('en', {}).get('system')
                or self.prompts.get('modes', {}).get('mixed', {}).get('zh', {}).get('system')
                or self.prompts.get('modes', {}).get('mixed', {}).get('system')
                or self.prompts.get('description')
            )
            return self.generate_response(image_path, mixed_prompt or '', max_tokens=512)
        except Exception as e:
            logger.error(f"annotate_image_mixed 出错: {e}")
            return None

    def annotate_image_custom(self, image_path, template_id: str):
        try:
            # Find custom template by ID
            template = None
            for t in self.user_prompts.get('custom_templates', []):
                if t.get('id') == template_id:
                    template = t
                    break
            
            if not template:
                return f"错误：未找到模板 ID: {template_id}"
            
            prompt = template.get('prompt', '')
            return self.generate_response(image_path, prompt, max_tokens=1024)
        except Exception as e:
            logger.error(f"annotate_image_custom 出错: {e}")
            return None

    def get_video_info(self, video_path):
        """获取视频基本信息"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"无法打开视频文件: {video_path}")
            
            # 获取视频属性
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            return {
                'fps': fps,
                'duration': duration,
                'width': width,
                'height': height,
                'frame_count': frame_count,
                'resolution': f"{width}*{height}"
            }
        except Exception as e:
            logger.error(f"获取视频信息失败: {e}")
            return None

    def _build_video_inputs(self, video_path, prompt_text, fps=None, max_pixels=None):
        """构建视频输入"""
        try:
            # 自动获取视频信息
            video_info = self.get_video_info(video_path)
            if video_info:
                # 使用视频原始帧率，但限制在合理范围内
                if fps is None:
                    fps = min(max(video_info['fps'], 0.5), 2.0)
                
                # 自动调整分辨率
                if max_pixels is None:
                    original_pixels = video_info['width'] * video_info['height']
                    if original_pixels > 1280*720:
                        max_pixels = 1280*720  # 降低分辨率以节省显存
                    else:
                        max_pixels = original_pixels
                
                logger.info(f"视频信息: {video_info['width']}x{video_info['height']}, {video_info['fps']:.1f}fps, {video_info['duration']:.1f}s")
                logger.info(f"处理参数: fps={fps:.1f}, max_pixels={max_pixels}")
            
            # 确保使用绝对路径
            abs_video_path = os.path.abspath(video_path)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": abs_video_path,  # 直接使用绝对路径，不加file://前缀
                        },
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            
            # 应用聊天模板
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # 处理视觉信息（简化API调用）
            image_inputs, video_inputs = process_vision_info(messages)
            
            # 使用processor处理输入
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                fps=fps,
                max_pixels=max_pixels,
                padding=True,
                return_tensors="pt",
            )
            
            return inputs
        except Exception as e:
            logger.error(f"视频输入构建失败: {e}")
            raise

    def process_video(self, video_path, language='zh', fps=None, max_pixels=None, max_tokens=1024):
        """处理视频并生成分析结果（自动检测视频属性）"""
        try:
            start_time = time.time()
            
            # 获取视频分析提示词
            video_prompt = (
                self.prompts.get('modes', {}).get('video', {}).get(language, {}).get('system')
                or self.prompts.get('modes', {}).get('video', {}).get('zh', {}).get('system')
                or "请详细分析这个视频的内容，包括主要场景、人物动作、时间变化和关键事件。"
            )
            
            inputs = self._build_video_inputs(video_path, video_prompt, fps, max_pixels)
            
            # 确保所有输入都正确移动到GPU
            inputs = {k: v.to(self.model.device, non_blocking=True) for k, v in inputs.items()}
            
            prep_time = time.time() - start_time
            logger.info(f"视频输入预处理耗时: {prep_time:.2f}s")
            
            gen_start = time.time()
            with torch.inference_mode():
                # 清理显存
                torch.cuda.empty_cache()
                
                # 生成视频分析
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    use_cache=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            gen_time = time.time() - gen_start
            logger.info(f"视频分析生成耗时: {gen_time:.2f}s")
            
            # 截取新生成部分
            new_tokens = output_ids[:, inputs["input_ids"].shape[1]:]
            text = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
            
            total_time = time.time() - start_time
            logger.info(f"视频处理总耗时: {total_time:.2f}s")
            
            return text.strip(), total_time
        except Exception as e:
            logger.error(f"视频处理失败: {e}")
            torch.cuda.empty_cache()
            return f"错误：视频处理失败 - {str(e)}", 0

# 全局变量用于存储处理状态
processing_status = {}

def resize_image(input_path, output_path, max_size=768):
    try:
        with Image.open(input_path) as image:
            width, height = image.size

            if width > height:
                new_width = max_size
                new_height = int((height / width) * max_size)
            else:
                new_height = max_size
                new_width = int((width / height) * max_size)

            if image.mode in ('RGBA', 'LA') or image.size != (new_width, new_height):
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            image.save(output_path, 'PNG', quality=100)
        return True
    except Exception as e:
        logger.error(f"转换错误: {input_path} - {e}")
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/process_images', methods=['POST'])
def process_images():
    data = request.json
    folder_path = data.get('folder_path')
    annotation_type = data.get('annotation_type')
    selected_types = data.get('selected_types', [])
    language = data.get('language', 'en')
    
    task_id = str(uuid.uuid4())
    processing_status[task_id] = {'status': 'processing', 'progress': 0, 'messages': []}
    
    def process_task():
        try:
            source_path = Path(folder_path)
            supported_formats = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp', '*.gif', '*.tiff', '*.tif', '*.jfif', '*.heic', '*.heif', '*.avif']
            image_paths = []
            for ext in supported_formats:
                image_paths.extend(source_path.glob(ext))

            total_images = len(image_paths)
            if total_images == 0:
                processing_status[task_id] = {'status': 'error', 'message': '在指定文件夹中没有找到支持的图片文件。'}
                return

            tool = get_model_instance()
            
            for index, image_path in enumerate(image_paths, start=1):
                progress = int((index / total_images) * 100)
                processing_status[task_id]['progress'] = progress
                processing_status[task_id]['messages'].append(f"正在处理第 {index} 张图片，共 {total_images} 张：{image_path.name}")
                
                if annotation_type == 'natural':
                    # Single output only (.txt), no refine, no .md
                    annotation_text = tool.annotate_image(image_path, selected_types, language)
                    if annotation_text:
                        txt_path = source_path / f"{image_path.stem}.txt"
                        with open(txt_path, 'w', encoding='utf-8') as f:
                            f.write(annotation_text)
                            
                elif annotation_type == 'tags':
                    annotation_text = tool.annotate_image_short(image_path, language)
                    if annotation_text:
                        txt_path = source_path / f"{image_path.stem}.txt"
                        with open(txt_path, 'w', encoding='utf-8') as f:
                            f.write(annotation_text)
                            
                elif annotation_type == 'mixed':
                    annotation_text = tool.annotate_image_mixed(image_path, language)
                    if annotation_text:
                        txt_path = source_path / f"{image_path.stem}.txt"
                        with open(txt_path, 'w', encoding='utf-8') as f:
                            f.write(annotation_text)
                            
                elif annotation_type.startswith('custom_'):
                    template_id = annotation_type.replace('custom_', '')
                    annotation_text = tool.annotate_image_custom(image_path, template_id)
                    if annotation_text:
                        txt_path = source_path / f"{image_path.stem}.txt"
                        with open(txt_path, 'w', encoding='utf-8') as f:
                            f.write(annotation_text)

            # 自然模式不再生成/移动 .md 推理记录
            
            # 检查错误文件（混合模式）
            if annotation_type == 'mixed':
                error_folder = source_path / "疑似错误"
                error_folder.mkdir(exist_ok=True)
                common_errors = ["can't assist", "sorry", "please upload"]
                error_list = []
                
                for txt_file in source_path.glob("*.txt"):
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                        if any(error in content for error in common_errors):
                            error_list.append(txt_file.stem)
                
                if error_list:
                    error_report_path = error_folder / "报错清单.txt"
                    with open(error_report_path, 'w', encoding='utf-8') as f:
                        for error_name in error_list:
                            f.write(f"{error_name}\n")

            processing_status[task_id] = {'status': 'completed', 'progress': 100, 'message': '处理完成！'}
            
        except Exception as e:
            processing_status[task_id] = {'status': 'error', 'message': str(e)}
    
    thread = threading.Thread(target=process_task)
    thread.start()
    
    return jsonify({'task_id': task_id})

@app.route('/api/process_status/<task_id>')
def get_process_status(task_id):
    return jsonify(processing_status.get(task_id, {'status': 'not_found'}))

@app.route('/api/convert_images', methods=['POST'])
def convert_images():
    data = request.json
    folder_path = data.get('folder_path')
    max_size = data.get('max_size', 768)
    rename_files = data.get('rename_files', False)
    output_format = data.get('output_format', 'PNG')
    
    task_id = str(uuid.uuid4())
    processing_status[task_id] = {'status': 'processing', 'progress': 0, 'messages': []}
    
    def convert_task():
        try:
            source_path = Path(folder_path)
            output_folder = source_path / 'resized'
            output_folder.mkdir(exist_ok=True)
            
            image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff', '.webp')
            image_files = [f for f in source_path.iterdir() if f.suffix.lower() in image_extensions]
            
            total_images = len(image_files)
            if total_images == 0:
                processing_status[task_id] = {'status': 'error', 'message': '在指定文件夹中没有找到支持的图片文件。'}
                return
            
            for index, image_file in enumerate(image_files, start=1):
                progress = int((index / total_images) * 100)
                processing_status[task_id]['progress'] = progress
                processing_status[task_id]['messages'].append(f"正在转换第 {index} 张图片，共 {total_images} 张。")
                
                if rename_files:
                    new_filename = str(uuid.uuid4()) + f'.{output_format.lower()}'
                else:
                    new_filename = f"{image_file.stem}.{output_format.lower()}"
                
                output_file = output_folder / new_filename
                
                try:
                    with Image.open(image_file) as image:
                        width, height = image.size
                        
                        if width > height:
                            new_width = max_size
                            new_height = int((height / width) * max_size)
                        else:
                            new_height = max_size
                            new_width = int((width / height) * max_size)
                        
                        if image.mode in ('RGBA', 'LA') or image.size != (new_width, new_height):
                            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                        
                        image.save(output_file, output_format, quality=100)
                        
                except Exception as e:
                    processing_status[task_id]['messages'].append(f"转换错误: {image_file.name} - {e}")
            
            processing_status[task_id] = {'status': 'completed', 'progress': 100, 'message': '图片转换完成！'}
            
        except Exception as e:
            processing_status[task_id] = {'status': 'error', 'message': str(e)}
    
    thread = threading.Thread(target=convert_task)
    thread.start()
    
    return jsonify({'task_id': task_id})

@app.route('/api/manage_tags', methods=['POST'])
def manage_tags():
    data = request.json
    action = data.get('action')
    folder_path = data.get('folder_path')
    
    if action == 'list_files':
        try:
            source_path = Path(folder_path)
            file_list = []
            for txt_file in source_path.glob("*.txt"):
                try:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    file_list.append({'filename': txt_file.name, 'content': content[:100] + '...' if len(content) > 100 else content})
                except Exception as e:
                    file_list.append({'filename': txt_file.name, 'content': f'读取错误: {str(e)}'})
            return jsonify({'files': file_list})
        except Exception as e:
            return jsonify({'error': str(e)})
    
    elif action == 'process_files':
        search_text = data.get('search_text', '')
        replace_text = data.get('replace_text', '')
        add_trigger = data.get('add_trigger', False)
        trigger_word = data.get('trigger_word', '')
        
        try:
            source_path = Path(folder_path)
            changed_files = []
            
            for txt_file in source_path.glob("*.txt"):
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                if search_text:
                    if replace_text:
                        content = content.replace(search_text, replace_text)
                    else:
                        content = content.replace(search_text, "")
                
                if add_trigger and trigger_word:
                    content = f"{trigger_word}, {content}"
                
                if content != original_content:
                    with open(txt_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    changed_files.append(txt_file.name)
            
            return jsonify({'message': f'处理完成。共修改了 {len(changed_files)} 个文件。', 'changed_files': changed_files})
            
        except Exception as e:
            return jsonify({'error': str(e)})
    
    elif action == 'convert_encoding':
        try:
            source_path = Path(folder_path)
            converted_files = []
            
            for txt_file in source_path.glob("*.txt"):
                try:
                    # 尝试读取文件并重新以UTF-8编码保存
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 额外功能：移除所有换行符（\r\n、\n、\r）
                    content_no_newlines = (
                        content.replace('\r\n', '').replace('\n', '').replace('\r', '')
                    )
                    
                    with open(txt_file, 'w', encoding='utf-8') as f:
                        f.write(content_no_newlines)
                    
                    converted_files.append(txt_file.name)
                except Exception as e:
                    logger.error(f"转换编码错误: {txt_file.name} - {e}")
            
            return jsonify({'message': f'编码转换完成。共处理了 {len(converted_files)} 个文件。', 'converted_files': converted_files})
            
        except Exception as e:
            return jsonify({'error': str(e)})

@app.route('/api/get_analysis_types', methods=['GET'])
def get_analysis_types():
    # Analysis types feature has been removed - return empty list
    return jsonify({'analysis_types': []})

@app.route('/api/process_single_image', methods=['POST'])
def process_single_image():
    try:
        # 获取上传的文件
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': '没有上传图片文件'})
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': '没有选择文件'})
        
        # 获取其他参数
        annotation_type = request.form.get('annotation_type', 'natural')
        selected_types = json.loads(request.form.get('selected_types', '[]'))
        language = request.form.get('language', 'en')
        
        # 保存临时文件
        temp_dir = Path('temp')
        temp_dir.mkdir(exist_ok=True)
        
        # 生成唯一文件名
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.tiff', '.tif']:
            return jsonify({'success': False, 'error': '不支持的图片格式'})
        
        temp_filename = f"{uuid.uuid4()}{file_extension}"
        temp_path = temp_dir / temp_filename
        
        # 保存文件
        file.save(temp_path)
        
        try:
            # 初始化工具（本地视觉模型）
            tool = get_model_instance()
            
            # 根据标注类型处理
            if annotation_type == 'natural':
                # Single paragraph output only
                annotation_text = tool.annotate_image(temp_path, selected_types, language)
                result_text = annotation_text if annotation_text else "标注失败，请重试"
                    
            elif annotation_type == 'tags':
                annotation_text = tool.annotate_image_short(temp_path, language)
                result_text = annotation_text if annotation_text else "标注失败，请重试"
                
            elif annotation_type == 'mixed':
                annotation_text = tool.annotate_image_mixed(temp_path, language)
                result_text = annotation_text if annotation_text else "标注失败，请重试"
                
            elif annotation_type.startswith('custom_'):
                template_id = annotation_type.replace('custom_', '')
                annotation_text = tool.annotate_image_custom(temp_path, template_id)
                result_text = annotation_text if annotation_text else "标注失败，请重试"
            
            
            else:
                result_text = "不支持的标注类型"
            
            # 清理临时文件
            if temp_path.exists():
                temp_path.unlink()
            
            if result_text and "标注失败" not in result_text:
                return jsonify({'success': True, 'annotation': result_text})
            else:
                return jsonify({'success': False, 'error': '图片标注失败，请检查API密钥或网络连接'})
                
        except Exception as e:
            # 清理临时文件
            if temp_path.exists():
                temp_path.unlink()
            logger.error(f"单张图片处理错误: {str(e)}")
            return jsonify({'success': False, 'error': f'处理错误: {str(e)}'})
            
    except Exception as e:
        logger.error(f"单张图片接口错误: {str(e)}")
        return jsonify({'success': False, 'error': f'接口错误: {str(e)}'})

# Custom template management endpoints
@app.route('/api/custom_templates', methods=['GET'])
def get_custom_templates():
    try:
        tool = get_model_instance()
        return jsonify({'templates': tool.user_prompts.get('custom_templates', [])})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/custom_templates', methods=['POST'])
def create_custom_template():
    try:
        data = request.json
        name = data.get('name', '').strip()
        description = data.get('description', '').strip()
        prompt = data.get('prompt', '').strip()
        
        if not name or not prompt:
            return jsonify({'error': '模板名称和提示词不能为空'})
        
        tool = get_model_instance()
        
        # Generate unique ID
        template_id = str(uuid.uuid4())
        
        # Create new template
        new_template = {
            'id': template_id,
            'name': name,
            'description': description,
            'prompt': prompt,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        
        tool.user_prompts['custom_templates'].append(new_template)
        tool.save_user_prompts()
        
        return jsonify({'success': True, 'template': new_template})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/custom_templates/<template_id>', methods=['PUT'])
def update_custom_template(template_id):
    try:
        data = request.json
        name = data.get('name', '').strip()
        description = data.get('description', '').strip()
        prompt = data.get('prompt', '').strip()
        
        if not name or not prompt:
            return jsonify({'error': '模板名称和提示词不能为空'})
        
        tool = get_model_instance()
        
        # Find and update template
        template_found = False
        for template in tool.user_prompts['custom_templates']:
            if template['id'] == template_id:
                template['name'] = name
                template['description'] = description
                template['prompt'] = prompt
                template['updated_at'] = datetime.now().isoformat()
                template_found = True
                break
        
        if not template_found:
            return jsonify({'error': '模板未找到'})
        
        tool.save_user_prompts()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/custom_templates/<template_id>', methods=['DELETE'])
def delete_custom_template(template_id):
    try:
        tool = get_model_instance()
        
        # Find and remove template
        original_count = len(tool.user_prompts['custom_templates'])
        tool.user_prompts['custom_templates'] = [
            t for t in tool.user_prompts['custom_templates'] 
            if t['id'] != template_id
        ]
        
        if len(tool.user_prompts['custom_templates']) == original_count:
            return jsonify({'error': '模板未找到'})
        
        tool.save_user_prompts()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/process_video', methods=['POST'])
def process_video():
    try:
        # 检查是否有上传的视频文件
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': '未找到视频文件'})
        
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'success': False, 'error': '未选择视频文件'})
        
        # 获取参数（简化为仅语言选择）
        language = request.form.get('language', 'zh')
        
        # 保存上传的视频文件
        temp_dir = Path('temp')
        temp_dir.mkdir(exist_ok=True)
        
        # 生成唯一文件名
        file_extension = Path(video_file.filename).suffix.lower()
        if file_extension not in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            return jsonify({'success': False, 'error': f'不支持的视频格式: {file_extension}'})
        
        temp_filename = f"{uuid.uuid4()}{file_extension}"
        temp_video_path = temp_dir / temp_filename
        
        # 保存文件
        video_file.save(temp_video_path)
        logger.info(f"视频文件已保存到: {temp_video_path}")
        
        try:
            # 获取模型实例并处理视频
            tool = get_model_instance()
            
            # 确保模型已加载
            if not tool.is_loaded():
                tool.load_model()
            
            # 处理视频（自动检测参数）
            annotation, processing_time = tool.process_video(
                str(temp_video_path), 
                language=language
            )
            
            return jsonify({
                'success': True,
                'annotation': annotation,
                'processing_time': f"{processing_time:.2f}秒"
            })
            
        finally:
            # 清理临时文件
            try:
                if temp_video_path.exists():
                    temp_video_path.unlink()
                    logger.info(f"临时视频文件已删除: {temp_video_path}")
            except Exception as cleanup_error:
                logger.warning(f"清理临时文件失败: {cleanup_error}")
                
    except Exception as e:
        logger.error(f"视频处理API错误: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/process_videos', methods=['POST'])
def process_videos():
    try:
        data = request.get_json()
        folder_path = data.get('folder_path')
        language = data.get('language', 'zh')
        
        if not folder_path or not os.path.exists(folder_path):
            return jsonify({'error': '文件夹路径无效'})
        
        # 生成任务ID
        task_id = str(uuid.uuid4())
        processing_status[task_id] = {
            'status': 'starting',
            'progress': 0,
            'message': '正在扫描视频文件...',
            'messages': []
        }
        
        # 在后台线程中处理视频
        thread = threading.Thread(target=batch_process_videos, args=(task_id, folder_path, language))
        thread.daemon = True
        thread.start()
        
        return jsonify({'task_id': task_id})
    except Exception as e:
        logger.error(f"批量视频处理启动失败: {e}")
        return jsonify({'error': str(e)})

def batch_process_videos(task_id, folder_path, language):
    try:
        # 获取模型实例
        tool = get_model_instance()
        if not tool.is_loaded():
            tool.load_model()
        
        # 扫描视频文件
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv']
        video_files = []
        
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in video_extensions):
                    video_files.append(os.path.join(root, file))
        
        if not video_files:
            processing_status[task_id] = {
                'status': 'error',
                'message': '未找到支持的视频文件'
            }
            return
        
        processing_status[task_id]['message'] = f'找到 {len(video_files)} 个视频文件，开始处理...'
        processing_status[task_id]['messages'].append(f'找到 {len(video_files)} 个视频文件')
        
        # 处理每个视频文件
        for i, video_file in enumerate(video_files):
            try:
                processing_status[task_id]['progress'] = int((i / len(video_files)) * 100)
                processing_status[task_id]['message'] = f'正在处理: {os.path.basename(video_file)}'
                processing_status[task_id]['messages'].append(f'处理视频: {os.path.basename(video_file)}')
                
                # 处理视频
                annotation, processing_time = tool.process_video(video_file, language=language)
                
                # 保存结果到视频文件同目录
                video_dir = os.path.dirname(video_file)
                video_name = os.path.splitext(os.path.basename(video_file))[0]
                output_file = os.path.join(video_dir, f'{video_name}_annotation.txt')
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(annotation)
                
                processing_status[task_id]['messages'].append(f'完成: {os.path.basename(video_file)} ({processing_time:.1f}s)')
                
            except Exception as e:
                error_msg = f'处理 {os.path.basename(video_file)} 失败: {str(e)}'
                processing_status[task_id]['messages'].append(error_msg)
                logger.error(error_msg)
        
        # 完成处理
        processing_status[task_id] = {
            'status': 'completed',
            'progress': 100,
            'message': f'批量视频标注完成！处理了 {len(video_files)} 个视频文件，结果保存在各视频文件旁边'
        }
        
    except Exception as e:
        logger.error(f"批量视频处理失败: {e}")
        processing_status[task_id] = {
            'status': 'error',
            'message': f'批量处理失败: {str(e)}'
        }

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5005)
