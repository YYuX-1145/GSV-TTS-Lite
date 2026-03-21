# GSV-TTS 异步功能使用说明

## 📊 1. 测试同步 vs 异步性能

运行测试脚本对比同步和异步的速度：

```bash
cd examples
python test_async_performance.py
```

## 🔧 2. 你现在有的接口

### 接口一：WebUI（原来的 Gradio 界面）
- **位置**：`WebUI/web.py`
- **启动方式**：
  ```bash
  cd WebUI
  python web.py
  ```
- **用途**：图形化界面，适合本地调试和体验
- **特点**：同步处理，适合单用户使用

### 接口二：TTS 类的 Python API
- **位置**：`gsv_tts/TTS.py`
- **新增的异步方法**：
  1. `infer_async()` - 单个异步请求（线程安全）
  2. `infer_batched_async()` - 批量异步请求（线程安全）

### 接口三：FastAPI 服务端 API
- **位置**：`fastapi_server_example.py`（项目根目录）
- **启动方式**：
  ```bash
  python fastapi_server_example.py
  ```
- **用途**：RESTful API 服务，适合服务端部署
- **特点**：异步处理，支持并发请求

## 💡 3. 怎么使用异步功能

### 方式一：直接使用 TTS 类

```python
import asyncio
from gsv_tts import TTS

tts = TTS(models_dir="WebUI/models")

async def main():
    # 单个异步请求
    audio = await tts.infer_async(
        spk_audio_path="examples/laffey.mp3",
        prompt_audio_path="examples/AnAn.ogg",
        prompt_audio_text="ちが……ちがう。レイア、貴様は間違っている。",
        text="你好，这是测试。"
    )
    audio.save("output.wav")
    
    # 批量异步请求（推荐，GPU 利用率更高）
    texts = ["第一个请求", "第二个请求", "第三个请求"]
    audios = await tts.infer_batched_async(
        spk_audio_paths="examples/laffey.mp3",
        prompt_audio_paths="examples/AnAn.ogg",
        prompt_audio_texts="ちが……ちがう。レイア、貴様は間違っている。",
        texts=texts
    )
    for i, audio in enumerate(audios):
        audio.save(f"output_{i}.wav")

asyncio.run(main())
```

### 方式二：FastAPI 服务端（推荐部署）

#### 1. 安装依赖
```bash
pip install fastapi uvicorn requests
```

#### 2. 启动服务
```bash
python fastapi_server_example.py
```

启动成功后会显示：
```
🚀 正在加载 TTS 模型...
✅ TTS 模型加载完成！
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

#### 3. 访问 API 文档
浏览器打开：`http://localhost:8000/docs`

#### 4. API 接口说明

**单个 TTS 请求**：
- **接口**：`POST /tts/single`
- **请求参数**：
  ```json
  {
    "text": "你好，这是测试。",
    "speaker_audio": "examples/laffey.mp3",
    "prompt_audio": "examples/AnAn.ogg",
    "prompt_text": "ちが……ちがう。レイア、貴様は間違っている。",
    "top_k": 5,
    "top_p": 0.9,
    "temperature": 1.0,
    "repetition_penalty": 1.35,
    "noise_scale": 0.5,
    "speed": 1.0
  }
  ```
- **响应示例**：
  ```json
  {
    "success": true,
    "audio_len": 1.72,
    "filename": "tts_06a1a5fc.wav"
  }
  ```

**批量 TTS 请求**：
- **接口**：`POST /tts/batch`
- **请求参数**：
  ```json
  {
    "texts": ["第一个请求", "第二个请求"],
    "speaker_audio": "examples/laffey.mp3",
    "prompt_audio": "examples/AnAn.ogg",
    "prompt_text": "ちが……ちがう。レイア、貴様は間違っている。"
  }
  ```
- **响应示例**：
  ```json
  {
    "success": true,
    "count": 2,
    "filenames": ["tts_ca15dce7.wav", "tts_08c8418f.wav"]
  }
  ```

**下载音频文件**：
- **接口**：`GET /audio/{filename}`
- **示例**：`http://localhost:8000/audio/tts_06a1a5fc.wav`

#### 5. 使用 Python 测试 API

```python
import requests
from pathlib import Path

# 获取项目根目录（假设脚本在项目根目录运行）
project_root = Path.cwd()

# 单个请求
response = requests.post(
    "http://localhost:8000/tts/single",
    json={
        "text": "你好，这是测试。",
        "speaker_audio": str(project_root / "examples" / "laffey.mp3"),
        "prompt_audio": str(project_root / "examples" / "AnAn.ogg"),
        "prompt_text": "ちが……ちがう。レイア、貴様は間違っている。"
    }
)
print(response.json())

# 批量请求
response = requests.post(
    "http://localhost:8000/tts/batch",
    json={
        "texts": ["第一个请求", "第二个请求"],
        "speaker_audio": str(project_root / "examples" / "laffey.mp3"),
        "prompt_audio": str(project_root / "examples" / "AnAn.ogg"),
        "prompt_text": "ちが……ちがう。レイア、貴様は間違っている。"
    }
)
print(response.json())
```

#### 6. 测试脚本

运行测试脚本：
```bash
cd examples
python test_fastapi_client.py
```

## 📝 4. 技术细节

### 线程安全
- ✅ 已添加 `threading.Lock` 确保线程安全
- ✅ 支持多线程并发请求
- ✅ 避免了 `LangSegment` 静态变量的竞态条件

### 性能对比
- **同步逐个处理**：串行执行，适合少量请求
- **异步批处理**：GPU 并行处理，适合批量请求，效率更高

### 推荐使用场景
| 场景 | 推荐方法 | 特点 |
|------|----------|------|
| 批量已知请求 | `infer_batched_async()` | ✅ 真正的 GPU 批处理，效率最高 |
| 随机单个请求 | `infer_async()` | ✅ 线程安全，适合服务端随机请求 |
| 本地调试 | WebUI | ✅ 图形化界面，易用 |
| 脚本处理 | 同步方法 | ✅ 简单直接 |
| 服务端部署 | FastAPI | ✅ RESTful API，支持并发 |

## 🔍 5. 常见问题

### Q: 为什么 `infer_async` 不是真正的并行？
A: 因为 `LangSegment` 类使用了静态变量，需要加锁保证线程安全，所以实际是串行执行的。

### Q: 什么时候用 `infer_batched_async`？
A: 当你有多个请求要处理时，用 `infer_batched_async` 可以真正利用 GPU 并行能力，速度更快。

### Q: 服务端部署推荐哪种方式？
A: 推荐使用 FastAPI 示例，它提供了完整的 RESTful API 接口。

### Q: FastAPI 服务启动失败怎么办？
A: 检查端口是否被占用，如果 8000 端口被占用，可以修改 `fastapi_server_example.py` 中的端口号。

### Q: 如何修改 FastAPI 端口？
A: 在 `fastapi_server_example.py` 中修改：
```python
uvicorn.run(app, host="0.0.0.0", port=8001)  # 改为其他端口
```

## 🎯 6. 最佳实践

1. **服务端部署**：使用 FastAPI 示例，提供 RESTful API
2. **批量处理**：尽量使用 `infer_batched_async` 提高 GPU 利用率
3. **预热模型**：在服务启动时预热模型，避免首次请求延迟
4. **错误处理**：在生产环境中添加适当的错误处理
5. **端口管理**：确保端口不被占用，必要时修改端口号

## 🚀 7. 快速开始

### 最简单的使用方式（FastAPI）

```bash
# 1. 安装依赖
pip install fastapi uvicorn requests

# 2. 启动服务
python fastapi_server_example.py

# 3. 打开浏览器访问 API 文档
# http://localhost:8000/docs

# 4. 测试 API
python examples/test_fastapi_client.py
```

**推荐**：部署到服务端时使用 FastAPI 示例或 TTS 类的 `infer_async()` 方法！