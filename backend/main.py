from __future__ import annotations

import os
import tempfile

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .valve_anomaly_pipeline import (
    get_device,
    MachineTypePipeline,
    ValveAnomalyPipeline,
)
from .audio_recorder import (
    list_audio_input_devices,
    record_audio,
    save_wav_file,
)


class MachineResult(BaseModel):
    """设备类型分类结果。"""

    machine_index: int
    machine_name: str
    probs: list[float]


class AnomalyResult(BaseModel):
    """某设备下的 section + 正常/异常 结果。"""

    section: int
    anomaly_score: float
    threshold: float
    is_normal: bool


class FeatureGroup(BaseModel):
    """特征类型级别的重要性（贡献度）"""

    name: str
    importance: float
    z_score: float | None = None
    is_abnormal: bool


class InferResponse(BaseModel):
    """综合推理结果：设备类型 + 异常检测。"""

    machine: MachineResult
    anomaly: AnomalyResult


class AnomalyExplainResult(AnomalyResult):
    """带特征类型解释信息的异常检测结果。"""

    feature_groups: list[FeatureGroup]


class InferExplainResponse(BaseModel):
    """综合推理 + 异常解释结果。"""

    machine: MachineResult
    anomaly: AnomalyExplainResult


class ModelAccuracyItem(BaseModel):
    """单个设备类型的演示准确率（大约 70% 左右，用于展示预测准确度）。"""

    machine_name: str
    accuracy: float


class TrainAccuracyItem(BaseModel):
    """在离线评估/验证集上，每个设备类型的真实分类准确率。"""

    machine_name: str
    accuracy: float
    correct: int
    total: int


class ModelAccuracyResponse(BaseModel):
    """
    模型整体功能介绍 + 准确率信息。

    - description: 文本介绍模型功能
    - overall_accuracy: 演示用的整体预测准确率（约 70%）
    - per_machine: 每个设备的演示预测准确率（约 70%）
    - train_accuracy: 离线评估时，每个设备的真实分类准确率（如终端截图所示）
    """

    description: str
    overall_accuracy: float
    per_machine: list[ModelAccuracyItem]
    train_accuracy: list[TrainAccuracyItem]


class AudioDevice(BaseModel):
    """音频输入设备信息。"""

    index: int
    name: str
    channels: int
    sample_rate: int


class RecordAudioRequest(BaseModel):
    """录制音频请求参数。"""

    device_index: int
    duration: float
    sample_rate: int = 44100
    channels: int = 1
    save_path: str | None = None


class RecordAudioResponse(BaseModel):
    """录制音频响应结果。"""

    filepath: str
    filename: str
    file_size_kb: float
    duration: float
    sample_rate: int
    channels: int


def create_app() -> FastAPI:
    """
    创建 FastAPI 应用：
    - 提供 /infer 接口：上传 .wav，返回设备类型 + section + 正常/异常
    - 复用现有的 MachineTypePipeline / ValveAnomalyPipeline
    """
    app = FastAPI(
        title="Industrial Audio Anomaly Detection Backend",
        description=(
            "后端服务：输入一段音频，先判断属于哪个设备（data 目录），"
            "再在对应设备下判断 section 和是否正常。"
        ),
        version="1.0.0",
    )

    # 允许前端跨域访问（按需修改 origins）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 设备与模型路径可通过环境变量覆盖，便于部署配置
    device_name = os.getenv("AUDIO_DEVICE", "cpu")  # "cuda" 或 "cpu"
    machine_model_path = os.getenv(
        "MACHINE_TYPE_MODEL_PATH",
        "models/machine_type_classifier.pth",
    )
    data_root_base = os.getenv("AUDIO_DATA_ROOT_BASE", "data")

    device = get_device(device_name)

    # 全局设备类型分类模型
    mt_pipeline = MachineTypePipeline(
        model_path=machine_model_path,
        device=device,
    )

    # 各设备的异常检测 pipeline 缓存： { "valve": ValveAnomalyPipeline(...), ... }
    pipeline_cache: dict[str, ValveAnomalyPipeline] = {}

    def get_anomaly_pipeline(machine_name: str) -> ValveAnomalyPipeline:
        """
        根据设备名返回对应的异常检测 pipeline，首次调用时创建并缓存。
        """
        if machine_name not in pipeline_cache:
            data_root = os.path.join(data_root_base, machine_name)
            pipeline_cache[machine_name] = ValveAnomalyPipeline(
                data_root=data_root,
                device=device,
            )
        return pipeline_cache[machine_name]

    @app.get("/health", summary="健康检查")
    async def health_check() -> dict[str, str]:
        return {"status": "ok"}

    @app.post(
        "/infer",
        summary="综合推理：设备类型 + section + 正常/异常",
        description=(
            "上传一段 .wav 音频：\n"
            "- 先用 MachineTypePipeline 预测属于哪个设备（data 目录，例如 valve/pump/...）\n"
            "- 再在对应设备下用 ValveAnomalyPipeline 预测 section + 正常/异常\n"
        ),
        response_model=InferResponse,
    )
    async def infer(file: UploadFile = File(...)) -> InferResponse:
        """
        输入：
        - multipart/form-data，字段名为 'file'，内容为 .wav 音频文件

        输出：
        - machine: 设备类型分类结果（machine_name, machine_index, probs）
        - anomaly: 对应设备下的 section + 正常/异常 结果
        """
        # 1) 将上传文件保存为临时 .wav
        suffix = os.path.splitext(file.filename or "")[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            # 2) 先判定属于哪个设备（data 目录）
            mt_result = mt_pipeline.predict_machine(tmp_path)
            machine_name = mt_result["machine_name"]

            # 3) 在对应设备下做 section + 正常/异常 检测
            anomaly_pipeline = get_anomaly_pipeline(machine_name)
            anomaly_result = anomaly_pipeline.predict(tmp_path)

            # 4) 返回统一结构（由 pydantic 模型自动校验/转换）
            return InferResponse(
                machine=MachineResult(**mt_result),
                anomaly=AnomalyResult(**anomaly_result),
            )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @app.post(
        "/infer/explain",
        summary="综合推理 + 特征类型重要性解释",
        description=(
            "上传一段 .wav 音频：\n"
            "- 先用 MachineTypePipeline 预测属于哪个设备（data 目录，例如 valve/pump/...）\n"
            "- 再在对应设备下用 ValveAnomalyPipeline 预测 section + 正常/异常\n"
            "- 同时给出 mel/MFCC/频谱/时域/工业特征这几类特征块对异常得分的相对贡献度"
        ),
        response_model=InferExplainResponse,
    )
    async def infer_with_explanation(
        file: UploadFile = File(...),
        use_true_section: bool = False,
        section_id: int | None = None,
    ) -> InferExplainResponse:
        """
        参数：
        - file: .wav 音频
        - use_true_section: 若为 True 且提供 section_id，则不再预测 section，直接用该 section 做异常检测
        - section_id: 已知的真实 section 编号（可选）
        """
        suffix = os.path.splitext(file.filename or "")[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            mt_result = mt_pipeline.predict_machine(tmp_path)
            machine_name = mt_result["machine_name"]

            anomaly_pipeline = get_anomaly_pipeline(machine_name)
            explain_result = anomaly_pipeline.explain_anomaly(
                tmp_path,
                section_id=section_id,
                use_true_section=use_true_section,
            )

            return InferExplainResponse(
                machine=MachineResult(**mt_result),
                anomaly=AnomalyExplainResult(**explain_result),
            )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @app.post(
        "/infer/machine",
        summary="仅预测设备类型（属于哪个 data 目录）",
        description=(
            "上传一段 .wav 音频，只使用 MachineTypePipeline，"
            "输出设备名称及概率分布，不做异常检测。"
        ),
        response_model=MachineResult,
    )
    async def infer_machine(file: UploadFile = File(...)) -> MachineResult:
        suffix = os.path.splitext(file.filename or "")[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            mt_result = mt_pipeline.predict_machine(tmp_path)
            return MachineResult(**mt_result)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @app.post(
        "/infer/anomaly",
        summary="仅在指定设备下做 section + 正常/异常 检测",
        description=(
            "上传一段 .wav 音频，并指定设备名 machine_name：\n"
            "- 使用对应 data/<machine_name> 下的 ValveAnomalyPipeline\n"
            "- 默认自动预测 section，再给出正常/异常结果\n"
            "- 如果已知真实 section_id，可设置 use_true_section=true 以避免 section 误判影响"
        ),
        response_model=AnomalyResult,
    )
    async def infer_anomaly(
        machine_name: str,
        file: UploadFile = File(...),
        section_id: int | None = None,
        use_true_section: bool = False,
    ) -> AnomalyResult:
        """
        参数：
        - machine_name: 设备名称，例如 'valve' / 'pump' / 'fan'
        - file: .wav 音频
        - section_id: （可选）已知的真实 section 编号
        - use_true_section: 若为 True 且提供了 section_id，则使用 predict_with_true_section
        """
        suffix = os.path.splitext(file.filename or "")[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            anomaly_pipeline = get_anomaly_pipeline(machine_name)
            if use_true_section and section_id is not None:
                result = anomaly_pipeline.predict_with_true_section(
                    tmp_path, section_id=section_id
                )
            else:
                result = anomaly_pipeline.predict(tmp_path)
            return AnomalyResult(**result)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @app.get(
        "/machines",
        summary="列出支持的设备类型（data 子目录）",
        description="返回训练好的设备类型分类模型中包含的 machine_names 列表。",
        response_model=list[str],
    )
    async def list_machines() -> list[str]:
        # MachineTypePipeline 内部持有 machine_names
        names = getattr(mt_pipeline, "machine_names", None)
        if names is None:
            return []
        return list(names)

    @app.get(
        "/model/accuracy",
        summary="获取模型功能说明和分类准确率（演示 + 真实评估数据）",
        description=(
                "上传一段 .wav 音频：\n"
                "- 先用 MachineTypePipeline 预测属于哪个设备（data 目录，例如 valve/pump/...）\n"
                "- 再在对应设备下用 ValveAnomalyPipeline 预测 section + 正常/异常\n"
                "- 同时给出 mel/MFCC/频谱/时域/工业特征这几类特征块对异常得分的相对贡献度"
        ),
        response_model=ModelAccuracyResponse,
    )
    async def get_model_accuracy() -> ModelAccuracyResponse:
        # 1) 前端用于展示“预测准确率”的演示数据（约 70% 左右）
        demo_per_machine = [
            ModelAccuracyItem(machine_name="ToyCar", accuracy=0.72),
            ModelAccuracyItem(machine_name="ToyTrain", accuracy=0.75),
            ModelAccuracyItem(machine_name="fan", accuracy=0.69),
            ModelAccuracyItem(machine_name="gearbox", accuracy=0.71),
            ModelAccuracyItem(machine_name="pump", accuracy=0.73),
            ModelAccuracyItem(machine_name="slider", accuracy=0.68),
            ModelAccuracyItem(machine_name="valve", accuracy=0.74),
        ]
        overall_accuracy = sum(m.accuracy for m in demo_per_machine) / len(
            demo_per_machine
        )

        # 2) 根据你训练日志中的结果（截图）填写的真实分类准确率
        train_accuracy = [
            TrainAccuracyItem(
                machine_name="ToyCar",
                accuracy=0.9689,
                correct=4078,
                total=4209,
            ),
            TrainAccuracyItem(
                machine_name="ToyTrain",
                accuracy=0.9990,
                correct=4205,
                total=4209,
            ),
            TrainAccuracyItem(
                machine_name="fan",
                accuracy=0.7021,
                correct=2955,
                total=4209,
            ),
            TrainAccuracyItem(
                machine_name="gearbox",
                accuracy=0.6661,
                correct=2953,
                total=4433,
            ),
            TrainAccuracyItem(
                machine_name="pump",
                accuracy=0.9613,
                correct=4046,
                total=4209,
            ),
            TrainAccuracyItem(
                machine_name="slider",
                accuracy=0.7807,
                correct=3297,
                total=4223,
            ),
            TrainAccuracyItem(
                machine_name="valve",
                accuracy=0.9815,
                correct=4131,
                total=4209,
            ),
        ]

        return ModelAccuracyResponse(
            description=(
                "上传一段 .wav 音频：\n"
                "- 先用 MachineTypePipeline 预测属于哪个设备（data 目录，例如 valve/pump/...）\n"
                "- 再在对应设备下用 ValveAnomalyPipeline 预测 section + 正常/异常\n"
                "- 同时给出 mel/MFCC/频谱/时域/工业特征这几类特征块对异常得分的相对贡献度"
            ),
            overall_accuracy=overall_accuracy,
            per_machine=demo_per_machine,
            train_accuracy=train_accuracy,
        )

    @app.get(
        "/audio/devices",
        summary="列出所有可用的音频输入设备",
        description="返回系统中所有可用的音频输入设备列表，包括设备索引、名称、通道数和采样率。",
        response_model=list[AudioDevice],
    )
    async def get_audio_devices() -> list[AudioDevice]:
        """获取所有可用的音频输入设备。"""
        devices = list_audio_input_devices(verbose=False)
        return [
            AudioDevice(
                index=device["index"],
                name=device["name"],
                channels=device["channels"],
                sample_rate=device["sample_rate"],
            )
            for device in devices
        ]

    @app.post(
        "/audio/record",
        summary="录制音频并保存为WAV文件",
        description=(
            "根据指定的参数录制音频：\n"
            "- device_index: 音频输入设备索引（可通过 /audio/devices 获取）\n"
            "- duration: 录制时长（秒）\n"
            "- sample_rate: 采样率（默认44100 Hz）\n"
            "- channels: 声道数（默认1，单声道）\n"
            "- save_path: 保存路径（可选，默认保存到 recordings 目录）\n"
        ),
        response_model=RecordAudioResponse,
    )
    async def record_audio_endpoint(request: RecordAudioRequest) -> RecordAudioResponse:
        """
        录制音频并保存为WAV文件。

        参数：
        - device_index: 音频输入设备索引
        - duration: 录制时长（秒）
        - sample_rate: 采样率（默认44100 Hz）
        - channels: 声道数（默认1）
        - save_path: 保存路径（可选）
        """
        # 确定保存路径
        if request.save_path:
            save_path = request.save_path
        else:
            save_path = os.path.join(os.getcwd(), "recordings")
        
        # 确保目录存在
        os.makedirs(save_path, exist_ok=True)
        
        # 录制音频
        frames = record_audio(
            device_index=request.device_index,
            duration=request.duration,
            sample_rate=request.sample_rate,
            channels=request.channels,
            verbose=False,
        )
        
        if not frames:
            raise ValueError("录制失败：未录制到音频数据")
        
        # 保存WAV文件
        filepath = save_wav_file(
            frames=frames,
            save_path=save_path,
            sample_rate=request.sample_rate,
            channels=request.channels,
            verbose=False,
        )
        
        # 获取文件信息
        filename = os.path.basename(filepath)
        file_size_kb = os.path.getsize(filepath) / 1024
        
        return RecordAudioResponse(
            filepath=filepath,
            filename=filename,
            file_size_kb=file_size_kb,
            duration=request.duration,
            sample_rate=request.sample_rate,
            channels=request.channels,
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)


