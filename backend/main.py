from __future__ import annotations

import os
import tempfile

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from valve_anomaly_pipeline import (
    get_device,
    MachineTypePipeline,
    ValveAnomalyPipeline,
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

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)


