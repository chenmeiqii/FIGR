import logging
import os
import threading
from contextlib import ExitStack
from enum import Enum
from typing import Any, Callable, Optional, TypeVar
from uuid import uuid4
import re
import ray
import base64
from verl.tools.base_tool import BaseTool
from verl.utils.reward_score.sandbox_fusion.utils import _process_single_case
from verl.utils.rollout_trace import rollout_trace_op

from .schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.dataset.vision_utils import process_image

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

T = TypeVar("T")


class PoolMode(Enum):
    ThreadMode = 1
    ProcessMode = 2


@ray.remote(concurrency_groups={"acquire": 1, "release": 10})
class TokenBucketWorker:
    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        # this only used for observalability
        self.current_count = 0
        self._semaphore = threading.Semaphore(rate_limit)

    @ray.method(concurrency_group="acquire")
    def acquire(self):
        self._semaphore.acquire()
        self.current_count += 1

    @ray.method(concurrency_group="release")
    def release(self):
        self._semaphore.release()
        self.current_count -= 1

    def get_current_count(self):
        return self.current_count


class ExecutionWorker:
    def __init__(self, enable_global_rate_limit=True, rate_limit=10):
        self.rate_limit_worker = self._init_rate_limit(rate_limit) if enable_global_rate_limit else None

    def _init_rate_limit(self, rate_limit):
        # TODO validation for rate_limit
        # A Singleton Rate Limitor
        return TokenBucketWorker.options(name="rate-limiter", get_if_exists=True).remote(rate_limit)

    def ping(self):
        return True

    def execute(self, fn: Callable[..., T], *fn_args, **fn_kwargs) -> T:
        with ExitStack() as stack:
            stack.callback(self.rate_limit_worker.release.remote)
            ray.get(self.rate_limit_worker.acquire.remote())
            try:
                return fn(*fn_args, **fn_kwargs)
            except Exception as e:
                # TODO we should make this available to the tool caller
                logger.warning(f"Error when executing code: {e}")


def init_execution_pool(
    num_workers: int, enable_global_rate_limit=True, rate_limit=10, mode: PoolMode = PoolMode.ThreadMode
):
    if mode == PoolMode.ThreadMode:
        return (
            ray.remote(ExecutionWorker)
            .options(max_concurrency=num_workers)
            .remote(enable_global_rate_limit=enable_global_rate_limit, rate_limit=rate_limit)
        )
    else:
        raise NotImplementedError("Process mode is not implemented yet")
        # return ray.util.multiprocessing.Pool(processes=num_workers)

def wrap_code_with_capture(code: str) -> str:
    HEADERS = """
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
_captured_figures = []
def _capturing_show(*args, **kwargs):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    _captured_image = base64.b64encode(buf.read()).decode('utf-8')
    _captured_figures.append(_captured_image)
    plt.close()
plt.show = _capturing_show
"""

    TAILS = """
for image_b64 in _captured_figures:
    with open(\"image.txt\", \"a\") as f:
        f.write(image_b64 + '\\n')
"""

    return HEADERS + code + TAILS


class SandboxFusionTool(BaseTool):
    """A tool for executing the code using sanbox fusion image.

    - `get_openai_tool_schema`: return the tool schema in OpenAI format.
    - `create`: create a tool instance for a trajectory.
    - `execute`: execute the tool.
    - `calc_reward`: calculate the reward respect to tool state.
    - `release`: release the tool instance.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        _tool_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "code_interpreter",
                "description": "A tool for execute code",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "code needs to be execute and grad",
                        },
                    },
                    "required": ["code"],
                },
            }
        })
        """
        super().__init__(config, tool_schema)
        self.code_pattern = re.compile(r"```python(.*?)```", re.DOTALL)
        self._instance_dict = {}
        # TODO: better documentation for the config
        self.num_workers = config.get("num_workers", 10)
        self.rate_limit = config.get("rate_limit", 10)
        self.default_timeout = config.get("default_timeout", 30)
        self.default_language = config.get("default_language", "python")
        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)
        self.execution_pool = init_execution_pool(
            num_workers=self.num_workers,
            enable_global_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
            mode=PoolMode.ThreadMode,
        )
        self.sandbox_fusion_url = config.get("sandbox_fusion_url", "")
        self.memory_limit_mb = config.get("memory_limit_mb", 1024)
        if self.sandbox_fusion_url == "":
            raise ValueError("sandbox_fusion_url is not set")
        log_msg = f"Init SandboxFusionTool with config: {config}"
        logger.info(log_msg)

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(
        self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs
    ) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": [],
        }
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[str, float, dict]:
        code = parameters.get("code", "")
        if code == "":
            self._instance_dict[instance_id]["reward"].append(-0.05)
            return ToolResponse(text="Error: code parameter is missing"), -0.05, {"success": False}
        elif not isinstance(code, str):
            code = str(code)

        code = code.replace("```py", "").replace("```python", "").replace("```", "")
        timeout = parameters.get("timeout", self.default_timeout)
        language = parameters.get("language", self.default_language)

        result = await self.execution_pool.execute.remote(self.execute_code, instance_id, code, timeout, language)
        if isinstance(result, dict):
            if result["stderr"] == "":
                if result["stdout"].strip() == "":
                    text_result = None
                else:
                    text_result = result["stdout"].strip()
            else:
                self._instance_dict[instance_id]["reward"].append(-0.05)
                return ToolResponse(text=result["stderr"]), -0.05, {"success": False}
            
        else:
            self._instance_dict[instance_id]["reward"].append(-0.05)
            return ToolResponse(text=result), -0.05, {"success": False}

        if text_result:
            if len(code) > 100:
                self._instance_dict[instance_id]["reward"].append(0.1)
                return ToolResponse(text=f"Text Result: {text_result}"), 0.1, {"success": True}
            else:
                self._instance_dict[instance_id]["reward"].append(-0.05)
                return ToolResponse(text=f"Text Result: {text_result}"), -0.05, {"success": True}
        else:
            self._instance_dict[instance_id]["reward"].append(-0.05)
            return ToolResponse(text="No stdout to show here, remember to use print()"), -0.05, {"success": False}   

    def execute_code(self, instance_id, code, timeout=30, language="python", fetch_files=['image.txt']):
        modified_code = wrap_code_with_capture(code)
        result_status, metadata = _process_single_case(
            0, None, None, self.sandbox_fusion_url, modified_code, timeout, self.memory_limit_mb, language, fetch_files
        )
        # we should always expect this since we don't have correct answer
        if metadata["run_status"] == "Finished":
            actual_output = metadata["stdout"] + metadata["stderr"]
            logger.debug(f"actual_output from sandbox fusion: {actual_output},{instance_id}")
            return metadata
        else:
            return "api error"

    async def calc_reward(self, instance_id: str, **kwargs) -> dict:

        code_score = self._instance_dict[instance_id].get("reward", [])
        avg_score = float(sum(code_score)) if code_score else 0.0
        return avg_score

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
