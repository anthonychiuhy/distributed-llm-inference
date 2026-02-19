import os
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from .main import DataLoader, Scheduler, MetricCollector, TrafficGenerator
from .traffic_simulation import TrafficLoad


# Environment-driven configuration with sensible defaults
IP_ADDR = os.environ.get("IP_ADDR", "vllm-service")
VLLM_PORT = os.environ.get("VLLM_PORT", "8080")
VLLM_URL = os.environ.get("VLLM_URL", f"http://{IP_ADDR}:{VLLM_PORT}/v1/chat/completions")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-Coder-Next")
SCHEDULE_PATH = os.environ.get("SCHEDULE_PATH", "schedules/schedule1.csv")
DATA_PATH = os.environ.get("DATA_PATH", "data/ShareGPT_V3_processed.json")
LOG_PATH = os.environ.get("LOG_PATH", "logs/log.json")
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "8192"))
STREAM = os.environ.get("STREAM", "false").lower() in ("1", "true", "yes")

config = {
    "schedule_path": SCHEDULE_PATH,
    "data_path": DATA_PATH,
    "log_path": LOG_PATH,
    "url": VLLM_URL,
    "model": MODEL_NAME,
    "temperature": TEMPERATURE,
    "max_tokens": MAX_TOKENS,
    "stream": STREAM,
}

data = DataLoader().get_data_from_path(data_path=config['data_path'])
generator = TrafficGenerator(data=data, config=config)


app = FastAPI()

class Params(BaseModel):
    alpha_0: float
    alpha_s: float
    alpha_r: float
    alpha_g: float
    t_s: float
    t_r: float
    t_g: float
    tau_r: float
    sigma: float
    duration: float


@app.get("/")
def send_schedule_traffic():
    print(f"Sending traffic to {config['url']}")

    schedule = Scheduler().get_schedule_from_path(schedule_path=config['schedule_path'])

    logger = MetricCollector()
    generator.start_profile(schedule, logger)

    logger.save(path=config['log_path'])

    return "traffic sent completed"


@app.post("/")
def send_traffic_load_function(params: Params):
    traffic_load_func = TrafficLoad(
        alpha_0=params.alpha_0,
        alpha_s=params.alpha_s,
        alpha_r=params.alpha_r,
        alpha_g=params.alpha_g,
        t_s=params.t_s,
        t_r=params.t_r,
        t_g=params.t_g,
        tau_r=params.tau_r,
        sigma=params.sigma
    )
    print(f"Sending traffic to {config['url']}")
    print(f"Using traffic load function {traffic_load_func}")

    file_prefix = f"{repr(traffic_load_func)};t=[0,{params.duration}]"
    schedule_path = f"schedules/{file_prefix}.csv"

    if os.path.isfile(schedule_path):
        schedule = Scheduler().get_schedule_from_path(schedule_path=schedule_path)
    else:
        schedule = Scheduler().get_schedule_from_traffic_load_function(traffic_load_func, params.duration, data, save_path=schedule_path)

    logger = MetricCollector()
    generator.start_profile(schedule, logger)

    logger.save(path=f'logs/{file_prefix}.json')

    return "traffic sent completed"


if __name__ == "__main__":
    uvicorn.run("traffic_generator.app:app", port=8080, host="0.0.0.0", log_level="info")