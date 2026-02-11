import os
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from .main import DataLoader, Scheduler, MetricCollector, TrafficGenerator
from .traffic_simulation import TrafficLoad


IP_ADDR = os.environ['IP_ADDR']

config = {
    'data_path': 'data/conversations.json',
    'max_prefill_prompt_len': 10000,
    'max_prefill_gen_len': 10000,
    'url': f'http://{IP_ADDR}:8000/v1/chat/completions',
    'model': 'google/gemma-3-1b-it',
    'temperature': 0.7,
    'max_tokens': 8192
}


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
def send_default_traffic():
    print(f"Send traffic to {config['url']}")

    data = DataLoader().get_data_from_path(data_path=config['data_path'])
    schedule = Scheduler().get_schedule_from_trace(trace_path='schedules/trace1.csv')
    logger = MetricCollector()
    generator = TrafficGenerator(data=data, schedule=schedule, config=config, logger=logger)
    generator.start_profile()

    logger.save(path='logs/trace1.json')

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
    print(f"Send traffic to {config['url']}")
    print(f"Using traffic load function {traffic_load_func}")

    data = DataLoader().get_data_from_path(data_path=config['data_path'])

    file_prefix = f"{repr(traffic_load_func)};t=[0,{params.duration}]"
    save_path = f"schedules/{file_prefix}.csv"

    if os.path.isfile(save_path):
        schedule = Scheduler().get_schedule_from_trace(trace_path=save_path)
    else:
        schedule = Scheduler().get_schedule_from_traffic_load_function(traffic_load_func, params.duration, data, save_path=save_path)
    
    logger = MetricCollector()
    generator = TrafficGenerator(data=data, schedule=schedule, config=config, logger=logger)

    
    generator.start_profile()

    logger.save(path=f'logs/{file_prefix}.json')

    return "traffic sent completed"

if __name__ == "__main__":
    uvicorn.run("traffic_generator.app:app", port=8080, host="0.0.0.0", log_level="info")