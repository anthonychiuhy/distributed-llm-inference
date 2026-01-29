import os
from fastapi import FastAPI
import uvicorn
from .main import DataLoader, Scheduler, MetricCollector, TrafficGenerator


app = FastAPI()

IP_ADDR = os.environ['IP_ADDR']
config = {
    'trace_path': 'data/trace1.csv',
    'data_path': 'data/conversations.json',
    'log_path': 'logs/log.json',
    'max_prefill_prompt_len': 10000,
    'max_prefill_gen_len': 10000,
    'url': f'http://{IP_ADDR}:8000/v1/chat/completions',
    'model': 'google/gemma-3-1b-it',
    'temperature': 0.7,
    'max_tokens': 8192
}

@app.get("/")
def send_traffic():
    print(f"Sending traffic to {config['url']}")

    data = DataLoader().get_data_from_path(data_path=config['data_path'])
    schedule = Scheduler().get_schedule_from_trace(trace_path=config['trace_path'])
    logger = MetricCollector()
    generator = TrafficGenerator(data=data, schedule=schedule, config=config, logger=logger)
    generator.start_profile()

    logger.save(path=config['log_path'])

    return "traffic sent completed"

if __name__ == "__main__":
    uvicorn.run("traffic_generator.app:app", port=8080, host="0.0.0.0", log_level="info")