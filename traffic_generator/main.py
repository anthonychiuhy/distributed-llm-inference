import os
from time import perf_counter
import json
# import argparse
import asyncio
import aiohttp
import numpy as np
import pandas as pd

# from langchain_ollama import ChatOllama


class SteadyUser:
    def __init__(self, name: str, req_freq: float, duration: float, delay_start: float = 0.0):
        self.name = name
        self.req_freq = req_freq
        self.duration = duration
        self.delay_start = delay_start
    
    def get_timestamps(self) -> list[float]:
        timestamps = []
        interval = 1.0 / self.req_freq
        t = 0.0
        while t <= self.duration:
            timestamps.append(t + self.delay_start)
            t += interval
        return timestamps


class BurstUser:
    def __init__(self, name: str, n_req: int, time: float):
        self.name = name
        self.n_req = n_req
        self.time = time
    
    def get_timestamps(self) -> list[float]:
        return [self.time] * self.n_req


class DataLoader:
    def __init__(self, config=None):
        self.config = config

    @staticmethod
    def load_json_from_path(file_path: str):
        with open(file_path, "r") as f:
            return json.load(f)
    
    def get_data_from_path(self, data_path: str) -> list[tuple]:
        data = self.load_json_from_path(data_path)
        return [(d["prompt"], d["len_prompt"], d["len_output"], d['output']) for d in data.values()]

class Scheduler:
    def __init__(self, config=None):
        self.config = config

    def get_schedule_from_trace(self, trace_path: str, max_trace: int) -> pd.DataFrame:
        return pd.read_csv(
            trace_path,
            nrows=max_trace,
            dtype={
                "Timestamp": float,
                "Request tokens": int,
                "Response tokens": int
            }
        )

    def get_schedule_from_users(self, users: list[SteadyUser | BurstUser]) -> pd.DataFrame:
        REQUEST_TOKENS = 500
        RESPONSE_TOKENS = 500
        
        dfs = []
        for user in users:
            timestamps = user.get_timestamps()
            dfs.append(pd.DataFrame(
                {
                    'Timestamp': timestamps,
                    'Request tokens': [REQUEST_TOKENS] * len(timestamps),
                    'Response tokens': [RESPONSE_TOKENS] * len(timestamps),
                    'User': [user.name] * len(timestamps)
                }
            ))

        return pd.concat(dfs).reset_index(drop=True)

class Query:
    def __init__(self, inputs: list, schedule: pd.DataFrame):
        self.inputs = inputs
        self.schedule = schedule.sort_values(by='Timestamp').reset_index(drop=True)
        self.query_id = -1
        self.query_time = 0
        self.max_prompt_len = MAX_PROMPT_LEN
        self.max_gen_len = MAX_GEN_LEN
        self.prefill_idx = self.get_prefill_idx()

    @staticmethod
    def _fill_missing_idx(arr, missing):
        n = len(arr)
        
        dist_to_left = [n] * n
        i = 0
        while i < n and arr[i] == missing:
            i += 1
        # if all missings then just return
        if i == n:
            return
        for j in range(i, n):
            if arr[j] == missing:
                dist += 1
            else:
                dist = 0
            dist_to_left[j] = dist
        
        dist_to_right = [n] * n
        i = n - 1
        while arr[i] == missing:
            i -= 1
        for j in range(i, -1, -1):
            if arr[j] == missing:
                dist += 1
            else:
                dist = 0
            dist_to_right[j] = dist
            
        for i in range(n):
            if dist_to_left[i] <= dist_to_right[i]:
                arr[i] = arr[i - dist_to_left[i]]
            else:
                arr[i] = arr[i + dist_to_right[i]]

    def get_prefill_idx(self):
        prefill_idx = np.ones((self.max_prompt_len+1, self.max_gen_len+1), dtype=int) * (-1)
        prompt_exist = np.zeros(self.max_prompt_len+1, dtype=bool)

        # prefill record
        for idx, data in enumerate(self.inputs):
            len_prompt = data[1]
            len_output = data[2]
            if len_prompt <= self.max_prompt_len and len_output <= self.max_gen_len:
                prefill_idx[len_prompt, len_output] = idx
                prompt_exist[len_prompt] = True

        # fill in missing row values
        for idx_ii in np.where(prompt_exist)[0]:
            self._fill_missing_idx(prefill_idx[idx_ii], missing=-1)
        
        # fill in missing rows
        row_idx_arr = prompt_exist * np.arange(self.max_prompt_len+1)
        self._fill_missing_idx(row_idx_arr, missing=0)

        missing_row_idx_arr = np.where(~prompt_exist)[0]
        prefill_idx[missing_row_idx_arr] = prefill_idx[row_idx_arr[missing_row_idx_arr]]

        return prefill_idx

    def get_query(self):
        # Use the trace
        self.query_id += 1

        self.query_time = self.schedule.at[self.query_id, 'Timestamp'].item()

        sampled_prompt_len = self.schedule.at[self.query_id, 'Request tokens'].item()
        sampled_prompt_len = min(sampled_prompt_len, self.max_prompt_len)
        sampled_output_len = self.schedule.at[self.query_id, 'Response tokens'].item()
        sampled_output_len = min(sampled_output_len, self.max_gen_len)

        sampled = self.inputs[self.prefill_idx[sampled_prompt_len][sampled_output_len]]

        return [
            sampled[0], # prompt
            sampled[1], # prompt input length
            sampled[2], # prompr output length
            self.query_id,
            self.query_time
        ]
    
    def reset(self):
        self.query_id = -1
        self.query_time = 0

    def __len__(self):
        return len(self.schedule)

class MetricCollector:
    def __init__(self):
        self.trace_config = TraceConfig()
        self.metrics = {}
    
    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.metrics, f)

class TraceConfig(aiohttp.TraceConfig):
    def __init__(self):
        super().__init__()
        self.on_request_start.append(self.on_request_start_callback)
        self.on_request_end.append(self.on_request_end_callback)
        self.on_request_exception.append(self.on_request_exception_callback)

    async def on_request_start_callback(self, session, ctx, params):
        # request start
        logger = ctx.trace_request_ctx['logger']
        query_id = ctx.trace_request_ctx['query_id']
        request_start_time = perf_counter() - logger.session_start_timestamp

        logger.metrics[query_id]['request_start_time'] = request_start_time

        print(f"[START] ID: {query_id}, Start: {request_start_time:.1f}")

    async def on_request_end_callback(self, session, ctx, params):
        # response status line and headers received
        logger = ctx.trace_request_ctx['logger']
        query_id = ctx.trace_request_ctx['query_id']

        logger.metrics[query_id]['response_headers_received_time'] = perf_counter() - logger.session_start_timestamp
    
    async def on_request_exception_callback(self, session, ctx, params):
        # request exception raised
        query_id = ctx.trace_request_ctx['query_id']
        logger.metrics[query_id]['response_headers_received_time'] = None

        print(f"[ERROR] ID: {query_id}, Request Exception")

# sending token rate  = (number of tokens sent / ackknowledge time)
# time to first token
# check queue is on the server side, need to check acknowledge time from server.
# 


class TrafficGenerator:
    """Generates LLM inference traffic and send it to inference endpoint"""
    def __init__(self, data: list, schedule: pd.DataFrame, config: dict, logger: MetricCollector):
        self.queries = Query(inputs=data, schedule=schedule)
        self.config = config
        self.logger = logger

        print(self.queries.schedule)

    async def inference_call(self, session, prompt, sleep_time, query_id):
        # Single inference call
        payload = {
            "model": self.config['model'],
            "prompt": prompt,
            "temperature": self.config['temperature'],
            "max_tokens": self.config['max_tokens'],
            "stream": STREAM
        }
        url = self.config['url']
        trace_request_ctx = {'query_id':query_id, 'logger':self.logger}

        success = False
        response_end_time = None
        first_token_arrive_time = None

        await asyncio.sleep(sleep_time)
        try:
            async with session.post(url, json=payload, trace_request_ctx=trace_request_ctx) as resp:
                resp.raise_for_status()
                first = True
                async for _ in resp.content:
                    if first:
                        first_token_arrive_time = perf_counter() - self.logger.session_start_timestamp
                        first = False
            success = True
            response_end_time = perf_counter() - self.logger.session_start_timestamp

            print(f"[END] ID: {query_id}, End: {response_end_time:.1f}, turnaround: {response_end_time - self.logger.metrics[query_id]['request_start_time']:.1f}")

        except aiohttp.ClientResponseError as e:
            print(f"ClientResponseError: {e}")
        except aiohttp.ClientConnectionError as e:
            print(f"ClientConnectionError: {e}")

        self.logger.metrics[query_id]['first_token_arrive_time'] = first_token_arrive_time
        self.logger.metrics[query_id]['response_end_time'] = response_end_time
        self.logger.metrics[query_id]['scheduled_start_time'] = sleep_time
        self.logger.metrics[query_id]['success'] = success

    async def issue_queries(self):
        # Multiple concurrent inference call
        async with aiohttp.ClientSession(trace_configs=[self.logger.trace_config]) as session:
            task_list = []
            for _ in range(len(self.queries)):
                prompt, in_num, out_num, query_id, sleep_time = self.queries.get_query()
                task_list.append(self.inference_call(session, prompt, sleep_time, query_id))
                
                self.logger.metrics[query_id] = {} # initialise
                self.logger.metrics[query_id]['number_of_input_tokens'] = in_num
            self.logger.session_start_timestamp = perf_counter()
            await asyncio.gather(*task_list)

    def start_profile(self):
        self.queries.reset()
        asyncio.run(self.issue_queries())



MAX_PROMPT_LEN = 1024
MAX_GEN_LEN = 1024
STREAM = True

config = {
    'trace_path': '../data/trace1.csv',
    'data_path': '../data/conversations.json',
    'max_trace': 100,
    'url': 'http://10.215.130.20:11434/api/generate', # OR 172.25.149.93
    'no_proxy': '10.215.130.20',
    'model': 'mistral',
    'temperature': 0.7,
    'max_tokens': 200,
    'save_log': False,
    'log_path': '../logs/log.json'
}

if __name__ == "__main__":
    # os.environ["NO_PROXY"] = config['no_proxy']

    data = DataLoader().get_data_from_path(data_path=config['data_path'])

    schedule = Scheduler().get_schedule_from_trace(trace_path=config['trace_path'], max_trace=config['max_trace'])

    logger = MetricCollector()

    # user1 = SteadyUser(name='u1', req_freq=1.0, duration=10.0, delay_start=0.0)
    # user2 = SteadyUser(name='u2', req_freq=1.0, duration=10.0, delay_start=0.3)
    # user3 = SteadyUser(name='u3', req_freq=1.0, duration=10.0, delay_start=0.6)
    # user4 = BurstUser(name='u4', n_req=5, time=5.5)
    # user5 = BurstUser(name='u5', n_req=5, time=2.5)
    # users = [user1, user2, user3, user4, user5]
    # schedule = Scheduler().get_schedule_from_users(users=users)

    # llm = ChatOllama(
    #     model=config['model'],
    #     base_url=config['host'],
    #     temperature=config['temperature'],
    #     num_predict=config['max_token']
    # )

    generator = TrafficGenerator(data=data, schedule=schedule, config=config, logger=logger)
    generator.start_profile()

    print(logger.metrics)
    logger.save(path=config['log_path'])