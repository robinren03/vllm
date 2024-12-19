from typing import List
from vllm.preserve.session_config import SessionConfig

def sp_at_time(config:SessionConfig, model_max_len: int, t:float):
    t0 = config.t0
    point1 = config.tau * (model_max_len-config.ip) / config.p
    point2 = config.tau * config.rounds
    
    new_t = t - t0
    if new_t > point2:
        return 0
    elif new_t > point1:
        return model_max_len
    return config.ip + config.p * new_t / config.tau

def sp_breaking_point(config:SessionConfig, model_max_len: int):
    t0 = config.t0
    point1 = round(config.tau * (model_max_len-config.ip) / config.p + t0, 2)
    point2 = round(config.tau * config.rounds + t0, 2)

    if point1 < point2:
        return [point1-0.01, point1+0.01, point2-0.01, point2+0.01]
    else:
        return [point2-0.01, point2+0.01]
    
def sum_sps(configs:List[SessionConfig], model_max_len: int, current_time:float, num_gpu_blocks:int):
    time_points = set([current_time])
    for config in configs:
        # print("config: ", config.ip, config.p, config.tau, config.rounds)    
        time_points.update(sp_breaking_point(config, model_max_len))
    time_points = sorted(list(time_points))
    # print("time_points: ", time_points)
    time_val = []
    prev_pair = (0, 0)
    ahead_pair = (-1, -1000)
    for tp in time_points:
        if tp < current_time - 0.01:
            continue

        new_pair = (tp, round(sum([sp_at_time(config, model_max_len, tp) for config in configs]), 3))
        if abs(prev_pair[1] - ahead_pair[1]) + abs(ahead_pair[1] - new_pair[1]) < 128: 
            # TODO (yanyu): 128 = block_size * 8, and 8 is a magic number
            time_val[-1] = new_pair
            prev_pair = new_pair 
        else:
            ahead_pair = prev_pair
            prev_pair = new_pair
            time_val.append(new_pair)
    
    return time_val
    
