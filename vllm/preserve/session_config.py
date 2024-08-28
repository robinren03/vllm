class SessionConfig:
    def __init__(self, ip: int, p: float, sum_p:int, tau: float, current_time: float, rounds: float):
        self.ip = ip if ip>0 else sum_p
        self.p = p
        self.prev_p = sum_p
        self.prev_time = current_time
        self.tau = tau
        self.t0 = current_time
        self.rounds = rounds
    
    def update(self, sum_p: int, current_time:float, session_reuse: int, rounds: float = -1):
        prev_p = self.prev_p
        delta_p = sum_p - prev_p
        tau = current_time - self.prev_time
        if prev_p - self.p >= sum_p:
            # restoration happens, go back
            self.ip = prev_p
            self.t0 = current_time
        elif session_reuse > 2:
            self.p = (self.p * 2 + delta_p) / 3
        
        self.tau = (tau + self.tau * 2 ) / 3
        
        self.prev_p = sum_p
        self.prev_time = current_time

        if (rounds > 0): self.rounds = rounds