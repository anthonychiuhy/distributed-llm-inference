import numpy as np

class TrafficLoad:
    """
    Defines the Traffic Load Function.
    """
    def __init__(self, alpha_0, alpha_s, alpha_r, alpha_g, t_s, t_r, t_g, tau_r, sigma):
        assert alpha_0 >= 0
        assert alpha_s >= 0
        assert alpha_r >= 0
        assert alpha_g >= 0
        assert tau_r > 0 # 1/tau_r must be a positive slope

        self.alpha_0 = alpha_0
        self.alpha_s = alpha_s
        self.alpha_r = alpha_r
        self.alpha_g = alpha_g
        self.t_s = t_s
        self.t_r = t_r
        self.t_g = t_g
        self.tau_r = tau_r
        self.sigma = sigma

        # Upper bound on maximum possible rate
        self.rate_max = self.alpha_0 + self.alpha_s + self.alpha_r + self.alpha_g
    
    def step(self, t):
        return float(t - self.t_s >= 0) # step(t_s) defined as 1

    def ramp(self, t):
        return min(1, max(0, (t - self.t_r) / self.tau_r))

    def gaussian(self, t):
        return np.exp(-0.5 * ((t - self.t_g) / self.sigma) ** 2)

    def __call__(self, t):
        return self.alpha_0 + self.alpha_s * self.step(t) + self.alpha_r * self.ramp(t) + self.alpha_g * self.gaussian(t)

def simulate_nonhomogeneous_poisson(rate, t_end, rng=None):
    """
    Generate samples from a non-homogeneous Poisson process with a given rate function.
    """
    rng = np.random.default_rng() if rng is None else rng
    rate_max = rate.rate_max
    times = []
    t = 0
    while True:
        # Propose next inter-arrival time from exponential distribution at the max rate.
        t += rng.exponential(1 / rate_max)
        if t > t_end:
            break
        # Poisson process thinning. Accepted samples distributed as Poisson(rate(t)) with p = rate(t) / rate_max.
        if rng.uniform(0, 1) < rate(t) / rate_max:
            times.append(t)
    return times