class ARGS:
    def __init__(self):
        self.config_file = ""
        self.resume = False
        self.eval_only = False
        self.num_gpus = 1
        self.num_machines = 1
        self.machine_rank = 0
        self.dist_url = "tcp://127.0.0.1:50152"
        self.opts = []
