class DALossScaleStepScheduler:
    def __init__(self, cfg):
        self.step_points = cfg.DOMAIN_ADAPTATION.DOMAIN_LOSS_SCALER_SCHEDULE[::2]
        self.step_idx = 0
        self.step_values = cfg.DOMAIN_ADAPTATION.DOMAIN_LOSS_SCALER_SCHEDULE[1::2]
        self.iter = 0
    
    def get_value_and_step(self):
        value = self.step_values[self.step_idx]
        self.step()
        return value
    
    def step(self):
        self.iter += 1
        if self.step_idx+1 < len(self.step_points):
            if self.iter >= self.step_points[self.step_idx+1]:
                self.step_idx += 1
