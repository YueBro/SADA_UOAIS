import torch
from torch import nn
from .grad_modules import GradReverse


__all__ = [
    "BackboneFeature_Discriminator",
    "HierarchyFeature_Discriminator",
]


class BackboneFeature_Discriminator(nn.Module):

    def __init__(self):
        super().__init__()

        self.layers = {
            "p2": StrideModule(256, channel_division=4, stride_n=5),
            "p3": StrideModule(256, channel_division=4, stride_n=4),
            "p4": StrideModule(256, channel_division=4, stride_n=3),
            "p5": StrideModule(256, channel_division=4, stride_n=2),
            "p6": StrideModule(256, channel_division=4, stride_n=1),
        }

        self.flatten_lyr = nn.Flatten()

        self.domain_classifier = nn.Sequential(
            # nn.Linear(5*16*4*5, 128),
            # nn.ReLU(),
            # nn.Linear(128, 128),
            # nn.ReLU(),
            # nn.Linear(128, 1),
            # nn.Sigmoid(),

            nn.Linear(5*16*4*5, 1),
            nn.Sigmoid(),
        )

        self.loss_fn = nn.BCELoss()

        self.layers = nn.ModuleDict(self.layers)

    def forward(self, feature_maps, labels):
        B = feature_maps["p2"].shape[0]
        if isinstance(labels, int):
            labels = torch.ones(size=(B,1)) if (labels==1 or labels is True) else torch.zeros(size=(B,1))
            if feature_maps["p2"][0].is_cuda is True:
                labels = labels.cuda()

        keys = ["p2", "p3", "p4", "p5", "p6"]
        features = []

        for key in keys:
            feature_map = feature_maps[key]
            feature_map_output = self.layers[key](feature_map)
            features.append(feature_map_output)
        
        features = torch.cat(features, dim=1)
        features = self.flatten_lyr(features)
        
        logits = self.domain_classifier(features)

        loss = {"domain_adapt_loss": self.loss_fn(logits, labels)}

        return logits, loss


class HierarchyFeature_Discriminator(nn.Module):
    def __init__(self, input_list_size: int):
        super().__init__()
        self.input_list_size = input_list_size

        layers = [self.sub_module() for _ in range(input_list_size)]
        self.layers = nn.ModuleList(layers)

        self.loss_fn = nn.BCELoss()

    def sub_module(self):
        module = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, stride=2),  # -> [B, 128, 7, 7]
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=2),  # -> [B, 64, 4, 4]
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, stride=2),  # -> [B, 32, 2, 2]
            nn.ReLU(),

            nn.Flatten(),

            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        return module
    
    def forward(self, x: list, label=None):
        assert len(x) == len(self.layers)
        if x[0].shape[0] == 0:
            return torch.tensor(0.0).to(x[0].device)

        x = [GradReverse.apply(_x) for _x in x]

        assert len(x)==len(self.layers)
        results = [layer(_x) for _x, layer in zip(x, self.layers)]

        if self.training is True:
            if label is False:
                y = torch.zeros_like(results[0])
            elif label is True:
                y = torch.ones_like(results[0])
            else:
                raise ValueError("\"label\" must be specified as True or False.")
            
            loss = self.loss_fn(results[0], y) + self.loss_fn(results[1], y)

            return loss

        else:
            return results


class StrideModule(nn.Module):
    def __init__(self, in_channels, channel_division, stride_n) -> None:
        super().__init__()
        self._chn_div = channel_division
        self._n_chn = in_channels
        
        self.layers = nn.Sequential()
        
        while stride_n > 0:
            old_chn, new_chn = self._next_channel_sizes()
            self.layers.add_module(f"lyr{len(self.layers)}", nn.Conv2d(
                old_chn, new_chn, kernel_size=3, padding=1, stride=2))
            self.layers.add_module(f"lyr{len(self.layers)}", nn.LeakyReLU())
            # self.layers.add_module(f"lyr{len(self.layers)}", nn.BatchNorm2d(new_chn))
            stride_n -= 1


        if self._chn_div > 0:
            while self._chn_div > 0:
                old_chn, new_chn = self._next_channel_sizes()
                self.layers.add_module(f"lyr{len(self.layers)}", nn.Conv2d(
                    old_chn, new_chn, kernel_size=3, padding=1))
                self.layers.add_module(f"lyr{len(self.layers)}", nn.LeakyReLU())
            # self.layers.add_module(f"lyr{len(self.layers)}", nn.BatchNorm2d(new_chn))
    
    def forward(self, x):
        return self.layers(x)

    def _next_channel_sizes(self):
        old_chn = self._n_chn
        new_chn = None
        if self._chn_div > 0:
            self._n_chn //= 2
            self._chn_div -= 1
            new_chn = self._n_chn
        else:
            new_chn = old_chn
        return old_chn, new_chn

