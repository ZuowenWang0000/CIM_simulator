from functionals.functionals import Conv2D, BatchNorm2D, I_NL_ReLU
import numpy as np

class layer_block():
    def __init__(self, layers: list):
        for layer in layers:
            setattr(self, f"{layer.__class__.__name__}", layer)

# create a class of CNN model
class SimpleNVPCNN():
    def __init__(self, config,I_NL=None):
        super(SimpleNVPCNN, self).__init__()

        self.total_latency = 0
        self.total_energy = 0
        self.total_ops = 0
        self.I_NL = I_NL

        # set Conv2D class parameter I_NL with self.I_NL
        Conv2D.I_NL = self.I_NL

        # omitting the batch norm layers and relu layers

        # the initial 4 conv layers
        self.conv1 = Conv2D(weight=(8,3,3), bias=(3,), stride=(2,2), 
                            padding=(1,1), config=config)
        self.bn1 = BatchNorm2D(running_mean=np.zeros(8), running_var=np.ones(8), 
                               gamma=1, beta=0, config=config)
        self.conv2 = Conv2D(weight=(16,3,3), bias=(16,), stride=(1,1),
                            padding=(1,1), config=config)
        self.bn2 = BatchNorm2D(running_mean=np.zeros(16), running_var=np.ones(16), 
                               gamma=1, beta=0, config=config)
        self.conv3 = Conv2D(weight=(28,3,3), bias=(28,), stride=(2,2),
                            padding=(1,1), config=config)
        self.bn3 = BatchNorm2D(running_mean=np.zeros(28), running_var=np.ones(28), 
                               gamma=1, beta=0, config=config)
        self.conv4 = Conv2D(weight=(28,3,3), bias=(28,), stride=(1,1),
                            padding=(1,1), config=config)
        self.bn4 = BatchNorm2D(running_mean=np.zeros(28), running_var=np.ones(28), 
                               gamma=1, beta=0, config=config)
        #  the 4 residual blocks, each has 2 conv layers
        for i in range(5,9):
            setattr(self, f"conv{i}_a", Conv2D(weight=(28,3,3), bias=(28,), stride=(1,1),
                            padding=(1,1), config=config))
            setattr(self, f"bn{i}_a", BatchNorm2D(running_mean=np.zeros(28), running_var=np.ones(16), 
                               gamma=1, beta=0, config=config))
            setattr(self, f"conv{i}_b", Conv2D(weight=(28,3,3), bias=(28,), stride=(1,1),
                            padding=(1,1), config=config))
            setattr(self, f"bn{i}_b", BatchNorm2D(running_mean=np.zeros(28), running_var=np.ones(16), 
                               gamma=1, beta=0, config=config))

        # the final conv layers
        self.conv9 = Conv2D(weight=(16,3,3), bias=(16,), stride=(1,1),
                    padding=(1,1), config=config)
        self.bn9 = BatchNorm2D(running_mean=0, running_var=1, gamma=1, beta=0, config=config)

        self.conv10 = Conv2D(weight=(1,3,3), bias=(1,), stride=(1,1),
                    padding=(1,1), config=config)
        
        self.I_NL_ReLU = I_NL_ReLU

    def forward(self, x):
        for first_stage_block in [(getattr(self, f'conv{i}'), getattr(self, f'bn{i}'), self.I_NL_ReLU) for i in range(1,5)]:
            for layer in first_stage_block:
                x, layer_latency, layer_energy, layer_ops = layer.forward(x)
                self.accumu_laten_energy(layer_latency, layer_energy, layer_ops)

        for residual_layer_block in [(getattr(self, f'conv{i}_a'), getattr(self, f'bn{i}_a'), self.I_NL_ReLU,\
                                      getattr(self, f'conv{i}_b'), getattr(self, f'bn{i}_b'), self.I_NL_ReLU) for i in range(5,9)]:
            for layer in residual_layer_block:
                x, layer_latency, layer_energy, layer_ops = layer.forward(x)
                self.accumu_laten_energy(layer_latency, layer_energy, layer_ops)

        for last_stage_layer in [self.conv9, self.bn9, self.I_NL_ReLU, self.conv10]:
            x, layer_latency, layer_energy, layer_ops = last_stage_layer.forward(x)
            self.accumu_laten_energy(layer_latency, layer_energy, layer_ops)

        return x, self.total_latency, self.total_energy, self.total_ops
    
    # accumulate the latency and energy
    def accumu_laten_energy(self, latency, energy, ops):
        self.total_latency += latency
        self.total_energy += energy
        self.total_ops += ops
    
    # set weights, bias for a specific layer
    def set_weights_bias(self, attribute_name, weight, bias):
        # get attribute by name
        attribute = getattr(self, attribute_name)
        # set weight and bias
        attribute.weight = weight
        attribute.bias = bias
