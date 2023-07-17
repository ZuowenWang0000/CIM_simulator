from functionals.functionals import Conv2D

# create a class of CNN model
class SimpleNVPCNN():
    def __init__(self, config):
        super(SimpleNVPCNN, self).__init__()

        self.total_latency = 0
        self.total_energy = 0
        self.total_ops = 0

        # omitting the batch norm layers and relu layers

        # the initial 4 conv layers
        self.conv1 = Conv2D(weight=(8,3,3), bias=(3,), stride=(2,2), 
                            padding=(1,1), config=config)
        self.conv2 = Conv2D(weight=(16,3,3), bias=(16,), stride=(1,1),
                            padding=(1,1), config=config)
        self.conv3 = Conv2D(weight=(24,3,3), bias=(24,), stride=(2,2),
                            padding=(1,1), config=config)
        self.conv4 = Conv2D(weight=(24,3,3), bias=(24,), stride=(1,1),
                            padding=(1,1), config=config)

        #  the 4 residual blocks, each has 2 conv layers
        self.conv5_a = Conv2D(weight=(24,3,3), bias=(24,), stride=(1,1),
                            padding=(1,1), config=config)
        self.conv5_b = Conv2D(weight=(24,3,3), bias=(24,), stride=(1,1),
                    padding=(1,1), config=config)
        
        self.conv6_a = Conv2D(weight=(24,3,3), bias=(24,), stride=(1,1),
                            padding=(1,1), config=config)
        self.conv6_b = Conv2D(weight=(24,3,3), bias=(24,), stride=(1,1),
                    padding=(1,1), config=config)
        
        self.conv7_a = Conv2D(weight=(24,3,3), bias=(24,), stride=(1,1),
                            padding=(1,1), config=config)
        self.conv7_b = Conv2D(weight=(24,3,3), bias=(24,), stride=(1,1),
                    padding=(1,1), config=config)
        
        self.conv8_a = Conv2D(weight=(24,3,3), bias=(24,), stride=(1,1),
                            padding=(1,1), config=config)
        self.conv8_b = Conv2D(weight=(24,3,3), bias=(24,), stride=(1,1),
                    padding=(1,1), config=config)

        # the final conv layers
        self.conv9 = Conv2D(weight=(16,3,3), bias=(16,), stride=(1,1),
                    padding=(1,1), config=config)

        self.conv10 = Conv2D(weight=(1,3,3), bias=(1,), stride=(1,1),
                    padding=(1,1), config=config)

    def forward(self, x):
        x, layer_latency, layer_energy, layer_ops = self.conv1.forward(x)
        self.accumu_laten_energy(layer_latency, layer_energy, layer_ops)

        x, layer_latency, layer_energy, layer_ops = self.conv2.forward(x)
        self.accumu_laten_energy(layer_latency, layer_energy, layer_ops)

        x, layer_latency, layer_energy, layer_ops = self.conv3.forward(x)
        self.accumu_laten_energy(layer_latency, layer_energy, layer_ops)

        x, layer_latency, layer_energy, layer_ops = self.conv4.forward(x)
        self.accumu_laten_energy(layer_latency, layer_energy, layer_ops)

        x, layer_latency, layer_energy, layer_ops = self.conv5_a.forward(x)
        self.accumu_laten_energy(layer_latency, layer_energy, layer_ops)

        x, layer_latency, layer_energy, layer_ops = self.conv5_b.forward(x)
        self.accumu_laten_energy(layer_latency, layer_energy, layer_ops)

        x, layer_latency, layer_energy, layer_ops = self.conv6_a.forward(x)
        self.accumu_laten_energy(layer_latency, layer_energy, layer_ops)

        x, layer_latency, layer_energy, layer_ops = self.conv6_b.forward(x)
        self.accumu_laten_energy(layer_latency, layer_energy, layer_ops)

        x, layer_latency, layer_energy, layer_ops = self.conv7_a.forward(x)
        self.accumu_laten_energy(layer_latency, layer_energy, layer_ops)

        x, layer_latency, layer_energy, layer_ops = self.conv7_b.forward(x)
        self.accumu_laten_energy(layer_latency, layer_energy, layer_ops)

        x, layer_latency, layer_energy, layer_ops = self.conv8_a.forward(x)
        self.accumu_laten_energy(layer_latency, layer_energy, layer_ops)

        x, layer_latency, layer_energy, layer_ops = self.conv8_b.forward(x)
        self.accumu_laten_energy(layer_latency, layer_energy, layer_ops)

        x, layer_latency, layer_energy, layer_ops = self.conv9.forward(x)
        self.accumu_laten_energy(layer_latency, layer_energy, layer_ops)

        x, layer_latency, layer_energy, layer_ops = self.conv10.forward(x)
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
