from functionals.functionals import Conv2D

# create a class of CNN model
class SimpleCNN():
    def __init__(self, config):
        super(SimpleCNN, self).__init__()

        self.total_latency = 0
        self.total_energy = 0

        self.conv1 = Conv2D(weight=(3,3,3), bias=(3,), stride=(2,2), 
                            padding=(0,0), config=config)
        self.conv2 = Conv2D(weight=(3,3,4), bias=(4,), stride=(2,2),
                            padding=(0,0), config=config)


    def forward(self, x):
        x, layer_latency, layer_energy = self.conv1.forward(x)
        self.accumu_laten_energy(layer_latency, layer_energy)

        x, layer_latency, layer_energy = self.conv2.forward(x)
        self.accumu_laten_energy(layer_latency, layer_energy)

        return x, self.total_latency, self.total_energy
    
    # accumulate the latency and energy
    def accumu_laten_energy(self, latency, energy):
        self.total_latency += latency
        self.total_energy += energy
    
    # set weights, bias for a specific layer
    def set_weights_bias(self, attribute_name, weight, bias):
        # get attribute by name
        attribute = getattr(self, attribute_name)
        # set weight and bias
        attribute.weight = weight
        attribute.bias = bias
