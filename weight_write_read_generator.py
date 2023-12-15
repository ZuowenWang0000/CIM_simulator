import random

def generate_random_weights(num_weights, bit_size=4):
    """
    Generate a series of weight data.
    Weights are signed integers
    range is -2^(bit_size-1) to 2^(bit_size-1) - 1.
    """
    min_val = -2**(bit_size - 1)
    max_val = 2**(bit_size - 1) - 1

    # Convert the weight to a 4-bit binary representation
    # Assuming the most significant bit (MSB) is the sign bit
    weights = [random.randint(min_val, max_val) for _ in range(num_weights)]
    return [format(weight& 0b1111, '04b') for weight in weights]

def write_weights_to_file(weights, file_path):
    """
    Write the weight data to a file.
    """
    with open(file_path, 'w') as file:
        for weight in weights:
            file.write(f"{weight}\n")




# IMC specific parameters
NUM_BANKS = 1
NUM_ROWS = 256
NUM_BLOCKS = 32
NUM_ROWS_PER_BLOCK = 8
assert NUM_ROWS % NUM_ROWS_PER_BLOCK == 0

WEIGHT_BITS = 4

# zw: not sure, but i think one IMC bank can handle 256 weights/MACs at a time
total_number_of_weights = NUM_BANKS\
      * NUM_BLOCKS * NUM_ROWS_PER_BLOCK

# generate main function to receive arguments from command line
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_weights", type=int, default=total_number_of_weights)
    parser.add_argument("--bit_size", type=int, default=WEIGHT_BITS)
    # parser.add_argument("--file_path", type=str, default=output_file_path)
    parser.add_argument("--mode", type=str, default="random")
    parser.add_argument("--const", type=int, default=0, help="constant value to be written to all weights if mode is const")
    args = parser.parse_args()

    if args.mode == "random":
        weights = generate_random_weights(args.num_weights, bit_size=args.bit_size)
    elif args.mode == "zeros":
        weights = ['0000'] * args.num_weights
    elif args.mode == "const":
        weights = [format(args.const& 0b1111, '04b')] * args.num_weights
        min_val = -2**(args.bit_size - 1)
        max_val = 2**(args.bit_size - 1) - 1    
        assert args.const >= min_val and args.const <= max_val
    else:
        raise NotImplementedError
    
    output_file_path = f'./{args.mode}_{total_number_of_weights}_weights.txt'
    write_weights_to_file(weights, output_file_path)
