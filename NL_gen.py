import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import sys
# Load current (I) non-linearity
I_NL = pd.read_excel('./data_I-nonlinearity.xlsx').to_numpy()[:,1]
# NL_mode = True

# # Parameters
# A_vec_ch, A_vec_bit = 64, 4
# A_vec_max = 2**A_vec_bit-1
# W_vec_bit = 3   # Sign-Magnitude
# multibit_mask = 2**torch.arange(W_vec_bit-1, -1, -1)

# # Random A & W generation
# product=[]
# product_NL=[]
# vectorize = True
# lookup = lambda t: I_NL[int(t)]
# vfunc = np.vectorize(lookup)
# print(2**W_vec_bit)
# for i in tqdm(range(10000)):
#   A_vec = np.random.randint(0, A_vec_max, (A_vec_ch,))
#   W_vec = np.random.randint(-(2**W_vec_bit-1), 2**W_vec_bit, (A_vec_ch,))
#   # print(f'Dot-Product: {np.dot(A_vec, W_vec).item()}')
#   product.append(np.dot(A_vec, W_vec))
#   # Generate IMC-aware tensors
#   A_vec_IMC = np.zeros((A_vec_ch, A_vec_max))
#   for idx, val in enumerate(A_vec, start=0):
#     # print(f"id:{idx} val:{val}")
#     A_vec_IMC[idx] = np.concatenate((np.ones(val), np.zeros(A_vec_max-val)), 0)
#   # A_vec_IMC = A_vec_IMC.long()  # Datatype matching for dot-product
#   W_vec_IMCp = W_vec.clip(0,2**W_vec_bit)
#   W_vec_IMCn = W_vec.clip(-2**W_vec_bit,0)

#   # Compute MAC
#   # MACp, MACn = torch.Tensor([0]), torch.Tensor([0])
#   MACp_accumulator, MACn_accumulator = np.zeros((A_vec_max)), np.zeros((A_vec_max))
#   MACp, MACn = 0, 0
#   if vectorize:
#     MACp_ideal, MACn_ideal = np.dot(A_vec_IMC.T, W_vec_IMCp),  np.dot(A_vec_IMC.T, W_vec_IMCn)
#     MACp_NL = vfunc(MACp_ideal)
#     MACn_NL = -vfunc(np.abs(MACn_ideal))
#     MACp += np.sum(MACp_NL) 
#     MACn += np.sum(MACn_NL) 
#   else:
#     for idx in range(A_vec_max):
#       MACp_ideal, MACn_ideal = np.dot(A_vec_IMC[:,idx], W_vec_IMCp), np.dot(A_vec_IMC[:,idx], W_vec_IMCn)
#       MACp_NL, MACn_NL = I_NL[int(MACp_ideal)], -I_NL[int(np.abs(MACn_ideal))]
#       MACp += MACp_NL 
#       MACn += MACn_NL 

#   product_NL.append(numpy.array(MACp+MACn))


# # plt.plot(np.array(product))
# # plt.plot(np.array(product_NL))
# f = np.polyfit(np.array(product),np.array(product_NL),1)
# plt.plot(np.array(product), np.array(product_NL))
# plt.xlabel('product_ideal')
# plt.xlim(-1500,1500)
# plt.ylim(-1200,1200)
# plt.ylabel('product_NL')
# plt.grid()
# plt.title('relation')
# plt.show()
# plt.savefig('./save.png')


def mac_64(A_vec, W_vec, I_NL, NL_mode):

  if len(A_vec) < 64:
    A_vec = np.pad(A_vec,[0,64-len(A_vec)])
    W_vec = np.pad(W_vec,[0,64-len(W_vec)])

  A_vec = (A_vec * 2**4).astype(int)
  W_vec = (W_vec* 2**3).astype(int)
  # I_NL = pd.read_excel('data_I-nonlinearity.xlsx').to_numpy()[:, 1]

  # Parameters
  A_vec_ch, A_vec_bit = 64, 4
  A_vec_max = 2 ** A_vec_bit - 1
  W_vec_bit = 3  # Sign-Magnitude
  multibit_mask = 2 ** torch.arange(W_vec_bit - 1, -1, -1)

  # Random A & W generation
  # A_vec = np.random.randint(0, A_vec_max, (A_vec_ch,))
  # W_vec = np.random.randint(-(2 ** W_vec_bit - 1), 2 ** W_vec_bit - 1, (A_vec_ch,))
  # print(f'Dot-Product: {np.dot(A_vec, W_vec).item()}')
  # Product = np.dot(A_vec, W_vec)

  # Generate IMC-aware tensors
  A_vec_IMC = np.zeros((A_vec_ch, A_vec_max))
  for idx, val in enumerate(A_vec, start=0):
    A_vec_IMC[idx] = np.concatenate((np.ones(val), np.zeros(A_vec_max - val)), 0)
  # A_vec_IMC = A_vec_IMC.long()  # Datatype matching for dot-product
  W_vec_IMCp = W_vec.clip(0, 2 ** W_vec_bit)
  W_vec_IMCn = W_vec.clip(-2 ** W_vec_bit, 0)

  # Compute MAC
  # MACp, MACn = torch.Tensor([0]), torch.Tensor([0])
  lookup = lambda t: I_NL[int(t)]
  vfunc = np.vectorize(lookup)
  MACp, MACn = 0, 0
  for idx in range(A_vec_max):
    MACp_ideal, MACn_ideal = np.dot(A_vec_IMC[:, idx], W_vec_IMCp), np.dot(A_vec_IMC[:, idx], W_vec_IMCn)
    MACp_NL, MACn_NL = I_NL[int(MACp_ideal)], -I_NL[int(np.abs(MACn_ideal))]
    MACp += MACp_NL if NL_mode else MACp_ideal
    MACn += MACn_NL if NL_mode else MACn_ideal
  # MACp_ideal, MACn_ideal = np.dot(A_vec_IMC.T, W_vec_IMCp),  np.dot(A_vec_IMC.T, W_vec_IMCn)
  # MACp_NL = vfunc(MACp_ideal)
  # MACn_NL = -vfunc(np.abs(MACn_ideal))
  # MACp += np.sum(MACp_NL) 
  # MACn += np.sum(MACn_NL) 
  Product_NL = MACp + MACn
  # print(f'MAC CompSutation: {(MACp + MACn).item()}')
  Product_NL=Product_NL/2**7
  return Product_NL

v_dot = lambda x,y: np.dot(x,y)
v_dot_func = np.vectorize(v_dot, signature='(n,m),(m)->(n)')
lookup = lambda t: I_NL[int(t)]
vfunc = np.vectorize(lookup)

def mac_64_kernel(A_mat, W_mat, I_NL, NL_mode, acc_length):
  # A_mat shape:(channel, kernel_size*kernel_size), W_mat shape:(channel, kernel_size*kernel_size)
  ch = A_mat.shape[0]
  ks_square = A_mat.shape[1]
  if ch < acc_length: # if number of channels is smaller than accumulation length, pad
    A_mat = np.pad(A_mat,pad_width = ((0,acc_length-A_mat.shape[0]), (0,0)), mode = "constant", constant_values=0)
    W_mat = np.pad(W_mat,pad_width = ((0,acc_length-W_mat.shape[0]), (0,0)), mode = "constant", constant_values=0)
    # print(f"shape of A :{A_mat.shape} shape of W:{W_mat.shape}")
  else:
    raise NotImplementedError
  A_mat = (A_mat * 2**4).astype(int)
  W_mat = (W_mat * 2**3).astype(int)
  # I_NL = pd.read_excel('data_I-nonlinearity.xlsx').to_numpy()[:, 1]

  # Parameters
  A_mat_ch, A_mat_bit = ch, 4
  A_mat_max = 2 ** A_mat_bit - 1
  W_mat_bit = 3  # Sign-Magnitude
  multibit_mask = 2 ** np.arange(W_mat_bit - 1, -1, -1)

  # Random A & W generation
  # A_vec = np.random.randint(0, A_vec_max, (A_vec_ch,))
  # W_vec = np.random.randint(-(2 ** W_vec_bit - 1), 2 ** W_vec_bit - 1, (A_vec_ch,))
  # print(f'Dot-Product: {np.dot(A_vec, W_vec).item()}')
  # Product = np.dot(A_vec, W_vec)

  # Generate IMC-aware tensors
  A_mat_IMC = np.zeros((acc_length, ks_square, A_mat_max))

  # for i in range(A_mat.shape[0]):
  #   for j, val in enumerate(A_mat[i], start=0):
  #     A_mat_IMC[i,j] = np.concatenate((np.ones(val), np.zeros(A_mat_max - val)), 0)
  v_serial = lambda val: np.concatenate((np.ones(int(val)), np.zeros(A_mat_max - int(val))), 0)
  A_mat_IMC = np.array(list(v_serial(A_mat_element) for A_mat_element in A_mat.reshape(-1))).reshape((acc_length, ks_square, A_mat_max))

  # A_vec_IMC = A_vec_IMC.long()  # Datatype matching for dot-product
  W_mat_IMCp = W_mat.clip(0, 2 ** W_mat_bit)
  W_mat_IMCn = W_mat.clip(-2 ** W_mat_bit, 0)

  # Compute MAC
  # MACp, MACn = torch.Tensor([0]), torch.Tensor([0])

  MACp, MACn = 0, 0

  A_mat_IMC = np.swapaxes(np.swapaxes(A_mat_IMC, 0,1), 1,2).astype(int)
  W_mat_IMCp = np.swapaxes(W_mat_IMCp,0,1)
  W_mat_IMCn = np.swapaxes(W_mat_IMCn,0,1)

  MACp_ideal = v_dot_func(A_mat_IMC, W_mat_IMCp)
  MACn_ideal = v_dot_func(A_mat_IMC, W_mat_IMCn)
  MACp_NL = vfunc(MACp_ideal)
  MACn_NL = -vfunc(np.abs(MACn_ideal))

  if NL_mode:
    MACp += np.sum(MACp_NL, axis=1) 
    MACn += np.sum(MACn_NL, axis=1)
  else:
    MACp += np.sum(MACp_ideal, axis=1) 
    MACn += np.sum(MACn_ideal, axis=1)    
  Product_NL = MACp + MACn

  Product_NL=Product_NL/2**7
  return Product_NL  # shape is kernel_size * kernel_size


def mac_64_multi_kernel(A_mat, W_mat, I_NL, NL_mode, acc_length):
  # A_mat shape:(channel, kernel_size*kernel_size), W_mat shape:(kernel_numb, channel, kernel_size*kernel_size)
  ch = A_mat.shape[0]
  ks_square = A_mat.shape[1]
  kn = W_mat.shape[0]
  if ch < acc_length: # if number of channels is smaller than accumulation length, pad
    A_mat = np.pad(A_mat,pad_width = ((0,acc_length-A_mat.shape[0]), (0,0)), mode = "constant", constant_values=0)
    W_mat = np.pad(W_mat,pad_width = ((0,0), (0,acc_length-W_mat.shape[0]), (0,0)), mode = "constant", constant_values=0)
    # print(f"shape of A :{A_mat.shape} shape of W:{W_mat.shape}")
  else:
    raise NotImplementedError
  A_mat = (A_mat * 2**4).astype(int)
  W_mat = (W_mat * 2**3).astype(int)
  # I_NL = pd.read_excel('data_I-nonlinearity.xlsx').to_numpy()[:, 1]

  # Parameters
  A_mat_ch, A_mat_bit = ch, 4
  A_mat_max = 2 ** A_mat_bit - 1
  W_mat_bit = 3  # Sign-Magnitude
  multibit_mask = 2 ** np.arange(W_mat_bit - 1, -1, -1)

  # Random A & W generation
  # A_vec = np.random.randint(0, A_vec_max, (A_vec_ch,))
  # W_vec = np.random.randint(-(2 ** W_vec_bit - 1), 2 ** W_vec_bit - 1, (A_vec_ch,))
  # print(f'Dot-Product: {np.dot(A_vec, W_vec).item()}')
  # Product = np.dot(A_vec, W_vec)

  # Generate IMC-aware tensors
  A_mat_IMC = np.zeros((acc_length, ks_square, A_mat_max))

  # for i in range(A_mat.shape[0]):
  #   for j, val in enumerate(A_mat[i], start=0):
  #     A_mat_IMC[i,j] = np.concatenate((np.ones(val), np.zeros(A_mat_max - val)), 0)
  v_serial = lambda val: np.concatenate((np.ones(int(val)), np.zeros(A_mat_max - int(val))), 0)
  A_mat_IMC = np.array(list(v_serial(A_mat_element) for A_mat_element in A_mat.reshape(-1))).reshape((acc_length, ks_square, A_mat_max))

  # A_vec_IMC = A_vec_IMC.long()  # Datatype matching for dot-product
  W_mat_IMCp = W_mat.clip(0, 2 ** W_mat_bit)
  W_mat_IMCn = W_mat.clip(-2 ** W_mat_bit, 0)

  # Compute MAC
  # MACp, MACn = torch.Tensor([0]), torch.Tensor([0])

  MACp, MACn = 0, 0

  A_mat_IMC = np.swapaxes(np.swapaxes(A_mat_IMC, 0,1), 1,2).astype(int)
  W_mat_IMCp = np.swapaxes(W_mat_IMCp,1,2)
  W_mat_IMCn = np.swapaxes(W_mat_IMCn,1,2)
  # print(f"W_mat_IMCp shape:{W_mat_IMCp.shape}")

  MACp_ideal = np.array([v_dot_func(A_mat_IMC, W_mat_IMCp_slice) for W_mat_IMCp_slice in W_mat_IMCp])
  MACn_ideal = np.array([v_dot_func(A_mat_IMC, W_mat_IMCn_slice) for W_mat_IMCn_slice in W_mat_IMCn])
  MACp_NL = vfunc(MACp_ideal)
  MACn_NL = -vfunc(np.abs(MACn_ideal))

  if NL_mode:
    MACp += np.sum(MACp_NL, axis=2) 
    MACn += np.sum(MACn_NL, axis=2)
  else:
    MACp += np.sum(MACp_ideal, axis=2) 
    MACn += np.sum(MACn_ideal, axis=2)    
  Product_NL = MACp + MACn

  Product_NL=Product_NL/2**7
  return Product_NL  # shape is kernel_number, kernel_size * kernel_size
