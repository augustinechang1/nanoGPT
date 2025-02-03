""" Each token in the input text is converted into a high-dimensional vector.
These vectors are then passed through multiple attention layers,
which allow the model to focus on different parts of the input sequence.
Following the attention layers, the vectors are processed by a multi-layer perceptron (MLP),
with normalization steps applied in between.
This process is repeated through many iterations,
during which each vector absorbs information from all other vectors in the sequence.
This enables the model to make predictions about the next token in the sequence.
The majority of the model's parameters are located in the multi-layer perceptron (approximately two-thirds),
while the remaining one-third of the parameters reside in the attention layers. """

import torch
print(torch.__version__)

with open('input.txt', 'r') as file:
    text = file.read()

# print(text[:1000])

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

# Create a mapping of unique characters to integers
# Check out tiktoken for larger code books
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
# encode the text
encode = lambda c: [char_to_int[char] for char in c]
# decode the text
decode = lambda c: ''.join([int_to_char[i] for i in c])
print(encode('augustine'))
print(decode(encode(('augustine'))))

# encode the text
data = torch.tensor(encode(text), dtype=torch.long)
# print(data.shape, data.dtype)
# print(data[:1000])

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 8
train_data[:block_size + 1]

x = train_data[:block_size]
y = train_data[1:block_size + 1]
for t in range(block_size):
    context = x[:t + 1]
    target = y[t]
    print(f"when input is {context} then target is {target}")

torch.manual_seed(1337)
batch_size = 4
block_size = 8

def get_batch(split):
    if split == 'train':
        data = train_data
    else:
        data = val_data
    start_idx = torch.randint(0, data.size(0) - block_size, (batch_size,))
    x = torch.stack([data[idx:idx + block_size] for idx in start_idx])
    y = torch.stack([data[idx + 1:idx + block_size + 1] for idx in start_idx])
    return x, y

# print the first batch
x, y = get_batch('train')
print('input:', x)
print(x.shape)
print('target:', y)
print(y.shape)

# print the first batch
for b in range(batch_size):
    for t in range(block_size):
        context = x[b, :t + 1]
        target = y[b, t]
        print(f"batch {b}, when input is {context} then target is {target}")

