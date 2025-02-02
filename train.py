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

import torch
print(torch.__version__)