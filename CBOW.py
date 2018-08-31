import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
EMBEDDING_DIM = 15
raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

# By deriving a set from `raw_text`, we deduplicate the array
vocab = set(raw_text)
vocab_size = len(vocab)

word_to_ix = {word: i for i, word in enumerate(vocab)}
data = []
for i in range(2, len(raw_text) - 2):
    context = [raw_text[i - 2], raw_text[i - 1],
               raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))
print(data[:5])


class CBOW(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(CBOW, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size*embedding_dim*2, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embedding = self.embedding(inputs)
        embedding = embedding.view(1, -1)
        layer_1 = self.linear1(embedding)
        layer_1 = F.relu(layer_1)
        layer_2 = self.linear2(layer_1)
        layer_2 = F.relu(layer_2)
        out = self.linear3(layer_2)
        out = F.relu(out)
        log_probs = F.log_softmax(out, dim =1)
        return log_probs


# create your model and train.  here are some functions to help you make
# the data ready for use by your module

def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)

losses = []
loss_function = nn.NLLLoss()
model = CBOW(vocab_size, EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)


for epoch in range(10):
    total_loss = torch.Tensor([0])
    for context, target in data:
        context_vector = make_context_vector(context, word_to_ix)
        model.zero_grad()

        y_hat = model(context_vector)
        loss = loss_function(y_hat, autograd.Variable(torch.LongTensor([word_to_ix[target]])))
        loss.backward()
        optimizer.step()

        total_loss += loss.data
    losses.append(total_loss)

print(losses)

