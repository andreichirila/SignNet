import glob, json, numpy as np
from collections import Counter

train_dir = "./landmarks_train"
dev_dir = "./landmarks_dev"
all_paths = glob.glob(train_dir + "/*.npz") + glob.glob(dev_dir + "/*.npz")
counter = Counter()

for path in all_paths:
    data = np.load(path, allow_pickle=True)
    counter.update(map(str, data['glosses']))

# Include anything with min frequency 1 for full coverage
min_freq = 1
vocab = {'<pad>': 0, '<blank>': 1, '<sos>': 2, '<eos>':3, '<unk>': 4}
for idx, gloss in enumerate(sorted([g for g, c in counter.items() if c >= min_freq])):
    vocab[gloss] = idx + 5
with open("vocab_union.json", "w") as f:
    json.dump(vocab, f, indent=2)
print(f"Vocab built: {len(vocab)} entries (min_freq={min_freq})")
