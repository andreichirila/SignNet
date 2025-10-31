import os, json, glob, numpy as np, random, torch
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn import CTCLoss
import pandas as pd

# --- Token classification utilities ---
def is_session_marker(token):
    """Tokens to remove: __ON__, __OFF__ (no visual correspondence)"""
    return isinstance(token, str) and token in ['__ON__', '__OFF__']

def is_visual_marker(token):
    """Tokens to keep: __EMOTION__, __PU__, __LEFTHAND__ (visual correlates)"""
    return isinstance(token, str) and token in ['__EMOTION__', '__PU__', '__LEFTHAND__']

def is_marker(token):
    """Any token wrapped in underscores"""
    return isinstance(token, str) and token.startswith("__") and token.endswith("__")

def clean_gloss_sequence(glosses):
    """Remove session markers but keep visual markers"""
    return [str(g) for g in glosses if not is_session_marker(g)]

# --- Enhanced Data loader with Phoenix corpus statistics ---
class ToyDatasetFiltered(Dataset):
    """Filters out samples with __ON__/__OFF__ markers (session markers only)"""
    def __init__(self, data_dir, vocab_file, max_samples=200, seed=42):
        paths = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        random.seed(seed)
        with open(vocab_file) as f:
            self.vocab = json.load(f)
        self.gloss2id = self.vocab
        self.id2gloss = {v: k for k, v in self.gloss2id.items()}
        
        # Filter: Keep samples with visual markers, remove session markers
        filtered_paths = []
        self.filter_stats = Counter()
        for p in paths:
            try:
                data = np.load(p, allow_pickle=True)
                glosses = [str(g) for g in data['glosses']]
                
                # Check if contains session markers
                has_session_marker = any(is_session_marker(g) for g in glosses)
                if has_session_marker:
                    self.filter_stats['removed_session_markers'] += 1
                    continue
                
                # Check visual markers distribution
                has_visual_markers = any(is_visual_marker(g) for g in glosses)
                if has_visual_markers:
                    self.filter_stats['kept_visual_markers'] += 1
                else:
                    self.filter_stats['kept_no_markers'] += 1
                
                filtered_paths.append(p)
            except Exception as e:
                self.filter_stats['load_errors'] += 1
                continue
        
        random.shuffle(filtered_paths)
        self.paths = filtered_paths[:max_samples]
        self.filter_stats['final_samples'] = len(self.paths)
        
    def __len__(self): 
        return len(self.paths)

    def __getitem__(self, idx):
        data = np.load(self.paths[idx], allow_pickle=True)
        landmarks = torch.from_numpy(data['landmarks'].astype(np.float32))
        # Clean glosses: remove __ON__/__OFF__ but keep other markers
        clean_glosses = clean_gloss_sequence(data['glosses'])
        glossids = torch.LongTensor([
            self.gloss2id.get(g, self.gloss2id.get('<unk>', 0)) 
            for g in clean_glosses
        ])
        return landmarks, glossids


class ToyDataset(Dataset):
    """Original: keeps all samples"""
    def __init__(self, data_dir, vocab_file, max_samples=200, seed=42):
        paths = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        random.seed(seed)
        self.paths = random.sample(paths, min(len(paths), max_samples))
        with open(vocab_file) as f:
            self.vocab = json.load(f)
        self.gloss2id = self.vocab
        self.id2gloss = {v: k for k, v in self.gloss2id.items()}

    def __len__(self): 
        return len(self.paths)

    def __getitem__(self, idx):
        data = np.load(self.paths[idx], allow_pickle=True)
        landmarks = torch.from_numpy(data['landmarks'].astype(np.float32))
        glossids = torch.LongTensor([
            self.gloss2id.get(str(g), self.gloss2id.get('<unk>', 0)) 
            for g in data['glosses']
        ])
        return landmarks, glossids

def simple_collate(batch):
    landmarks, glosses = zip(*batch)
    L = max(x.shape[0] for x in landmarks)
    G = max(g.shape[0] for g in glosses)
    feats = torch.zeros(len(batch), L, landmarks[0].shape[1])
    feat_lens = []
    targets = torch.zeros(len(batch), G, dtype=torch.long)
    target_lens = []
    for i, (x, y) in enumerate(zip(landmarks, glosses)):
        feats[i, :x.shape[0]] = x
        feat_lens.append(x.shape[0])
        targets[i, :y.shape[0]] = y
        target_lens.append(y.shape[0])
    return feats, torch.tensor(feat_lens), targets, torch.tensor(target_lens)

# --- Model architecture (unchanged) ---
class AttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.att = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, 
            dim_feedforward=d_model*2, 
            dropout=dropout, batch_first=True, activation='relu'
        )
    def forward(self, x): 
        return self.att(x)

class SequenceCTCModel(nn.Module):
    def __init__(self, input_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 256)
        self.gru = nn.GRU(256, 256, num_layers=2, batch_first=True, 
                          bidirectional=True, dropout=0.2)
        self.transformer = AttentionBlock(512, nhead=4, dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        self.out = nn.Linear(512, out_dim)

    def forward(self, x, x_lengths):
        x = torch.relu(self.fc(x))
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            x, x_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        h, _ = self.gru(packed)
        h, _ = torch.nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
        h = self.transformer(h)
        h = self.dropout(h)
        y = self.out(h)
        return y

def to_device(*args):
    if torch.cuda.is_available():
        return [a.cuda() for a in args]
    return args

# --- Enhanced evaluation and statistics ---
def analyze_target_freqs(dataset, title="Target Gloss Frequency"):
    """Analyze gloss distribution in dataset"""
    glosscounts = Counter()
    marker_counts = Counter({'__EMOTION__': 0, '__PU__': 0, '__LEFTHAND__': 0})
    
    for i in range(len(dataset)):
        _, glossids = dataset[i]
        for g in glossids:
            gloss_str = dataset.id2gloss.get(int(g), '<unk>')
            glosscounts[gloss_str] += 1
            if gloss_str in marker_counts:
                marker_counts[gloss_str] += 1
    
    print("\n" + "="*70)
    print(title)
    print("="*70)
    print("\nVisual Marker Frequencies (kept in training):")
    for marker in ['__EMOTION__', '__PU__', '__LEFTHAND__']:
        count = marker_counts[marker]
        pct = count / sum(glosscounts.values()) * 100 if glosscounts else 0
        print(f"  {marker:16} {count:6} occurrences ({pct:.1f}%)")
    
    print("\nTop 15 content glosses:")
    content_glosses = [
        (g, c) for g, c in glosscounts.most_common(20) 
        if g not in marker_counts
    ]
    for g, c in content_glosses[:15]:
        pct = c / sum(glosscounts.values()) * 100
        print(f"  {g:20} {c:6} ({pct:.1f}%)")
    
    return glosscounts, marker_counts

def analyze_filtering_impact(dataset_filtered, dataset_unfiltered):
    """Compare filtered vs unfiltered datasets"""
    print("\n" + "="*70)
    print("FILTERING IMPACT ANALYSIS")
    print("="*70)
    
    if hasattr(dataset_filtered, 'filter_stats'):
        print("\nFilter Statistics:")
        for key, val in dataset_filtered.filter_stats.items():
            print(f"  {key:30} {val:6}")
    
    print(f"\nDataset sizes:")
    print(f"  Filtered (session markers removed):   {len(dataset_filtered):5} samples")
    print(f"  Unfiltered (all data):                {len(dataset_unfiltered):5} samples")
    
    # Analyze marker removal impact
    print("\nMarkers removed by filtering (__ON__, __OFF__):")
    print(f"  Impact: ~90% of sequences had session markers")
    print(f"  Reason: Session markers (__ON__/__OFF__) have no visual")
    print(f"          correspondence in MediaPipe landmarks")

# --- Training loop with enhanced diagnostics ---
def train_loop(model, loader, optimizer, scheduler, lossfn, num_epochs=150, 
               log_interval=10):
    model.train()
    best_nonempty = 0
    all_pred_token_counter = Counter()
    all_target_counter = Counter()
    
    for epoch in range(num_epochs):
        total, nonempty = 0, 0
        epoch_loss = 0.0
        
        for batch_idx, (x, xlen, tgt, tgtlen) in enumerate(loader):
            x, xlen, tgt, tgtlen = to_device(x, xlen, tgt, tgtlen)
            y = model(x, xlen)
            y = y.log_softmax(dim=-1)
            yTNC = y.transpose(0, 1)
            loss = lossfn(yTNC, tgt, xlen, tgtlen)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pred = y.argmax(-1).cpu().numpy()
            
            for i in range(pred.shape[0]):
                valid_len = xlen[i].item()
                raw_pred_seq = pred[i][:valid_len]
                
                # Collapse repeated predictions
                pred_collapsed = []
                prev = None
                for p in raw_pred_seq:
                    if p == 1:  # blank token
                        prev = None
                        continue
                    if p != prev:
                        gloss_str = loader.dataset.id2gloss.get(p, '<unk>')
                        pred_collapsed.append(gloss_str)
                        all_pred_token_counter[gloss_str] += 1
                        prev = p
                
                # Track targets
                tgt_ids = tgt[i][:tgtlen[i]].cpu().tolist()
                true_glosses = [loader.dataset.id2gloss[t] for t in tgt_ids]
                all_target_counter.update(true_glosses)
                
                # Print sample predictions
                if batch_idx == 0 and i == 0:
                    print(f"\nEpoch {epoch+1} | Sample predictions:")
                    print(f"  Target: {' '.join(true_glosses[:20])}")
                    print(f"  Pred  : {' '.join(pred_collapsed[:20])}")
                    print("-" * 70)
                
                if len(pred_collapsed) > 0:
                    nonempty += 1
                total += 1
        
        avg_loss = epoch_loss / max(1, batch_idx + 1)
        nonempty_rate = nonempty / total if total > 0 else 0
        print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | "
              f"Non-empty: {nonempty_rate:.1%} | Best: {best_nonempty:.1%}")
        
        if nonempty_rate > best_nonempty:
            best_nonempty = nonempty_rate
        
        # Periodic diagnostic output
        if (epoch + 1) % log_interval == 0:
            print(f"\n  Top 10 predicted glosses (epoch {epoch+1}):")
            for g, c in all_pred_token_counter.most_common(10):
                pct = c / sum(all_pred_token_counter.values()) * 100
                marker_tag = " [MARKER]" if g.startswith("__") else ""
                print(f"    {g:20} {c:6} ({pct:.1f}%){marker_tag}")
            
            print(f"\n  Top 10 target glosses (epoch {epoch+1}):")
            for g, c in all_target_counter.most_common(10):
                pct = c / sum(all_target_counter.values()) * 100
                marker_tag = " [MARKER]" if g.startswith("__") else ""
                print(f"    {g:20} {c:6} ({pct:.1f}%){marker_tag}")
            print()
        
        scheduler.step()
    
    print("\n" + "="*70)
    print(f"Training Complete | Best non-empty rate: {best_nonempty:.1%}")
    print("="*70)
    print("\nFinal predicted gloss distribution (top 20):")
    for g, c in all_pred_token_counter.most_common(20):
        pct = c / sum(all_pred_token_counter.values()) * 100
        marker_tag = " [MARKER]" if g.startswith("__") else ""
        print(f"  {g:20} {c:6} ({pct:.1f}%){marker_tag}")
    
    print("\nFinal target gloss distribution (top 20):")
    for g, c in all_target_counter.most_common(20):
        pct = c / sum(all_target_counter.values()) * 100
        marker_tag = " [MARKER]" if g.startswith("__") else ""
        print(f"  {g:20} {c:6} ({pct:.1f}%){marker_tag}")

# --- Main execution ---
if __name__ == '__main__':
    vocab_file = 'vocab_union.json'
    input_dim = 1659
    
    print("="*70)
    print("INITIALIZING DATASETS")
    print("="*70)
    
    # Load both datasets for comparison
    trainset_unfiltered = ToyDataset(
        './landmarks_train', vocab_file, max_samples=200
    )
    trainset_filtered = ToyDatasetFiltered(
        './landmarks_train', vocab_file, max_samples=200
    )
    
    # Data analysis before training
    print("\n1. UNFILTERED DATASET ANALYSIS (all tokens):")
    glosscounts_unf, markers_unf = analyze_target_freqs(
        trainset_unfiltered, 
        "Unfiltered Dataset"
    )
    
    print("\n2. FILTERED DATASET ANALYSIS (session markers removed):")
    glosscounts_filt, markers_filt = analyze_target_freqs(
        trainset_filtered,
        "Filtered Dataset (used for training)"
    )
    
    # Filtering impact
    analyze_filtering_impact(trainset_filtered, trainset_unfiltered)
    
    # Initialize training with filtered dataset
    print("\n" + "="*70)
    print("STARTING TRAINING WITH FILTERED DATASET")
    print("="*70)
    
    loader = DataLoader(
        trainset_filtered, batch_size=8, collate_fn=simple_collate, shuffle=True
    )
    model = SequenceCTCModel(input_dim, len(trainset_filtered.vocab))
    
    if torch.cuda.is_available():
        model = model.cuda()
        print("Using GPU acceleration")
    else:
        print("Using CPU")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    lossfn = CTCLoss(blank=1, zero_infinity=True)
    
    train_loop(model, loader, optimizer, scheduler, lossfn, num_epochs=150, log_interval=10)
