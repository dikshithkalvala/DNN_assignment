# src/data.py
import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import math

# Simple tokenizer & Vocab
class Vocab:
    PAD = "<pad>"
    UNK = "<unk>"
    BOS = "<bos>"
    EOS = "<eos>"

    def __init__(self, min_freq: int = 1):
        self.min_freq = min_freq
        self.freqs = {}
        self.itos = []
        self.stoi = {}

    def add_sentence(self, sent: str):
        for tok in sent.strip().split():
            self.freqs[tok] = self.freqs.get(tok, 0) + 1

    def build(self):
        tokens = [t for t, f in self.freqs.items() if f >= self.min_freq]
        self.itos = [self.PAD, self.UNK, self.BOS, self.EOS] + sorted(tokens)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def encode(self, sent: str, max_len: int):
        toks = sent.strip().split()
        toks = [self.BOS] + toks[: (max_len - 2)] + [self.EOS]
        ids = [self.stoi.get(t, self.stoi[self.UNK]) for t in toks]
        if len(ids) < max_len:
            ids += [self.stoi[self.PAD]] * (max_len - len(ids))
        return ids

    def __len__(self):
        return len(self.itos)


def parse_xml_text(xml_path: str) -> str:
    """Extract meaningful text from XML. Fallback to joining tag names and text."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        texts = []
        # gather text content and attribute-like tags
        for elem in root.iter():
            if elem.text and elem.text.strip():
                texts.append(elem.text.strip())
            # include tag names that might be object names
            if elem.tag and elem.tag not in ("annotation", "root"):
                # skip generic root tags
                pass
        if len(texts) > 0:
            return " ".join(texts).replace("\n", " ").strip()
    except Exception:
        pass
    # fallback: filename derived caption
    return os.path.splitext(os.path.basename(xml_path))[0]


class StoryDataset(Dataset):
    """
    Constructs sequences (K -> K+1) from image+xml pairs.
    Assumes files are named in sorted order, e.g., image0001_.jpg, image0001_.xml
    """

    def __init__(
        self,
        root_dir: str,
        seq_len: int = 3,
        max_text_len: int = 20,
        split: str = "train",
        splits: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        transform=None,
        vocab: Vocab = None,
        build_vocab: bool = False,
    ):
        """
        root_dir: folder containing image*.jpg and image*.xml
        seq_len: number of past pairs to use as input
        split: 'train'|'val'|'test'
        vocab: if provided, used for encoding; if not provided and build_vocab True, builds vocab
        """
        assert split in ("train", "val", "test")
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.max_text_len = max_text_len
        self.transform = transform or T.Compose([T.Resize((224, 224)), T.ToTensor()])
        # discover pairs
        files = sorted(os.listdir(root_dir))
        images = [f for f in files if f.lower().endswith(".jpg") or f.lower().endswith(".png")]
        # assume xml names match image basename with .xml
        image_paths = [os.path.join(root_dir, f) for f in images]
        # create list of (img_path, xml_path)
        pairs = []
        for img in image_paths:
            base = os.path.splitext(os.path.basename(img))[0]
            # try several xml variants
            xml_candidates = [
                os.path.join(root_dir, base + ".xml"),
                os.path.join(root_dir, base.replace("_", "") + ".xml"),
            ]
            xml = None
            for c in xml_candidates:
                if os.path.exists(c):
                    xml = c
                    break
            if xml is None:
                # try same index-based xml
                xml_alt = img.replace(".jpg", ".xml").replace(".png", ".xml")
                if os.path.exists(xml_alt):
                    xml = xml_alt
            if xml:
                pairs.append((img, xml))
        pairs = sorted(pairs, key=lambda x: x[0])  # order by filename
        self.pairs = pairs

        n = len(pairs)
        t0 = 0
        t1 = int(math.floor(splits[0] * n))
        t2 = t1 + int(math.floor(splits[1] * n))
        if split == "train":
            sub = pairs[t0:t1]
        elif split == "val":
            sub = pairs[t1:t2]
        else:
            sub = pairs[t2:]
        # build sequences: need seq_len + 1 for target
        self.samples = []
        for i in range(len(sub) - seq_len):
            inp = sub[i : i + seq_len]  # list of (img, xml)
            target = sub[i + seq_len]  # (img, xml)
            self.samples.append((inp, target))

        # Vocab handling
        self.vocab = vocab or Vocab(min_freq=1)
        if build_vocab:
            for (_, xml) in pairs:
                txt = parse_xml_text(xml)
                self.vocab.add_sentence(txt)
            self.vocab.build()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inp, target = self.samples[idx]
        # process images and texts
        images = []
        texts = []
        for img_path, xml_path in inp:
            im = Image.open(img_path).convert("RGB")
            im = self.transform(im)
            images.append(im)
            txt = parse_xml_text(xml_path)
            texts.append(txt)
        # stack images: (seq_len, C, H, W)
        images = torch.stack(images, dim=0)
        # encode texts
        text_ids = [torch.tensor(self.vocab.encode(t, self.max_text_len), dtype=torch.long) for t in texts]
        text_ids = torch.stack(text_ids, dim=0)  # (seq_len, max_text_len)
        # target
        tgt_img = Image.open(target[0]).convert("RGB")
        tgt_img = self.transform(tgt_img)
        tgt_txt = parse_xml_text(target[1])
        tgt_txt_ids = torch.tensor(self.vocab.encode(tgt_txt, self.max_text_len), dtype=torch.long)
        return {
            "images": images,  # seq_len x C x H x W
            "texts": text_ids,  # seq_len x max_text_len
            "target_image": tgt_img,
            "target_text": tgt_txt_ids,
            "raw_target_text": tgt_txt,
        }


def make_dataloaders(root_dir: str, seq_len=3, batch_size=16, max_text_len=20, build_vocab=True):
    # Build vocab on training split
    train_ds = StoryDataset(root_dir, seq_len=seq_len, max_text_len=max_text_len, split="train", build_vocab=build_vocab)
    val_ds = StoryDataset(root_dir, seq_len=seq_len, max_text_len=max_text_len, split="val", vocab=train_ds.vocab)
    test_ds = StoryDataset(root_dir, seq_len=seq_len, max_text_len=max_text_len, split="test", vocab=train_ds.vocab)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=1)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=1)

    return train_loader, val_loader, test_loader, train_ds.vocab
