"""Part C helpers that do not download AraVec or score the official test."""

import numpy as np
import torch

from evaluate import build_part_c_rows
from models.deep import (
    AdditiveAttention,
    BiLSTMAttention,
    BiLSTMClassifier,
    CharCNN,
    MiniTransformer,
    count_trainable,
    embedding_from_matrix,
    sinusoidal_position_encoding,
)
from models.static import (
    PAD_ID,
    build_embedding_matrix,
    build_vocab,
    clean_str,
    oov_stats,
    tokenize,
)
from train.deep import TrainConfig, class_weights, early_stop_update, fit_model, oversample_minority


class _FakeVectors:
    dim = 2

    def __init__(self):
        self.table = {"ا": np.array([1.0, 2.0], dtype=np.float32)}

    def __contains__(self, key):
        return key in self.table

    def __getitem__(self, key):
        return self.table[key]


def test_vocab_keeps_frequent_types_and_reports_oov():
    vocab = build_vocab(["aa aa", "bb"], min_freq=2)

    assert vocab.itos[PAD_ID] == "<pad>"
    assert "aa" in vocab.stoi
    assert "bb" not in vocab.stoi
    stats = oov_stats(["aa zz"], vocab)
    assert stats["n_tokens"] == 2
    assert stats["n_oov"] == 1
    assert stats["oov_rate"] == 0.5


def test_equal_counts_are_sorted_by_token():
    vocab = build_vocab(["bb aa", "bb aa"], min_freq=2)
    assert vocab.itos[2:] == ["aa", "bb"]


def test_tokenize_keeps_cues_and_clean_str_is_lookup_only():
    assert tokenize("أ") == ["أ"]
    assert tokenize('"مرحبا"') == ["مرحبا"]
    assert tokenize("مرحبا!!!") == ["مرحبا", "!!!"]
    assert tokenize("هههه") == ["هههه"]
    assert tokenize("بَبت") == ["بَبت"]
    assert clean_str("أ") == "ا"
    assert clean_str("إ") == "ا"
    assert clean_str("آ") == "ا"
    assert clean_str("ة") == "ه"
    assert clean_str("ى") == "ي"
    assert clean_str("بَ") == "ب"
    assert clean_str("ههههه") == "هه"
    assert clean_str("كتاااب") == "كتاب"


def test_emoji_run_is_one_token():
    assert tokenize("مرحبا 😂😂") == ["مرحبا", "😂😂"]


def test_aravec_lookup_uses_clean_str():
    vocab = build_vocab(["أ أ", "ب ب"], min_freq=2)
    matrix, coverage = build_embedding_matrix(vocab, _FakeVectors(), dim=2, seed=0)

    assert np.allclose(matrix[PAD_ID], 0.0)
    assert np.allclose(matrix[vocab.stoi["أ"]], [1.0, 2.0])
    assert coverage["vocab_hits"] == 1
    assert not np.allclose(matrix[vocab.stoi["ب"]], [1.0, 2.0])


def test_attention_ignores_padding():
    torch.manual_seed(0)
    layer = AdditiveAttention(8)
    hidden = torch.randn(2, 4, 8)
    mask = torch.tensor([[True, True, False, False], [True, False, False, False]])

    _context, alpha = layer(hidden, mask)

    assert torch.allclose(alpha.sum(dim=-1), torch.ones(2), atol=1e-5)
    assert torch.all(alpha[0, 2:] == 0)
    assert torch.all(alpha[1, 1:] == 0)
    assert torch.allclose(alpha[1, 0], torch.tensor(1.0), atol=1e-5)


def test_sinusoidal_encoding_shape_and_origin():
    encoding = sinusoidal_position_encoding(10, 8)

    assert encoding.shape == (10, 8)
    assert torch.allclose(encoding[0, 0::2], torch.zeros(4))
    assert torch.allclose(encoding[0, 1::2], torch.ones(4))


def test_transformer_padding_does_not_change_logits():
    torch.manual_seed(0)
    model = MiniTransformer(12, 8, d_model=8, nhead=2, num_layers=2, dropout=0.0, max_len=8)
    model.eval()
    masked = torch.tensor([[True, True, False, False]])
    padded = torch.tensor([[2, 3, 0, 0]])
    other_pad = torch.tensor([[2, 3, 7, 4]])
    short = torch.tensor([[2, 3]])
    short_mask = torch.tensor([[True, True]])

    logits = model(padded, masked)
    assert torch.allclose(logits, model(other_pad, masked), atol=1e-5)
    assert torch.allclose(logits, model(short, short_mask), atol=1e-5)


def test_positional_encoding_can_be_removed():
    with_positions = MiniTransformer(12, 8, d_model=8, nhead=2, num_layers=2, use_positional=True)
    without_positions = MiniTransformer(12, 8, d_model=8, nhead=2, num_layers=2, use_positional=False)

    assert with_positions.positional is not None
    assert without_positions.positional is None


def test_max_pool_ignores_padding():
    torch.manual_seed(0)
    model = BiLSTMClassifier(12, 8, 4, dropout=0.0, pool="max")
    model.eval()
    mask = torch.tensor([[True, True, False, False]])
    padded = model(torch.tensor([[2, 3, 0, 0]]), mask)
    changed_pad = model(torch.tensor([[2, 3, 7, 4]]), mask)
    short = model(torch.tensor([[2, 3]]), torch.tensor([[True, True]]))

    assert torch.allclose(padded, changed_pad, atol=1e-5)
    assert torch.allclose(padded, short, atol=1e-5)


def test_last_state_ignores_padding():
    torch.manual_seed(0)
    model = BiLSTMClassifier(12, 8, 4, dropout=0.0, pool="last")
    model.eval()
    mask = torch.tensor([[True, True, False, False]])
    padded = model(torch.tensor([[2, 3, 0, 0]]), mask)
    changed_pad = model(torch.tensor([[2, 3, 7, 4]]), mask)
    short = model(torch.tensor([[2, 3]]), torch.tensor([[True, True]]))

    assert torch.allclose(padded, changed_pad, atol=1e-5)
    assert torch.allclose(padded, short, atol=1e-5)


def test_bilstm_attention_backward():
    torch.manual_seed(0)
    model = BiLSTMAttention(12, 8, 4, dropout=0.0)
    logits, alpha = model(
        torch.tensor([[2, 3, 0]]),
        torch.tensor([[True, True, False]]),
        return_attention=True,
    )
    logits.sum().backward()

    assert logits.shape == (1, 2)
    assert torch.allclose(alpha.sum(dim=-1), torch.ones(1), atol=1e-5)
    assert alpha[0, 2].detach().item() == 0.0


def test_char_cnn_logits_shape():
    model = CharCNN(10, embed_dim=8, channels=4, kernels=(3,), dropout=0.0)
    logits = model(torch.tensor([[2, 3, 0, 0]]), torch.tensor([[True, True, False, False]]))
    assert logits.shape == (1, 2)


def test_frozen_embedding_is_not_trainable():
    matrix = np.ones((6, 4), dtype=np.float32)
    matrix[0] = 0.0
    embedding = embedding_from_matrix(matrix, freeze=True)
    model = BiLSTMClassifier(6, 4, 4, embedding=embedding)

    assert model.embedding.weight.requires_grad is False
    assert count_trainable(model) < sum(parameter.numel() for parameter in model.parameters())


def test_early_stopping_resets_on_improvement():
    best, wait, improved, stop = early_stop_update(0.4, -1.0, 0, 2)
    assert improved and not stop and best == 0.4 and wait == 0
    best, wait, improved, stop = early_stop_update(0.3, best, wait, 2)
    assert not improved and not stop and wait == 1
    best, wait, improved, stop = early_stop_update(0.3, best, wait, 2)
    assert stop and wait == 2 and best == 0.4
    best, wait, improved, stop = early_stop_update(0.5, 0.4, 2, 2)
    assert improved and wait == 0 and not stop


def test_oversample_duplicates_sarcasm_inside_train_only():
    texts, labels = oversample_minority(["pos", "neg-a", "neg-b"], [1, 0, 0], seed=1)

    assert int((labels == 1).sum()) == 2
    assert int((labels == 0).sum()) == 2
    assert texts.count("pos") == 2
    assert texts.count("neg-a") == 1
    assert texts.count("neg-b") == 1


def test_class_weights_upweight_the_rare_class():
    weights = class_weights([1, 0, 0, 0])
    assert float(weights[1]) > float(weights[0])


def test_part_c_rows_leave_the_test_empty():
    labels = np.array([True, False])
    runs = [
        {
            "model": "c1_bilstm_random",
            "preprocess": "word_min2",
            "pred": np.array([True, False]),
            "proba": np.array([0.7, 0.2]),
            "threshold": 0.5,
            "train_time_s": 1.0,
            "n_trainable_params": 10,
            "notes": "test non utilisé",
        }
    ]

    rows = build_part_c_rows(labels, runs, seed=42)

    assert rows[0]["model"] == "c1_bilstm_random"
    assert rows[0]["test_f1_sarcasm"] == ""
    assert rows[0]["test_pr_auc"] == ""
    assert rows[0]["test_macro_f1"] == ""
    assert rows[0]["val_f1_sarcasm"] != ""


def test_fit_returns_validation_scores_without_a_test_split():
    config = TrainConfig(
        max_epochs=2,
        patience=1,
        batch_size=2,
        hidden=4,
        embed_dim=8,
        dropout=0.0,
        max_len=8,
        lr=1e-2,
        grad_clip=1.0,
    )
    texts = ["sarcasme haha", "sarcasme haha !!", "normal normal", "normal jour"]
    labels = [1, 1, 0, 0]
    vocab = build_vocab(texts, min_freq=1)
    model = BiLSTMClassifier(len(vocab), config.embed_dim, config.hidden, dropout=0.0)

    result = fit_model(model, texts, labels, texts, labels, vocab, config)

    assert result.proba.shape == (4,)
    assert result.pred.shape == (4,)
    assert result.n_trainable_params > 0
    assert 1 <= result.best_epoch <= 2
    assert result.threshold == 0.5
