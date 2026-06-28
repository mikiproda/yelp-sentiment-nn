"""nm_sentiment — comparative deep-learning sentiment analysis on Yelp Review Full.

Four architectures (LSTM, Bi-LSTM, BERT, RoBERTa) trained and evaluated on an
identical, persisted stratified split so the comparison is fair. See
IMPLEMENTATION_PLAN.md for the full design.
"""

from .config import DataConfig, RNNConfig, HFConfig
from .seeding import set_seed

__all__ = ["DataConfig", "RNNConfig", "HFConfig", "set_seed"]
