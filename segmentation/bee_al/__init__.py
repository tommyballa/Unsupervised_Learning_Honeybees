"""Active learning + semi-supervised learning to group bee crops into clusters."""

UNKNOWN_LABEL = "__unknown__"   # "Discard bee" button: set aside, never in the kNN, never proposed again
REMOVED_LABEL = "__removed__"   # removal event in the label store
