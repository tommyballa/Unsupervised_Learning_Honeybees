"""Active learning + semi-supervised per raggruppare crop di api in cluster."""

UNKNOWN_LABEL = "__unknown__"   # "Non so": tenuta da parte, mai nel kNN, mai riproposta
REMOVED_LABEL = "__removed__"   # evento di rimozione nel label store
