def select_cols(w, m):
    n_cols = w.shape[1]
    return w[:, [*range(0, m), *range(n_cols-m, w.shape[1])]]


def flat(lists):
    return [item for sublist in lists for item in sublist]
