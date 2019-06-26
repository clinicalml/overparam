import numpy as np

def match_cost_L1_prob(prior1, prior2, fail1, fail2):
    if prior1 < p_threshold or prior2 < p_threshold:
        return 1000.0
    return np.sum(np.abs(fail1 - fail2))

def match_cost_L1(prior1, prior2, fail1, fail2):
    if prior1 < p_threshold or prior2 < p_threshold:
        return 1000.0
    return np.sum(np.abs(np.log(fail1) - np.log(fail2)))

def match_cost_Linf_prob(prior1, prior2, fail1, fail2):
    if prior1 < p_threshold or prior2 < p_threshold:
        return 1000.0
    return np.amax(np.abs(fail1 - fail2)) # L-infinity distance

def match_cost_Linf(prior1, prior2, fail1, fail2):
    if prior1 < p_threshold or prior2 < p_threshold:
        return 1000.0
    return np.amax(np.abs(np.log(fail1) - np.log(fail2))) # L-infinity distance

p_threshold = 0.02
f_threshold = 0.80
match_threshold_filter = 4.0

def filter_parameters(p, f):
    L = p.shape[0]
    filtered_latent = [x for x in range(L) if (p[x] > p_threshold and np.amin(f[x]) < f_threshold)]

    p_filtered = p[filtered_latent]
    f_filtered = f[filtered_latent]

    # If duplicates, keep one with largest prior.
    p_sorted = np.flip(np.argsort(p_filtered), 0)
    new_filter = []
    for i in range(p_filtered.shape[0]):
        duplicate = False
        # Check if duplicate.
        for j in range(0, i):
            if match_cost_L1_prob(p_filtered[p_sorted[i]], p_filtered[p_sorted[j]], f_filtered[p_sorted[i]], f_filtered[p_sorted[j]]) <= match_threshold_filter:
                duplicate = True
                break
        if not duplicate:
            new_filter.append(p_sorted[i])

    return p_filtered[new_filter], f_filtered[new_filter]

def filter_parameters_indices(p, f):
    L = p.shape[0]
    filtered_latent = [(p[x] > p_threshold and np.amin(f[x]) < f_threshold) for x in range(L)]

    # If duplicates, keep one with largest prior.
    p_sorted = np.flip(np.argsort(p), 0)
    new_filter = []
    for i in range(p.shape[0]):
        if filtered_latent[i]:
            duplicate = False
            # Check if duplicate.
            for j in range(0, i):
                if filtered_latent[j] and match_cost_L1_prob(p[p_sorted[i]], p[p_sorted[j]], f[p_sorted[i]], f[p_sorted[j]]) <= match_threshold_filter:
                    duplicate = True
                    break
            if not duplicate:
                new_filter.append(p_sorted[i])

    return new_filter
