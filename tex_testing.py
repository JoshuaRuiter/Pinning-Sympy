# Testing for writing pinned group objects to a summary tex file

from pinned_group import pinned_group

n_min = 2
n_max = 4
q_min = 1
q_max = 2
overwrite = True

# Special linear groups
for n in range(n_min, n_max + 1):
    name_string = f"SL(n={n})"
    SL_n = pinned_group.load_from_file(name_string)
    SL_n.write_to_tex(overwrite)
    
# Split special orthogonal groups
for q in range(max(q_min, 2), q_max + 1):
    n_range = [n for n in (2*q, 2*q+1) if n_min <= n and n <= n_max]
    for n in n_range:
        name_string = f"SO(n={n}, q={q})"
        SO_n_q = pinned_group.load_from_file(name_string)
        SO_n_q.write_to_tex(overwrite)
        
# Non split special orthogonal groups
for q in range(q_min, q_max + 1):
    n_min = max(2*q+2, n_min) # only non-split if n >= 2q+2
    for n in range(n_min, n_max + 1):
        name_string = f"SO(n={n}, q={q})"
        SO_n_q = pinned_group.load_from_file(name_string)
        SO_n_q.write_to_tex(overwrite)
    