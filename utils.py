import itertools

def generate_stim_combos(left, right):
    combo = []
    for l, r in itertools.product(left, right):
        combo.append(''.join(sorted([l, r]))) 
    return combo

def generate_group_stims(group):
    group_dic = {'1': ('abcd', 'efgh'), '2': ('abgh', 'cdef'), '3':('abef', 'cdgh')}
    return generate_stim_combos(group_dic[group][0], group_dic[group][1])