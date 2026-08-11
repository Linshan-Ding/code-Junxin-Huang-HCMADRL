
def Triangular_fuzzy_time(add_time, rem_time):
    """确定化统计量"""
    add_l, add_m, add_u = add_time
    rem_l, rem_m, rem_u = rem_time
    total_l = add_l + rem_l
    total_m = add_m + rem_m
    total_u = add_u + rem_u
    t  = (total_l + 4*total_m + total_u) / 6
    return t

def delta_fuzzy_time(add_time, rem_time):

    add_l, add_m, add_u = add_time
    rem_l, rem_m, rem_u = rem_time
    total_l  = add_l + rem_l
    total_m = add_m + rem_m
    total_u = add_u + rem_u
    return total_u - total_l
