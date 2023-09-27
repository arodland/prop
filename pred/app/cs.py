def cs_to_stdev(cs, adj100=False):
    if cs == -1:
        cs = 55
    if cs == 100 and adj100:
        cs = 85
    return 0.185 - 0.00173 * cs

def stdev_to_cs(sd):
    cs = (0.185 - sd) / 0.00173

    if cs < 0:
        cs = 0
    if cs > 100:
        cs = 100

    return round(cs)

cs_coeffs = {
    'old':  {'int': 0.185, 'sl': 0.935135, 'replace_-1': 55, 'replace_100': 85},
    'fof2': {'int': 0.21315160999708233, 'sl': 0.9920850108292543, 'replace_-1': 75, 'replace_100': 75},
    'hmf2': {'int': 0.24213093225134258, 'sl': 0.9459521822390862, 'replace_-1': 75, 'replace_100': 75},
    'mufd': {'int': 0.22235171520011915, 'sl': 0.9536767599362671, 'replace_-1': 75, 'replace_100': 75},
}

def cs_to_stdev_new(cs, metric, adj100=True):
    coeff = cs_coeffs[metric]
    if cs == -1:
        cs = coeff['replace_-1']
    if cs == 100 and adj100:
        cs = coeff['replace_100']
    return coeff['int'] * (1. - coeff['sl']*cs/100)


