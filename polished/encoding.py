

def yeo_network():
    visual = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]
    somatomotor = [0, 1, 16, 17, 18, 19, 56, 57, 68, 69]
    dorsal_attention = [58, 59, 60, 61, 6, 7]
    ventral_attention = [28, 29, 62, 63]
    limbic = [36, 37, 38, 39, 40, 41, 24, 25, 20, 21]
    frontoparietal = [2, 3, 6, 7, 10, 11, 12, 13]
    default_mode = [22, 23, 32, 33, 34, 35, 64, 65, 66, 67, 84, 85]
    functional_groups = {
        'visual': visual,
        'somatomotor': somatomotor,
        'dorsal_attention': dorsal_attention,
        'ventral_attention': ventral_attention,
        'limbic': limbic,
        'frontoparietal': frontoparietal,
        'default_mode': default_mode,
        'other': []
    }
    all_assigned_nodes = (visual + somatomotor + dorsal_attention +
                          ventral_attention + limbic + frontoparietal + default_mode)
    for i in range(116):
        if i not in all_assigned_nodes:
            functional_groups['other'].append(i)
    return functional_groups