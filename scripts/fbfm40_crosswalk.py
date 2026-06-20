# FBFM40 → Anderson 13 (1..13) ou não-queimável (91..99)
FBFM40_TO_ANDERSON = {
    # Não-queimáveis
    91: 91, 92: 92, 93: 93, 98: 98, 99: 99,
    # GR — grama (101-109): grama curta → 1; grama alta → 3
    101: 1, 102: 1, 103: 3, 104: 3, 105: 3, 106: 3, 107: 3, 108: 3, 109: 3,
    # GS — grama-arbusto (121-124) → 2 (timber-grass-shrub) / 1
    121: 1, 122: 2, 123: 2, 124: 2,
    # SH — arbusto (141-149) → 5/6 (brush); chamise denso → 4
    141: 5, 142: 5, 143: 6, 144: 6, 145: 6, 146: 6, 147: 6, 148: 6, 149: 4,
    # TU — timber-understory (161-165) → 8/10
    161: 8, 162: 8, 163: 8, 164: 10, 165: 10,
    # TL — timber-litter (181-189) → 8/9
    181: 8, 182: 8, 183: 8, 184: 9, 185: 9, 186: 9, 187: 8, 188: 9, 189: 9,
    # SB — slash-blowdown (201-204) → 11/12/13
    201: 11, 202: 11, 203: 12, 204: 13,
}


def fbfm40_to_anderson(arr):
    """Aplica o crosswalk a um array numpy de códigos FBFM40. Códigos
    desconhecidos viram 99 (não-queimável)."""
    import numpy as np
    out = np.full(arr.shape, 99, dtype=np.float32)
    for code, anderson in FBFM40_TO_ANDERSON.items():
        out[arr.astype(int) == code] = anderson
    return out


if __name__ == "__main__":
    import numpy as np
    sample = np.array([[101, 143, 183, 201], [91, 122, 165, 99]])
    print("FBFM40:\n", sample)
    print("→ Anderson:\n", fbfm40_to_anderson(sample).astype(int))
