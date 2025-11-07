# config.py (ou onde você quiser)

import os

def load_facies_colors(path=None):
    """
    Lê o arquivo color_reference_facies.txt e devolve um dict
    {facie_int: (r, g, b, a)} com floats.
    Ignora o cabeçalho "Facie R G B A".
    """
    if path is None:
        # tenta achar o arquivo na mesma pasta deste .py
        path = os.path.join(os.path.dirname(__file__), "color_reference_facies.txt")

    colors = {}
    with open(path, "r") as f:
        lines = f.readlines()

    # pula a primeira linha (cabeçalho)
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        # garante que a primeira coluna é número
        if not parts[0].isdigit():
            continue

        facie = int(parts[0])
        r = float(parts[1])
        g = float(parts[2])
        b = float(parts[3])
        a = float(parts[4])
        colors[facie] = (r, g, b, a)

    return colors
