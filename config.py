# config.py

import os

# Configurações de Geometria e Transformação
ANCHOR_Y = 8923962.0 

# Se True, aplica a reflexão Y (2*Anchor - Y) em todos os dados (Grid e Poços)
APPLY_REFLECTION = True

def load_facies_colors(path=None):
    """
    Lê o arquivo color_reference_facies.txt e devolve um dict
    {facie_int: (r, g, b, a)} com floats.
    Ignora o cabeçalho "Facie R G B A".
    """
    if path is None:
        # tenta achar o arquivo na mesma pasta deste .py
        path = os.path.join(os.path.dirname(__file__), "assets/color_reference_facies.txt")

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

def load_markers(path):
    """
    Lê o arquivo wellMarkers.txt
    Retorna: Dict { 'NomePoço': [ {'nome': 'Base', 'md': 972.41}, ... ] }
    """
    markers_db = {}
    try:
        with open(path, 'r', encoding='latin-1') as f:
            lines = [l.strip() for l in f if l.strip()]
        
        # Estrutura baseada no seu arquivo:
        # Linha 0: Markers
        # Linha 1: 2 (Count?)
        # Linha 2: Base_Datum (Nome Marker 1)
        # Linha 3: Base (Nome Marker 2)
        # Linha 4: Well 5 (Header seção poços)
        # Linha 5+: 100 855.94 972.41
        
        # Identificando nomes dos markers
        # Assumindo que linhas 2 e 3 são os nomes. 
        # Para ser robusto, vamos pegar tudo entre a linha de count e a linha "Well"
        marker_names = []
        start_data_idx = 0
        
        for i, line in enumerate(lines):
            if line.startswith("Well"):
                start_data_idx = i + 1
                break
            if i > 1: # Pula "Markers" e o Count
                marker_names.append(line)
                
        # Lendo os dados
        for i in range(start_data_idx, len(lines)):
            parts = lines[i].split()
            if not parts: continue
            
            well_name = parts[0]
            mds = parts[1:]
            
            well_markers = []
            for m_idx, md_val in enumerate(mds):
                if m_idx < len(marker_names):
                    try:
                        md_float = float(md_val)
                        well_markers.append({
                            "name": marker_names[m_idx],
                            "md": md_float
                        })
                    except: pass
            
            markers_db[well_name] = well_markers
            
        return markers_db
            
    except Exception as e:
        print(f"Erro ao ler markers: {e}")
        return {}