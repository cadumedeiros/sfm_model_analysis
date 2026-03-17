# config.py

import os

# Configurações de Geometria e Transformação
ANCHOR_Y = None

# Se True, aplica a reflexão Y (2*Anchor - Y) em todos os dados (Grid e Poços)
APPLY_REFLECTION = True

def load_facies_colors(path=None):
    """
    Lê o arquivo color_reference_facies.txt e devolve um dict
    {facie_id: (r, g, b, a)} com floats.

    Observação:
    - `facie_id` será `int` quando possível (ex.: "23" -> 23)
      e `str` quando o identificador não for numérico (ex.: "A", "AB").
    - Linhas de comentário (começando com '#') são ignoradas.
    - A linha de cabeçalho "Facie R G B A" (ou variações) é ignorada.
    """
    ref = load_facies_reference(path)
    return {facie_id: rgba for facie_id, rgba in ref}


def load_facies_reference(path=None):
    """Carrega a referência completa de fácies e cores.

    Retorna uma lista ordenada (na ordem do arquivo):
        [(facie_id, (r,g,b,a)), ...]

    `facie_id` é `int` quando possível, senão `str`.
    """
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "assets/color_reference_facies_mixed_novo.txt")

    ref = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.replace("\t", " ").split()
            if len(parts) < 5:
                continue

            if parts[0].lower() in ("facie", "facies", "facies_id", "faciesid"):
                continue

            facie_tok = parts[0]
            try:
                facie_id = int(facie_tok)
            except ValueError:
                facie_id = facie_tok

            try:
                r, g, b, a = map(float, parts[1:5])
            except ValueError:
                continue

            ref.append((facie_id, (r, g, b, a)))

    return ref


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