# wells.py
import numpy as np
import pandas as pd
import pyvista as pv

class Well:
    def __init__(self, name, dev_path, las_path):
        self.name = name
        self.dev_path = dev_path
        self.las_path = las_path
        self.trajectory = None
        self.logs = None
        self.data = None
        self.load_data()

    def load_data(self):
        self.trajectory = self._parse_dev(self.dev_path)
        self.logs = self._parse_las(self.las_path)
        
        if self.trajectory is not None and self.logs is not None:
            if not self.trajectory.empty and not self.logs.empty:
                self.data = self._merge_spatial_and_logs()

    def _parse_dev(self, path):
        try:
            rows = []
            cols = ["MD", "X", "Y", "Z", "TVD", "DX", "DY", "AZIM", "INCL", "DLS"]

            with open(path, "r", encoding="latin-1") as f:
                for line in f:
                    ls = line.strip()
                    if not ls:
                        continue
                    if ls.startswith("#"):
                        continue

                    # ignora cabeçalho tipo: MD X Y Z ...
                    upper = ls.upper()
                    if upper.startswith("MD " ) or upper == "MD":
                        continue

                    # normaliza decimal com vírgula
                    ls = ls.replace(",", ".")

                    parts = ls.split()
                    if len(parts) < 10:
                        continue

                    rows.append(parts[:10])

            if not rows:
                return None

            df = pd.DataFrame(rows, columns=cols)
            df = df.apply(pd.to_numeric, errors="coerce")

            # exige pelo menos a geometria principal
            df.dropna(subset=["MD", "X", "Y", "Z"], inplace=True)

            if df.empty:
                return None

            return df.sort_values("MD").reset_index(drop=True)

        except Exception as e:
            print(f"Erro ao ler DEV {path}: {e}")
            return None

    def _parse_las(self, path):
        import re
        data_rows = []
        curve_names = []
        in_curve = False
        in_ascii = False
        null_value = -9999.99

        try:
            with open(path, 'r', encoding='latin-1') as f:
                for line in f:
                    ls = line.strip()
                    if not ls:
                        continue

                    up = ls.upper()

                    # captura NULL do header
                    if up.startswith("NULL"):
                        nums = re.findall(r"[-+]?\d*\.?\d+", ls)
                        if nums:
                            null_value = float(nums[-1])

                    # início da seção de curvas
                    if up.startswith("~CURVE"):
                        in_curve = True
                        in_ascii = False
                        continue

                    # início da seção ASCII
                    if up.startswith("~ASCII") or up.startswith("~A"):
                        in_ascii = True
                        in_curve = False
                        continue

                    # saiu da seção atual
                    if ls.startswith("~") and not (up.startswith("~CURVE") or up.startswith("~ASCII") or up.startswith("~A")):
                        in_curve = False
                        in_ascii = False
                        continue

                    if in_curve and not ls.startswith("#"):
                        # pega o nome antes do ponto, ex: DEPT .m
                        m = re.match(r"([^\.\s]+)", ls)
                        if m:
                            curve_names.append(m.group(1).strip())

                    if in_ascii and not ls.startswith("#"):
                        try:
                            data_rows.append([float(x) for x in ls.split()])
                        except:
                            pass

            if not data_rows:
                return None

            ncols = len(data_rows[0])

            # fallback se por algum motivo não conseguiu ler ~CURVE
            if len(curve_names) != ncols:
                curve_names = [f"COL_{i}" for i in range(ncols)]

            df = pd.DataFrame(data_rows, columns=curve_names)
            df.replace(null_value, np.nan, inplace=True)

            # padronizações úteis
            rename_map = {}
            for c in df.columns:
                cu = c.strip().upper()
                if cu == "DEPT":
                    rename_map[c] = "DEPT"
                elif cu == "FAC":
                    rename_map[c] = "fac"
                elif cu == "BAT":
                    rename_map[c] = "bat"
                elif cu == "LITO_UPSCALED":
                    rename_map[c] = "lito_upscaled"
                elif cu == "FAC_DION":
                    rename_map[c] = "fac_dion"

            df.rename(columns=rename_map, inplace=True)
            return df

        except Exception as e:
            print(f"Erro ao ler LAS {path}: {e}")
            return None

    def _merge_spatial_and_logs(self):
        traj = self.trajectory.sort_values("MD")
        logs = self.logs.sort_values("DEPT")

        if logs.empty or traj.empty:
            return None

        merged = logs.copy()

        md = merged["DEPT"].to_numpy(dtype=float)

        # Interpola trajetória SEM cortar o LAS
        merged["X"] = np.interp(
            md, traj["MD"], traj["X"],
            left=traj["X"].iloc[0],
            right=traj["X"].iloc[-1]
        )
        merged["Y"] = np.interp(
            md, traj["MD"], traj["Y"],
            left=traj["Y"].iloc[0],
            right=traj["Y"].iloc[-1]
        )
        merged["Z"] = np.interp(
            md, traj["MD"], traj["Z"],
            left=traj["Z"].iloc[0],
            right=traj["Z"].iloc[-1]
        )

        return merged


    def get_vtk_polydata(self, z_exag=1.0):
        if self.data is None: return None
        # Coordenadas cruas + Exagero Z
        points = self.data[["X", "Y", "Z"]].values.copy()
        points[:, 2] *= z_exag
        
        poly = pv.lines_from_points(points)
        # Tenta pegar lito_upscaled, senao fac
        col = "lito_upscaled" if "lito_upscaled" in self.data.columns else "fac"
        if col in self.data.columns:
            poly.point_data["Facies_Real"] = self.data[col].fillna(0).values
        
        poly.point_data["MD"] = self.data["DEPT"].values
        return poly.tube(radius=30) # Tubo grosso pra ver a cor

    def get_markers_mesh(self, markers_list, z_exag=1.0):
        if self.data is None or not markers_list: return None, None
        pts = []
        labels = []
        
        # Interpolação simples
        traj = self.data.sort_values("DEPT")
        mds, xs, ys, zs = traj["DEPT"].values, traj["X"].values, traj["Y"].values, traj["Z"].values
        
        for m in markers_list:
            md_t = m['md']
            if md_t >= mds.min() and md_t <= mds.max():
                x = np.interp(md_t, mds, xs)
                y = np.interp(md_t, mds, ys)
                z = np.interp(md_t, mds, zs)
                pts.append([x, y, z * z_exag])
                labels.append(m['name'])
        
        if not pts: return None, None
        return pv.PolyData(pts).glyph(geom=pv.Sphere(radius=60), scale=False), labels