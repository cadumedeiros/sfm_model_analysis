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

        def _norm(name):
            return re.sub(r'[^A-Z0-9]+', '_', str(name).strip().upper()).strip('_')

        def _find_col(df, candidates):
            norm_to_real = {_norm(c): c for c in df.columns}
            for cand in candidates:
                real = norm_to_real.get(_norm(cand))
                if real is not None:
                    return real
            return None

        try:
            with open(path, 'r', encoding='latin-1') as f:
                for line in f:
                    ls = line.strip()
                    if not ls:
                        continue

                    up = ls.upper()

                    # NULL do header
                    if up.startswith("NULL"):
                        nums = re.findall(r"[-+]?\d*\.?\d+", ls)
                        if nums:
                            null_value = float(nums[-1])

                    # seção ~Curve
                    if up.startswith("~CURVE"):
                        in_curve = True
                        in_ascii = False
                        continue

                    # seção ~Ascii
                    if up.startswith("~ASCII") or up.startswith("~A"):
                        in_ascii = True
                        in_curve = False
                        continue

                    # saiu da seção
                    if ls.startswith("~") and not (up.startswith("~CURVE") or up.startswith("~ASCII") or up.startswith("~A")):
                        in_curve = False
                        in_ascii = False
                        continue

                    if in_curve and not ls.startswith("#"):
                        m = re.match(r"([^\.\s]+)", ls)
                        if m:
                            curve_names.append(m.group(1).strip())

                    if in_ascii and not ls.startswith("#"):
                        try:
                            data_rows.append([float(x) for x in ls.split()])
                        except Exception:
                            pass

            if not data_rows:
                return None

            ncols = len(data_rows[0])

            if len(curve_names) != ncols:
                curve_names = [f"COL_{i}" for i in range(ncols)]

            df = pd.DataFrame(data_rows, columns=curve_names)
            df.replace(null_value, np.nan, inplace=True)

            # garante numérico
            for c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

            # aliases úteis para o app, SEM perder o nome original do LAS
            aliases = {
                "DEPT": ["DEPT", "DEPTH", "MD"],
                "fac": ["FAC", "FACIES"],
                "lito_upscaled": ["LITO_UPSCALED"],
                "fac_dion": ["FAC_DION", "FACIES_DION"],
                "lito": ["LITO", "LITOLOGIA"],
                "bat": ["BAT", "BATIMETRIA"],
                "ntg": ["NTG"],
            }

            for alias, candidates in aliases.items():
                if alias in df.columns:
                    continue
                src = _find_col(df, candidates)
                if src is not None:
                    df[alias] = pd.to_numeric(df[src], errors="coerce")

            return df

        except Exception as e:
            print(f"Erro ao ler LAS {path}: {e}")
            return None
        
    def get_facies_column(self):
        if self.data is None or self.data.empty:
            return None

        preferred = (
            "fac",
            "FACIES",
            "FAC",
            "lito_upscaled",
            "LITO_UPSCALED",
            "fac_dion",
            "FAC_DION",
            "lito",
            "LITO",
        )

        for col in preferred:
            if col in self.data.columns:
                return col

        return None


    def get_log_columns(self, include_spatial=False):
        if self.data is None or self.data.empty:
            return []

        skip = set()
        if not include_spatial:
            skip.update(["X", "Y", "Z"])

        return [c for c in self.data.columns if c not in skip]

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
        if self.data is None or self.data.empty:
            return None

        points = self.data[["X", "Y", "Z"]].values.copy()
        points[:, 2] *= z_exag

        poly = pv.lines_from_points(points)

        fac_col = self.get_facies_column()
        if fac_col is not None:
            poly.point_data["Facies_Real"] = (
                pd.to_numeric(self.data[fac_col], errors="coerce")
                .fillna(0)
                .to_numpy(dtype=float)
            )

        if "DEPT" in self.data.columns:
            poly.point_data["MD"] = (
                pd.to_numeric(self.data["DEPT"], errors="coerce")
                .to_numpy(dtype=float)
            )

        # adiciona todas as curvas do LAS como point_data
        for col in self.get_log_columns(include_spatial=False):
            if col in ("DEPT", fac_col):
                continue

            arr = pd.to_numeric(self.data[col], errors="coerce")
            if arr.notna().any():
                poly.point_data[str(col)] = arr.to_numpy(dtype=float)

        return poly.tube(radius=100)

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
        return pv.PolyData(pts).glyph(geom=pv.Sphere(radius=150), scale=False), labels