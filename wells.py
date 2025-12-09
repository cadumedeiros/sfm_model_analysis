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
            # Lê coordenadas puras, sem inventar moda
            df = pd.read_csv(
                path, sep=r'\s+', comment='#', header=None,
                names=["MD", "X", "Y", "Z", "TVD", "DX", "DY", "AZIM", "INCL", "DLS"],
                engine='python'
            )
            df = df.apply(pd.to_numeric, errors='coerce')
            df.dropna(inplace=True)
            return df
        except: return None

    def _parse_las(self, path):
        data_rows = []
        in_ascii = False
        try:
            with open(path, 'r', encoding='latin-1') as f:
                for line in f:
                    ls = line.strip()
                    if not ls: continue
                    if ls.startswith("~Ascii") or ls.startswith("~A"):
                        in_ascii = True; continue
                    if in_ascii and not ls.startswith("#"):
                        try: data_rows.append([float(x) for x in ls.split()])
                        except: pass
            
            if not data_rows: return None
            df = pd.DataFrame(data_rows, columns=["DEPT", "fac", "bat", "lito_upscaled"])
            df.replace(-999.25, np.nan, inplace=True)
            return df
        except: return None

    def _merge_spatial_and_logs(self):
        traj = self.trajectory.sort_values("MD")
        logs = self.logs.sort_values("DEPT")
        min_md, max_md = traj["MD"].min(), traj["MD"].max()
        merged = logs[(logs["DEPT"] >= min_md) & (logs["DEPT"] <= max_md)].copy()
        if merged.empty: return None

        merged["X"] = np.interp(merged["DEPT"], traj["MD"], traj["X"])
        merged["Y"] = np.interp(merged["DEPT"], traj["MD"], traj["Y"])
        merged["Z"] = np.interp(merged["DEPT"], traj["MD"], traj["Z"])
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
            poly.point_data["Facies_Real"] = self.data[col].fillna(-1).values
        
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