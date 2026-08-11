from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

class FrozenRidgeRegressor:
    """Pure NumPy inference for a sealed frozen Ridge payload."""
    def __init__(self,payload:dict[str,object])->None:
        clean=dict(payload); claimed=str(clean.pop("model_payload_sha256"))
        actual=hashlib.sha256(json.dumps(clean,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode()).hexdigest()
        if claimed!=actual: raise ValueError("Frozen Ridge payload hash mismatch.")
        self.model_payload_sha256=claimed; self.model_name=str(payload["model_name"]); self.feature_names=tuple(map(str,payload["feature_names"]))
        self._median=np.asarray(payload["imputer_median"],float); self._indicator=np.asarray(payload["missing_indicator_feature_indices"],int)
        self._mean=np.asarray(payload["scaler_mean"],float); self._scale=np.asarray(payload["scaler_scale"],float); self._coef=np.asarray(payload["ridge_coefficients"],float); self._intercept=float(payload["ridge_intercept"])
        width=len(self.feature_names)+len(self._indicator)
        if len(self._median)!=len(self.feature_names) or not(len(self._mean)==len(self._scale)==len(self._coef)==width): raise ValueError("Frozen Ridge dimensions are inconsistent.")
        if (self._scale<=0).any() or not np.isfinite(self._scale).all(): raise ValueError("Frozen Ridge scaler is invalid.")
    @classmethod
    def from_path(cls,path:Path)->"FrozenRidgeRegressor": return cls(json.loads(path.read_text(encoding="utf-8")))
    def predict(self,frame:pd.DataFrame)->np.ndarray:
        missing=set(self.feature_names).difference(frame.columns)
        if missing: raise ValueError(f"Frozen Ridge prediction missing features: {sorted(missing)}")
        x=frame.loc[:,self.feature_names].apply(pd.to_numeric,errors="coerce").to_numpy(float); nan=np.isnan(x)
        if not (np.isfinite(x)|nan).all(): raise ValueError("Frozen Ridge features contain infinite values.")
        filled=np.where(nan,self._median.reshape(1,-1),x); transformed=np.column_stack([filled,nan[:,self._indicator].astype(float)]) if len(self._indicator) else filled
        pred=self._intercept+((transformed-self._mean)/self._scale)@self._coef
        if not np.isfinite(pred).all(): raise ValueError("Frozen Ridge produced non-finite predictions.")
        return pred.astype(float)
