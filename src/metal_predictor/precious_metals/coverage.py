from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Protocol
import pandas as pd

class CoverageFold(Protocol):
    number:int; train:pd.DataFrame; validation:pd.DataFrame
class DevelopmentSplitter(Protocol):
    def split(self,frame:pd.DataFrame)->tuple[CoverageFold,...]: ...

@dataclass(frozen=True)
class PreciousMetalsCoveragePolicy:
    min_full_metal_coverage:float=.50
    min_train_metal_coverage:float=.40
    min_validation_metal_coverage:float=.60
    min_validation_joint_coverage:float=.50
    min_train_joint_rows:int=2000
    min_validation_joint_rows:int=500
    def __post_init__(self):
        ratios=(self.min_full_metal_coverage,self.min_train_metal_coverage,self.min_validation_metal_coverage,self.min_validation_joint_coverage)
        if any(not 0<v<=1 for v in ratios):raise ValueError("Coverage ratios must be in (0, 1].")
        if self.min_train_joint_rows<1 or self.min_validation_joint_rows<1:raise ValueError("Minimum joint-row counts must be positive.")

class PreciousMetalsCoverageValidator:
    _REQUIRED=("xpt_has_exact_current","xpd_has_exact_current","both_metals_have_exact_current")
    def __init__(self,policy=None):self._policy=policy or PreciousMetalsCoveragePolicy()
    def validate(self,development,splitter):
        missing=set(self._REQUIRED).difference(development.columns)
        if missing:raise ValueError(f"Precious-metals coverage gate missing columns: {sorted(missing)}")
        if development.empty:raise ValueError("Precious-metals coverage gate received empty development data.")
        full=self._coverage(development)
        if full["xpt"]<self._policy.min_full_metal_coverage:raise ValueError("XPT full-development exact coverage is below the pre-registered minimum.")
        if full["xpd"]<self._policy.min_full_metal_coverage:raise ValueError("XPD full-development exact coverage is below the pre-registered minimum.")
        reports=[]
        for fold in splitter.split(development):
            train=self._coverage(fold.train); val=self._coverage(fold.validation); tr=int(fold.train["both_metals_have_exact_current"].eq(1).sum()); vr=int(fold.validation["both_metals_have_exact_current"].eq(1).sum()); failures=[]
            for metal in ("xpt","xpd"):
                if train[metal]<self._policy.min_train_metal_coverage:failures.append(f"train_{metal}_coverage")
                if val[metal]<self._policy.min_validation_metal_coverage:failures.append(f"validation_{metal}_coverage")
            if val["joint"]<self._policy.min_validation_joint_coverage:failures.append("validation_joint_coverage")
            if tr<self._policy.min_train_joint_rows:failures.append("train_joint_rows")
            if vr<self._policy.min_validation_joint_rows:failures.append("validation_joint_rows")
            reports.append({"fold":fold.number,"passed":not failures,"failures":failures})
            if failures:raise ValueError(f"Precious-metals coverage gate failed fold {fold.number}: {failures}")
        return {"status":"PASS","policy_fixed_before_result":True,"policy":asdict(self._policy),"full_development_coverage":full,"folds":reports}
    @staticmethod
    def _coverage(frame):
        n=len(frame); return {"xpt":float(frame["xpt_has_exact_current"].eq(1).sum()/n),"xpd":float(frame["xpd_has_exact_current"].eq(1).sum()/n),"joint":float(frame["both_metals_have_exact_current"].eq(1).sum()/n)}
