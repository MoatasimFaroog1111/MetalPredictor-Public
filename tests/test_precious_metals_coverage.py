from dataclasses import dataclass
import pandas as pd
import pytest
from metal_predictor.precious_metals.coverage import PreciousMetalsCoveragePolicy,PreciousMetalsCoverageValidator

@dataclass
class Fold:
    number:int; train:pd.DataFrame; validation:pd.DataFrame
class FakeSplitter:
    def split(self,frame):
        return (Fold(1,frame.iloc[:300].copy(),frame.iloc[300:450].copy()),Fold(2,frame.iloc[:400].copy(),frame.iloc[450:].copy()))

def frame(sparse_after=None):
    n=600; xpt=pd.Series(1,index=range(n),dtype="int8"); xpd=xpt.copy()
    if sparse_after is not None:xpt.iloc[sparse_after:]=0;xpd.iloc[sparse_after:]=0
    return pd.DataFrame({"xpt_has_exact_current":xpt,"xpd_has_exact_current":xpd,"both_metals_have_exact_current":(xpt.eq(1)&xpd.eq(1)).astype("int8")})
def policy():
    return PreciousMetalsCoveragePolicy(min_full_metal_coverage=.5,min_train_metal_coverage=.4,min_validation_metal_coverage=.6,min_validation_joint_coverage=.5,min_train_joint_rows=100,min_validation_joint_rows=50)

def test_coverage_gate_passes_supported_folds():
    report=PreciousMetalsCoverageValidator(policy()).validate(frame(),FakeSplitter());assert report["status"]=="PASS";assert report["policy_fixed_before_result"] is True

def test_coverage_gate_fails_sparse_validation():
    with pytest.raises(ValueError,match="coverage gate failed fold"):
        PreciousMetalsCoverageValidator(policy()).validate(frame(400),FakeSplitter())
