from __future__ import annotations
import numpy as np, pandas as pd
from metal_predictor.core import PipelineConfig
from metal_predictor.data import SilverDatasetValidator
from metal_predictor.features import PriceActionFeatures,MomentumFeatures,VolatilityFeatures,TrendFeatures,TemporalFeatures,QualityFeatures

class SilverFeatureAssembler:
    def __init__(self,config:PipelineConfig|None=None)->None:
        self._config=config or PipelineConfig(); c=self._config.columns; f=self._config.features
        self._components=(PriceActionFeatures(c),MomentumFeatures(c,f),VolatilityFeatures(c,f),TrendFeatures(c,f),TemporalFeatures(c),QualityFeatures(c)); self._validator=SilverDatasetValidator(c)
        self.feature_names=tuple(n for component in self._components for n in component.feature_names)
    def transform(self,hourly:pd.DataFrame)->pd.DataFrame:
        c=self._config.columns; out=hourly.copy(deep=True); out[c.timestamp]=pd.to_datetime(out[c.timestamp],utc=True,errors="raise"); out=out.sort_values(c.timestamp).reset_index(drop=True)
        for n in (c.open,c.high,c.low,c.close): out[n]=pd.to_numeric(out[n],errors="raise").astype(float)
        self._validator.validate(out)
        for component in self._components: out=component.transform(out)
        if np.isinf(out.loc[:,self.feature_names].apply(pd.to_numeric,errors="coerce").to_numpy(float)).any(): raise ValueError("Feature graph produced infinite values.")
        return out
