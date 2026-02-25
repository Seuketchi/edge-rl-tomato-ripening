# k1 Ripening Rate Constant — Calibration Notes

## Summary

The ripening rate constant `k1` in the ODE `dX/dt = -k1 * (T - T_base) * X` was recalibrated from **0.08 to 0.02 day^-1 °C^-1** based on empirical ripening time data from published literature. The original value produced unrealistically fast ripening (2-4 days at 20-25°C vs. 10-14 days in reality).

## ODE Model

```
X(t) = X0 * exp(-k1 * (T - T_base) * t)
```

- `X`: Continuous Chromatic Index [0, 1] (1 = Green, 0 = Red)
- `T`: Chamber temperature (°C)
- `T_base`: 12.5°C (minimum temperature for ripening)
- `t`: Time in days

## Empirical Data

Source: Ogundiwin et al. (2022), "Modeling the metachronous ripening pattern of mature green tomato as affected by cultivar and storage temperature," *Scientific Reports*, DOI: 10.1038/s41598-022-12219-z

| Storage Temp | Days to Fully Red | Derived k1 (day^-1 °C^-1) |
|---|---|---|
| 15°C | ~21 | 0.044 |
| 20°C | ~14 | 0.022 |
| 25°C | ~10 | 0.018 |
| 30°C | ~14* | N/A |

*At 30°C, lycopene biosynthesis is inhibited (Saltveit, 2005). The tomato softens but does not turn red, producing yellow-orange colour instead.

### Derivation

k1 is derived by inverting the analytical solution:

```
k1 = -ln(X_ripe / X0) / ((T - T_base) * t_ripe)
```

Using X0 = 1.0, X_ripe = 0.15:

- At 20°C: k1 = -ln(0.15) / (7.5 * 14) = 1.897 / 105 = 0.018
- At 25°C: k1 = -ln(0.15) / (12.5 * 10) = 1.897 / 125 = 0.015

The variation across temperatures (0.015-0.044) reflects the limitation of the linear (T - T_base) approximation vs. true Arrhenius kinetics. Within the practical operating range of 20-25°C, values converge around 0.02.

## T_base Justification

- **12.5°C (55°F)**: UC Davis Postharvest Technology Center recommendation for minimum ripening temperature
- **Saltveit (2005)**: Acceptable ripening range is 12.5-25°C
- **Below 10°C**: Chilling injury prevents normal ripening after 14+ days

No change from original value. Well-supported.

## Lycopene Inhibition Above 25°C

Saltveit (2005) documents that above ~25°C, the enzyme phytoene synthase (responsible for lycopene production) is inhibited. Effects:
- Softening and senescence continue
- Red colour (lycopene) does not develop
- Beta-carotene (yellow-orange) accumulates instead

### Decision: Not modeled in the ODE

Rationale:
- The deployment environment (Iligan City, indoor, no AC) has ambient 22-31°C
- The chamber regularly exceeds 25°C during daytime
- Modeling this would require separating colour and texture into two ODEs (overscoped)
- The Continuous Chromatic Index X is interpreted as "overall ripeness progression" rather than strictly red pigmentation
- This is documented as a limitation in the thesis

## Deployment Environment: Iligan City, Philippines

Climate data (annual):
- Mean temperature: ~27°C (29°C daytime, 22°C nighttime)
- Diurnal swing: ~8°C
- Relative humidity: 77-84%
- Minimal seasonal variation (tropical rainforest, Koppen Af)

Indoor, no AC conditions:
- Daytime: 28-31°C
- Nighttime: 22-25°C
- Average: ~27°C
- Std: ~3°C

## Ripening Projections at Iligan Conditions

With calibrated k1 = 0.02, T_base = 12.5, starting from X0 = 0.85:

| Condition | Temp (°C) | Days to X = 0.15 |
|---|---|---|
| Night only (22°C) | 22 | ~10.0 |
| Average ambient (27°C) | 27 | ~6.0 |
| Day only (31°C) | 31 | ~4.7 |
| Heater active (33°C) | 33 | ~4.2 |

The agent's controllable range is approximately **4-10 days**, which fits the target harvest window of 3-7 days for partially-ripe starting fruit.

## Parameter Changes

| Parameter | Old Value | New Value | File(s) Affected |
|---|---|---|---|
| `k1` | 0.08 | **0.02** | `config.yaml`, `simulator.py` |
| `k1_variation` | 0.02 | **0.008** | `config.yaml`, `simulator.py` |
| `ambient_temp_mean` | 27.0 | **27.0** (unchanged) | — |
| `ambient_temp_std` | 2.0 | **3.0** | `config.yaml`, `simulator.py` |
| `initial_temp_range` | [25, 30] | **[25, 31]** | `config.yaml`, `simulator.py` |
| `T_base` | 12.5 | **12.5** (unchanged) | — |

## References

1. Ogundiwin et al. (2022). "Modeling the metachronous ripening pattern of mature green tomato as affected by cultivar and storage temperature." *Scientific Reports*. DOI: 10.1038/s41598-022-12219-z
2. Saltveit, M. E. (2005). "Fruit ripening and fruit quality." In: Heuvelink, E. (ed.) *Tomatoes*, 2nd ed., CABI Publishing, pp. 145-170.
3. UC Davis Postharvest Technology Center. "Tomato: Recommendations for Maintaining Postharvest Quality." https://postharvest.ucdavis.edu/produce-facts-sheets/tomato
4. Magdalita, P. M. & Naredo, E. C. (2020). "Morphological and physicochemical characterization of 'Diamante Max' tomato." *Philippine Journal of Crop Science*, 45(1), 30-38.

## Impact on Retraining

The k1 change means the RL policy must be retrained. The reward function and architecture do not change — only the simulator dynamics are slower. Expected effects:
- Episodes will have more steps of meaningful decision-making (ripening takes longer)
- The agent should still learn the three-phase strategy (heat early, coast, cool late)
- Domain randomization now covers a biologically realistic range
- Distillation and edge deployment pipeline remain unchanged
