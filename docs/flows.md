# V1 and V2 Flow Diagrams

## V1 Flow

```mermaid
flowchart TD
    A["Junction API or sample payload"] --> B["Raw sleep payload"]
    B --> C["Normalize to user_id / provider / date feature table"]
    C --> D["Engineer reliability features"]
    D --> E["Heuristic confidence score"]
    E --> F{"Policy threshold"}
    F --> G["Show"]
    F --> H["Show with warning"]
    F --> I["Suppress"]
    E --> J["V1 report and plots"]
```

### V1 Summary

- Ingest Junction-style sleep data
- Build a daily feature table
- Compute completeness, baseline, stability, and provider-agreement features
- Combine them into a hand-tuned confidence score
- Map that score into `show`, `show_with_warning`, or `suppress`

## V2 Flow

```mermaid
flowchart TD
    A["Early Junction-style snapshot"] --> C["Early feature table"]
    B["Mature backfilled snapshot"] --> D["Mature feature table"]
    C --> E["Align rows on user_id / provider / date"]
    D --> E
    E --> F["Create labels: label_observed and safe_to_show"]
    F --> G["Train regularized logistic regression"]
    G --> H["Calibrate predicted probabilities"]
    H --> I["Score reliability: p(safe_to_show)"]
    I --> J{"Policy threshold"}
    J --> K["Show"]
    J --> L["Show with warning"]
    J --> M["Suppress"]
    F --> N["Model label observability"]
    N --> O["IPW / AIPW evaluation correction"]
    I --> O
    O --> P["V2 report, metrics, and comparison vs V1"]
```

### V2 Summary

- Compare an early snapshot to a later mature snapshot
- Label whether an early signal stayed close enough to the mature version
- Train a calibrated reliability model
- Convert predicted probabilities into a product policy
- Use IPW and AIPW to correct evaluation when only some rows become verifiable later

## Interview Framing

### V1

`V1` is a heuristic reliability policy:
- useful for demonstrating product reasoning
- highly interpretable
- not learned from an explicit target

### V2

`V2` is a learned reliability model:
- turns reliability into a supervised learning problem
- keeps the output product-facing
- uses causal methods only to debias evaluation, not to replace the predictor
