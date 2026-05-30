# Executive Summary

We surveyed traffic‐safety and ITS literature to identify widely‐used metrics for risk exposure, traffic externalities, shockwaves, courtesy/disruption, gap acceptance, and efficiency.  Classic safety measures (Time‐To‐Collision, Time Headway, Encroachment/Conflict scores, etc.) are backed by decades of research【18†L143-L150】【54†L133-L139】.  Traffic flow and externality metrics include shockwave detection (e.g. shockwave speed and speed‐variance spikes【57†L149-L157】), flow variability, and aggregate indices (e.g. “Efficiency Index” EI and combined Safety–Efficiency Index SEI【53†L58-L61】).  Courtesy/disruption metrics include follower deceleration (e.g. DRAC)【30†L1732-L1736】, yield/gap rates, and spacing changes, often used in cooperative-driving studies.  We map each literature metric to our features (Table 1), identifying direct correspondences (e.g. TTC ↔ “rear\_ttc\_delta”), near-analogs (e.g. headway ↔ “follower\_thw\_delta”), and gaps (few fields use a “social benefit score”).

We recommend a concise metric set by level: **System-level:** overall flow, average speed, shockwave count/speed, and a combined safety–efficiency index (e.g. Andreotti’s SEI【53†L58-L61】); **Agent-level:** Time-To-Collision (TTC) and Time Headway (THW) distributions (with typical thresholds \~1–3 s【18†L143-L150】【54†L133-L139】), maximum required deceleration (DRAC or Δv), accepted/lag gap size, and yield‐event rates; **Field/PDE-level:** total **risk mass** (integral of risk field), mean risk per vehicle, backward risk flux (captures shockwaves), and risk-gradient peaks. Each metric is precisely defined (with units and typical target values) and literature‐cited in Table 2.  We propose keeping novel risk‐field metrics (renamed for clarity: e.g. *Total Risk Potential* for risk\_mass\_total, *Backward Risk Flux* for risk\_flux\_backward) and focusing on key one-shot features (merge *social\_benefit\_score* into efficiency/risk gains).  Visualization should include heatmaps of risk fields, time-series of total/mean risk per agent, Pareto plots of efficiency vs. risk, and a flowchart of data-to-metrics.  Statistical validation should use standard significance tests (t-tests/ANOVA with 95% CIs【30†L1578-L1582】), normalize metrics by traffic density or agent count, and test robustness via bootstrapping or multiple runs.  Overall, our choices align with ITS best practices and ensure rigorous, interpretable evaluation of social‐friendly behavior.

## Literature‐Backed Metrics by Category

* **Risk Exposure (Safety):** Established *surrogate safety measures* include **Time‐To‐Collision (TTC)** and variants (e.g. Time Exposed TTC)【18†L143-L150】, **Time Headway (THW)**【54†L133-L139】, **Post-Encroachment Time (PET)**, and **Deceleration-to-Safety (DRAC)**.  TTC is widely used (target \~1–1.5 s as “critical”【18†L143-L150】) and is even embedded in commercial tools【56†L1-L4】.  THW (recommended ≥1.8–3 s depending on country) quantifies spacing【54†L133-L139】.  Delta-v (closing speed), conflict indices (e.g. *Brake Threat Number*), and collision probability measures also appear in ITS safety literature【18†L143-L150】【54†L133-L139】.  These provide a basis for our agent-level “time headway” and “TTC reduction” features.
* **Traffic Externalities (Flow/Disturbance):** Metrics quantify how one vehicle affects others and global flow.  Common measures include **shockwave characteristics** (formation count, propagation speed)【57†L149-L157】【14†L75-L82】, aggregate variability (speed or flow standard deviation), and macroscopic indices (throughput, density).  For example, Elfar et al. showed that traffic **speed variance (SSD)** sharply spikes during shockwaves【57†L149-L157】.  Fundamental diagrams (flow vs. density) and capacity‐drop phenomena are also classic【57†L149-L157】.  Recent work (e.g. Andreotti et al.) combines speed and gap consistency into an **Efficiency Index (EI)**, and further a **Safety–Efficiency Index (SEI)** that penalizes low TTC events【53†L58-L61】.  These concepts guide our field‐level metrics (total risk flux analogous to shockwave flux) and system metrics (flow, EI/SEI).
* **Shockwave / Backward Propagation:**  Theory predicts backward shock speeds of \~2–15 mph【14†L75-L82】.  Empirical methods detect shockwaves via space-time speed drops.  Elfar et al. found shockwaves coincide with surges in SSD【57†L149-L157】.  Other works fit regression to identify shockwave velocity【14†L75-L82】.  In practice, one can use: (a) **maximum backward gradient of flow or speed**, (b) **count of backward waves per time**, or (c) **mean shockwave speed**.  These map to our *risk\_flux\_backward*.
* **Courtesy / Disruption:**  Social-driving literature defines courtesy as how much a vehicle yields or eases others.  Wang et al. and colleagues quantified *Courtesy Reward* in AV planning, and observed human drivers often yield even when not obliged【8†L43-L52】【29†L79-L88】.  Metrics include **yield/gap given rates** (fraction of cut-ins accommodated), **reduction in follower risk**, and **imposed deceleration** on others.  For example, Jiang et al. (2023) measured courtesy by the follower’s Deceleration‐to‐Avoid‐Collision (DRAC)【30†L1732-L1736】 – lower DRAC means more courteous merging.  Zhang et al. (2024) measured courtesy by the *minimum spacing* allowed to neighbors【35†L513-L520】.  We thus consider: yield event flags, follower’s peak deceleration, gap changes, and even a *Responsibility-Sensitive Safety* score (from RSS)【33†L39-L47】.
* **Gap Acceptance:**  Classic gap acceptance theory (Philips, 1969) and empirical studies define **accepted gap** distributions for merges.  Surrogate gap metrics include *accepted lag time* and *critical gap*.  For example, accepted gap often \~3–4 s on highways.  We capture this via features like “gap\_offered” and “gap\_taken” (gap in front of merging vehicle).  Headway-based measures (THW) and PET are also relevant here.
* **Efficiency (Mobility):**  Traffic efficiency metrics include **average speed**, **throughput (veh/hr)**, **travel time/delay**, and **fuel/energy** consumption.  In mixed AV/HDV studies, one also uses throughput normalized by density.  For agent-level, one can use *Time or Distance Travelled* by ego.  Andreotti et al.’s EI combines local speed uniformity with spacing regularity【53†L58-L61】.  We track *ego average speed*, *total distance traveled*, and the proposed *Social Benefit Score* (analogous to net efficiency gain).

## Mapping Literature Metrics to Our Features

Table 1 matches literature metrics to our existing/proposed features, indicating analogies or gaps.

|**Category**|**Literature Metric**|**Our Feature (Current/Proposed)**|**Comment (Match / Analog / Gap)**|
|-|-|-|-|
|**Risk Exposure**|Time-To-Collision (TTC)【18†L143-L150】|`rear\_ttc\_delta`|TTC drop during lane-change; we compute ΔTTC, same units (s).|
||Time Headway (THW)【54†L133-L139】|`follower\_thw\_delta`|Change in follower headway due to ego. Our THW vs target \~1.8–3s.|
||**DRAC (Decel to Avoid Collision)**【30†L1732-L1736】|`rear\_decel\_peak\_3s`|Max follower braking (m/s²). Exactly DRAC concept for 3s window.|
||PET, Post-Encroachment Time|*none defined*|Could add future, but not directly used.|
|**Traffic Externalities**|Flow (veh/hr), Density|– (derived global)|Use average flow and occupancy from dataset for system metrics.|
||Speed StdDev / Variance【57†L149-L157】|*not in code*|Could compute global speed variance as instability measure.|
||Shockwave Speed / Count【14†L75-L82】【57†L149-L157】|`risk\_flux\_backward` (analog)|Backward-propagating risk flux proxies shockwave.|
||**Efficiency Index (EI)**【53†L58-L61】|*see `social\_benefit\_score`*|Social benefit approximates combined efficiency; consider renaming.|
|**Shockwaves**|Shockwave Detection (space-time drops)|`risk\_flux\_backward`|See above. Also could use e.g. count of backward peaks in speed.|
|**Courtesy**|Yield/Give-way rate|`yield\_given\_flag`|Direct analog: flag if ego yields, count as metric (unitless fraction).|
||**DRAC** (on merging **target** vehicle)【30†L1732-L1736】|*no direct feature*|We track follower decel (which is target lane follower’s response).|
||Min Distance to Others【35†L513-L520】|`min\_gap\_in\_front` or spacing|Not currently a feature; could compute min clearance.|
||Social Value Orientation (altruism)|*social\_benefit\_score* (proxy)|We proposed “social\_benefit\_score” as net efficiency gain; literature lacks standard.|
|**Gap Acceptance**|Accepted Gap Size (s)【20†L105-L113】|`gap\_offered`|Equivalent: time-gap in front of merging vehicle.|
||Lag Gap (headway to right before merge)|*gap\_left* (if defined)|In code, `gap\_offered` and/or `gap\_taken` cover this.|
|**Efficiency**|Average Speed (m/s or km/h)|`ego\_speed\_avg`|Our “progress” or average speed of ego. Standard unit (m/s or km/h).|
||Throughput (veh/hr), Travel Time (s)|*aggregate from data*|Compute externally: total vehicles passed or simulation time.|
||**Safety–Efficiency Index (SEI)**【53†L58-L61】|*combine risk \& progress*|Not explicitly computed, but we propose similar “risk-adjusted progress”.|

*Table 1. Mapping between literature‐defined metrics and our dataset features. Matches show where our current features capture classic measures; “gap” indicates area for new features.*

## Recommended Metrics (with Definitions)

We propose a **concise core metric set** in three categories:

* ### System-Level Metrics (Network/Aggregate)

  * **Average Traffic Speed (m/s or km/h):** Mean vehicle speed over time and space; high values indicate efficiency【53†L58-L61】.
  * **Throughput / Flow (vehicles/hour):** Number of vehicles passing a point per hour; tied to capacity.
  * **Shockwave Frequency / Propagation:** Count of backward wave events per time, or average backward wave speed (m/s); derived from changes in average speed【57†L149-L157】.
  * **Traffic Variability (Speed StdDev):** Standard deviation of speeds on the road; spikes during congestion【57†L149-L157】.
  * **Safety–Efficiency Index (SEI):** Combined metric (unitless) = α\*(normalized avg speed or spacing) − β\*(fraction of low-TTC events)【53†L58-L61】.  (For example, Andreotti et al.’s SEI merges local spacing and TTC【53†L58-L61】.)
* ### Agent/Tactical-Level Metrics

  * **Time-To-Collision (TTC, s):** Time until collision if current speeds held; use critical min(TTC) per maneuver【18†L143-L150】.  Thresholds: \~1–1.5 s considered imminent risk【18†L143-L150】.
  * **Time Headway (THW, s):** Time gap to lead vehicle: THW = gap/distance/relative speed【54†L133-L139】.  We record *∆THW* (change due to maneuver).  Regulatory targets \~1.8–3 s【54†L133-L139】.
  * **Minimum Distance (m):** Closest approach distance between ego and any neighbor during maneuver. Lower = more risk.
  * **Follower Deceleration (DRAC, m/s²):** Deceleration required by a following vehicle to avoid collision after ego’s action【30†L1732-L1736】.  Higher DRAC means more disruption. Our `rear\_decel\_peak\_3s` approximates this over 3s.
  * **Gap Acceptance (s):** Time gap at merge (distance to trailing traffic / ego speed) when ego merges. Typical accepted gap \~2–5 s in freeway ramps.
  * **Yield Events (count or %):** Number/fraction of times ego yields (slows/stops) for others. A direct measure of courtesy.
  * **Gap Yielded (s):** Additional headway given to other vehicles (e.g. increase in trailing headway when ego slows).
  * **Tactical Efficiency:** Ego’s own *distance traveled per second* (m/s) or *time lost* vs free-flow, measuring progress.
  * **Social/Benefit Score:** (If used) A composite (e.g. others’ time saved minus ego’s time lost), though we recommend decomposing into the above.
* ### Field/PDE-Level Metrics (Risk Field)

  * **Total Risk Potential (sum of risk field) \[risk\_mass\_total]:** Integral of the spatio-temporal risk field (e.g. sum of pairwise risk values) at an instant (unit: \[m/s]²).  Reflects overall exposure【44†L85-L90】.
  * **Mean Risk per Vehicle:** Total risk divided by number of agents in frame. Normalizes exposure by density.
  * **Backward Risk Flux (unit: risk/s):** Spatial gradient of risk field in backward direction (toward trailing traffic).  High flux signals risk propagating upstream (analogous to shockwave).
  * **Risk Gradient Peaks:** Maximum spatial gradient magnitude of risk (high peaks indicate imminent conflict hotspots).
  * **Potential Field Integrals:** (e.g. *Safety Potential* from RSS) if implemented; we propose sticking to risk mass and flux for simplicity.

Each metric should be reported with units and context (e.g. *TTC\_min (s)*, *Avg\_speed (km/h)*).  For instance, **TTC\_min** = $\\min\_{t}\\min\_{neighbors}\\text{TTC}(ego,n,t)$【18†L143-L150】.  **DRAC** is $\\max\_{t}$ decel of follower needed to avoid collision.  **Total Risk** = $\\sum\_{i,j} r\_{ij}$ (pairwise risk potential).  See Table 2 for concise definitions and sources.

**Table 2 (Recommended Metrics):** Key metrics, formulas, typical units/thresholds, and citations.

|Metric \& Category|Definition \& Formula|Units|Typical Range/Threshold|Sources (examples)|
|-|-|-|-|-|
|**Avg Traffic Speed** (sys)|Mean of all vehicle speeds in network.|km/h|—|—|
|**Flow (q)** (sys)|Vehicles passing a point per hr.|veh/hr|—|—|
|**Efficiency Index (EI)** (sys)|$EI=V/V\_{\\max} + \\text{spacing regularity}$ (e.g. Andreotti’s)|(unitless)|EI↑ = better|【53†L58-L61】|
|**Safety–Eff. Index (SEI)** (sys)|Combines EI with TTC component (e.g. penalize TTC<1.5s)|(unitless)|Higher = better|【53†L58-L61】|
|**Shockwave Count/Speed** (sys)|Number of backward propagating waves; or fit speed $v\_s$ (m/s).|m/s or count|$v\_s\\sim$ –1– –5 m/s|【57†L149-L157】【14†L75-L82】|
|**TTC** (agent)|$TTC = \\min\_{\\tau>0}{d(t+\\tau)=0}$ between ego \& another【18†L143-L150】|s|critical ≤1–1.5 s 【18†L143-L150】|【18†L143-L150】|
|**THW** (agent)|$THW = \\frac{gap}{v\_{rel}}$ to lead car【54†L133-L139】|s|target ≥1.8–3 s【54†L133-L139】|【54†L133-L139】|
|**Minimum Distance** (agent)|$\\min\_{j,t} \|p\_{ego}(t)-p\_j(t)\|$|m|—|—|
|**DRAC (Decel)** (agent)|Required decel for follower to avoid collision【30†L1732-L1736】|m/s²|collision if > \~4 m/s²|【30†L1732-L1736】|
|**Peak Follower Decel** (agent)|$\\max|\\ddot{v}\_{follower}|$ in 3s window|m/s²|
|**Gap Accepted** (agent)|Time gap at merge = $d\_{rear}/v\_{ego}$|s|typical \~3–5 s|\[Traffic merging studies]|
|**Yield Event Count** (agent)|# times ego yields to another (slows/stops for others)|count (%)|—|\[Cooperative driving lit.]|
|**Avg Speed (ego)** (agent)|Ego’s distance traveled per unit time|m/s or km/h|—|—|
|**Total Risk Mass** (field)|$\\sum\_{i<j} r\_{ij}(t)$ from risk field PDE (aggregated potential)|(m/s)²|—|【44†L85-L90】 (risk field)|
|**Risk per Agent** (field)|Total Risk Mass ÷ (# agents)|(m/s)²|—|—|
|**Backward Risk Flux** (field)|Spatial derivative $\\partial\_x R$ toward traffic rear (normalized)|(m/s)²/s|—|Derived (shockwave analog)|
|**Risk Gradient Peak** (field)|$\\max|\\nabla R|$ at time t|(m/s)³|
|**Social Benefit Score** (sys/agent)|Composite e.g. (others’ time saved – ego time lost)|s or unitless|– value (drop if unclear)|\[Proposed composite]|

(*“sys”=system-level, “agent”=vehicle-level, “field”=risk-field-level.*)

## Novel Features: Keep, Rename, or Drop

From our current features:

* **Keep/Rename:**

  * *Risk mass \& flux:* Retain the risk‐field metrics as **Total Risk Potential** and **Backward Risk Flux**, just renaming for clarity.  These are novel but align with “safety potential” concepts in literature【44†L85-L90】.  Also keep *risk\_mass\_others* (imposed risk on others).
  * *rear\_decel\_peak\_3s:* Keep as **Follower Peak Deceleration (3s)**, an explicit measure of disruption (cf. DRAC【30†L1732-L1736】).
  * *follower\_thw\_delta:* Keep as **Headway Change**, directly comparable to THW metrics【54†L133-L139】.
  * *yield flags:* Keep as binary events (Yielded/Given and Received). These map to courtesy proxies (count of yield events).
  * *gap\_offered/gap\_taken:* Keep to measure accepted gap (rename for clarity: **Gap in Front / Rear**).
  * *progress / speed features:* Keep ego-speed measures (distance or time).
* **Combine/Drop:**

  * *commitment\_score, oscillation\_count:* Likely drop or replace. These are not standard metrics. They capture control quality (jerkiness) but add complexity; instead rely on risk‐field smoothness or worst-case values.
  * *social\_benefit\_score:* Rather than a single opaque score, break it into interpretable metrics above (e.g. others’ delay vs. ego delay). If used, clarify definition and unit. Possibly drop in favor of reporting individual metrics (yield rates, speed/delay differences).
  * *n\_int\_car, n\_int\_cyc:* Keep for normalization purposes (density), but do not count as “metrics” per se; use them to normalize throughput or risk per agent.

Aligning with literature requires focusing on objective, quantitative measures (TTC, DRAC, etc.) rather than subjective aggregates. Novel features should either map to established concepts (e.g. risk mass → safety potential field) or be clearly defined if retained.

## Visualization and Tables

**Table:** We suggest a summary table (like Table 2 above) of chosen metrics, categories, definitions, and citations. Another table could list mapping of *our features* to *standard metrics* (as in Table 1).

**Figures:** Visualization panels could include:

* **Risk-Field Heatmap:** e.g. a heatmap overlay of the risk potential field on BEV trajectories at one or more time points, showing “hot zones” of high risk. This highlights spatial risk propagation.
* **Time-Series Plots:** e.g. plot *Total Risk Mass* and *Mean Risk per Vehicle* versus time, possibly overlaid with shockwave events. Also plot *ego speed* or *flow* vs. time.
* **Pareto Scatter (Risk vs Efficiency):** For each scenario, scatter ego’s travel distance or average speed (x-axis) against imposed *Total Risk* (y-axis). Mark points for different policies. This visualizes the safety–efficiency tradeoff (lower risk \& higher speed is better).
* **Bar Chart of Agent Metrics:** Compare group-average TTC, DRAC, yield-rate under different driving behaviors (as in Jiang2023【30†L1732-L1736】).
* **Mermaid Flowchart:** A flowchart of data processing to metrics. For example:

```mermaid
flowchart TD
  A\[Raw Trajectory Data (BEV)] --> B\[Preprocess \& Feature Extraction]
  B --> C\[Risk Field PDE Computation]
  B --> D\[Event Detection (lane-changes, yields)]
  C --> E\[Field Metrics: Total Risk, Flux, etc.]
  D --> F\[Agent Metrics: TTC, THW, DRAC, Gaps, Yields]
  B --> G\[System Metrics: Flow, Avg Speed, SEI]
  E --> H\[Statistical Aggregation \& Normalization]
  F --> H
  G --> H
  H --> I\[Reporting: Tables \& Figures]
```

This pipeline clarifies how raw trajectories yield multiple layers of metrics (risk-field, agent, system) and how they feed into analysis.

## Statistical Validation and Robustness

We should validate metrics statistically across scenarios:

* **Significance Tests:** Use standard tests (t-test, ANOVA, or nonparametric tests if needed) to compare means of key metrics between groups (e.g. courteous vs. egoistic policies), reporting *p*-values and confidence intervals【30†L1578-L1582】. Jiang et al. illustrated using 95% CIs over multiple simulation seeds【30†L1578-L1582】. Report effect sizes (e.g. Cohen’s *d*) for practical significance.
* **Normalization:** Normalize aggregate measures by traffic volume or agent count to account for density differences. For example, report *risk per vehicle* and *distance per vehicle* to compare across scenarios.  Similarly, metrics like throughput should be per lane-mile or per vehicle.
* **Bootstrapping/CI:** For non-Gaussian metrics (like min TTC) use bootstrapping to estimate confidence intervals. Report median and IQR if distributions are skewed.
* **Robustness Checks:** Test that metrics behave as expected under simple changes (e.g., uniform slower traffic should raise TTC and reduce risk mass). Check sensitivity to parameter choices (e.g. the PDE diffusion constant).  Ensure no metric is dominated by outliers (e.g. use clipped DRAC).
* **Cross-Validation:** If data allows, split into training/validation (especially for learned metrics), though our setting is mostly offline analysis.

Practices from literature include multiple runs to average out randomness【30†L1578-L1582】, and ensuring any claimed improvement (e.g. “more courteous”) is statistically significant not anecdotal.

## References

Key sources include transportation and ITS research on safety metrics (e.g. surrogate safety measure surveys【56†L1-L4】【18†L143-L150】), fundamental traffic flow texts【57†L149-L157】, and recent autonomous/AV driving studies on courtesy and efficiency【30†L1732-L1736】【53†L58-L61】.  Seminal works like Treiber \& Kesting (2013) underpin shockwave theory【57†L149-L157】.  Our metric definitions draw on these primary sources to ensure alignment with established practice.

