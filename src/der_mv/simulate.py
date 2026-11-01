"""Synthetic smart-thermostat fleet with demand-response events and known ground truth.

Why simulate? An M&V method's *bias* can only be measured when the true
counterfactual is known. Real-data checks (placebo tests, randomised control
groups) are applied on top, and they are the same checks used in production
where the truth is never observed.

Physical and behavioural model (per home, hourly, local standard time):

* Non-cooling load: a diurnal occupancy shape times a home-specific base level.
* Cooling load: proportional to how far the *lagged* outdoor temperature (thermal
  mass) exceeds a home-specific balance point, capped by AC size, with a weekday
  daytime setback.
* Day-to-day behaviour: a home-day multiplicative random effect, plus hourly noise.
* Events (treated homes only): pre-cooling before the event, a setpoint raise
  that sheds a home-specific share of AC load and fades as the house warms,
  customer overrides (more likely on hotter days), devices that miss the
  signal, and snapback (rebound) after the event.
* Telemetry: random missing readings, multi-day device outages and duplicated
  rows from ingestion retries.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from solar_rooftop.config import SiteConfig
from solar_rooftop.weather import load_weather

from .config import FleetConfig

TREATMENT, CONTROL = "treatment", "control"


@dataclass(frozen=True)
class Fleet:
    """Simulation output: observable tables plus hidden ground truth."""

    # Observable (what a DERMS platform would have)
    devices: pd.DataFrame  # device_id, grp
    weather: pd.DataFrame  # ts, temp_c
    events: pd.DataFrame  # event_id, event_date, start_ts, end_ts
    telemetry: pd.DataFrame  # device_id, ts, kw, received_at
    # Hidden ground truth (only available because this is a simulation)
    hours: pd.DatetimeIndex
    counterfactual_kw: np.ndarray  # (n_homes, n_hours) load with no events
    actual_kw: np.ndarray  # (n_homes, n_hours) true load incl. event response
    true_impact: pd.DataFrame  # device_id, event_id, ts, impact_kw (treated, event window)


def season_hours(fleet: FleetConfig) -> pd.DatetimeIndex:
    start = pd.Timestamp(fleet.season_start)
    end = pd.Timestamp(fleet.season_end) + pd.Timedelta(23, unit="h")
    return pd.date_range(start, end, freq="h", name="ts")


def outdoor_temperature(site: SiteConfig, hours: pd.DatetimeIndex) -> pd.Series:
    """Hour-beginning outdoor temperature (deg C) in local standard time."""
    wx = load_weather(site).data["temp_air"]
    # TMY data is labelled mid-hour after alignment; convert to hour-beginning, naive LST.
    idx = (wx.index - pd.Timedelta(30, unit="min")).tz_localize(None)
    temp = pd.Series(wx.to_numpy(), index=idx)
    return temp.reindex(hours).interpolate(limit_direction="both").rename("temp_c")


def _diurnal_shape(hod: np.ndarray, weekend: np.ndarray) -> np.ndarray:
    """Non-cooling load multiplier by hour of day (morning and evening peaks)."""
    weekday = np.array(
        [0.75, 0.7, 0.7, 0.7, 0.75, 0.9, 1.15, 1.25, 1.05, 0.85, 0.8, 0.8,
         0.8, 0.8, 0.85, 0.95, 1.1, 1.3, 1.45, 1.5, 1.4, 1.2, 1.0, 0.85]
    )  # fmt: skip
    weekend_shape = np.array(
        [0.8, 0.75, 0.7, 0.7, 0.7, 0.75, 0.85, 1.0, 1.15, 1.2, 1.15, 1.1,
         1.1, 1.05, 1.05, 1.1, 1.2, 1.35, 1.45, 1.45, 1.35, 1.2, 1.05, 0.9]
    )  # fmt: skip
    return np.where(weekend, weekend_shape[hod], weekday[hod])


def select_event_days(temp: pd.Series, fleet: FleetConfig) -> list[pd.Timestamp]:
    """Utilities dispatch on the hottest weekdays: this is what biases naive baselines."""
    daily_max = temp.resample("D").max()
    first_allowed = daily_max.index[0] + pd.Timedelta(fleet.min_lookback_days, unit="D")
    cand = daily_max[(daily_max.index.dayofweek < 5) & (daily_max.index >= first_allowed)]
    return sorted(cand.nlargest(fleet.n_events).index)


def simulate_fleet(site: SiteConfig, fleet: FleetConfig) -> Fleet:
    rng = np.random.default_rng(fleet.seed)
    hours = season_hours(fleet)
    n, n_h = fleet.n_homes, len(hours)
    temp = outdoor_temperature(site, hours)

    hod = np.asarray(hours.hour)
    weekend = np.asarray(hours.dayofweek >= 5)
    day_idx = np.asarray((hours.normalize() - hours[0].normalize()).days)
    n_days = int(day_idx.max()) + 1

    # --- Home characteristics (heterogeneous fleet) -------------------------
    base = rng.lognormal(np.log(0.8), 0.3, n)  # kW
    # Outdoor deg C where cooling starts (below indoor setpoint: solar and internal gains)
    balance = rng.uniform(17.0, 21.0, n)
    slope = rng.uniform(0.15, 0.35, n)  # kW per deg C above balance (typical central AC)
    ac_cap = rng.uniform(2.5, 4.5, n)  # kW
    shed_frac = rng.uniform(0.35, 0.75, n)  # share of AC load shed at event start
    rebound_frac = rng.uniform(0.3, 0.7, n)  # share of shed energy recovered afterwards

    # --- Counterfactual load ---------------------------------------------------
    t_eff = temp.ewm(halflife=3).mean().to_numpy()  # thermal mass lag
    setback = np.where(~weekend & (hod >= 9) & (hod < 16), 0.8, 1.0)
    ac = np.clip(slope[:, None] * (t_eff[None, :] - balance[:, None]) * setback[None, :],
                 0, ac_cap[:, None])  # fmt: skip
    other = base[:, None] * _diurnal_shape(hod, weekend)[None, :]
    day_effect = rng.normal(1.0, 0.08, (n, n_days))[:, day_idx]
    noise = rng.normal(0.0, 0.12, (n, n_h))
    counterfactual = np.clip((other + ac) * day_effect + noise, 0.05, None)
    ac_actual = ac * day_effect  # cooling component as realised that day

    # --- Treatment assignment (randomised holdout) ------------------------------
    n_control = round(n * fleet.holdout_share)
    is_control = np.zeros(n, dtype=bool)
    is_control[rng.permutation(n)[:n_control]] = True
    treated_idx = np.flatnonzero(~is_control)

    # --- Events ----------------------------------------------------------------
    event_days = select_event_days(temp, fleet)
    daily_max = temp.resample("D").max()
    hot_ref = float(daily_max.loc[event_days].mean())
    actual = counterfactual.copy()
    s = fleet.event_start_hour
    decay = np.linspace(1.0, 0.65, fleet.event_hours)  # shed fades as the house warms
    rebound_w = np.linspace(1.0, 0.4, fleet.snapback_hours)
    rebound_w /= rebound_w.sum()

    event_rows, impact_frames = [], []
    for k, day in enumerate(event_days):
        h0 = int(hours.get_loc(day + pd.Timedelta(s, unit="h")))
        win = slice(h0, h0 + fleet.event_hours)
        pre = slice(h0 - fleet.precool_hours, h0)
        post = slice(h0 + fleet.event_hours, h0 + fleet.event_hours + fleet.snapback_hours)
        event_rows.append(
            {
                "event_id": k + 1,
                "event_date": day.date(),
                "start_ts": hours[h0],
                "end_ts": hours[h0] + pd.Timedelta(fleet.event_hours, unit="h"),
            }
        )

        idx = treated_idx[rng.random(len(treated_idx)) >= fleet.offline_rate]  # got the signal
        # Overrides are more likely on hotter days; they start in a random event hour.
        p_override = np.clip(fleet.override_rate * (1 + 0.15 * (daily_max[day] - hot_ref)), 0, 1)
        override_at = np.where(
            rng.random(len(idx)) < p_override,
            rng.integers(0, fleet.event_hours, len(idx)),
            fleet.event_hours,
        )
        active = np.arange(fleet.event_hours)[None, :] < override_at[:, None]

        shed = ac_actual[idx, win] * shed_frac[idx, None] * decay[None, :] * active
        precool = 0.25 * ac_actual[idx, pre]
        rebound = (shed.sum(axis=1) * rebound_frac[idx])[:, None] * rebound_w[None, :]

        actual[idx, win] -= shed
        actual[idx, pre] += precool
        actual[idx, post] += rebound

        impact = counterfactual[treated_idx, win] - actual[treated_idx, win]
        impact_frames.append(
            pd.DataFrame(
                {
                    "device_id": np.repeat(treated_idx + 1, fleet.event_hours),
                    "event_id": k + 1,
                    "ts": np.tile(hours[win], len(treated_idx)),
                    "impact_kw": impact.ravel(),
                }
            )
        )
    actual = np.clip(actual, 0.05, None)

    devices = pd.DataFrame(
        {
            "device_id": np.arange(1, n + 1),
            "grp": np.where(is_control, CONTROL, TREATMENT),
        }
    )
    return Fleet(
        devices=devices,
        weather=temp.reset_index(),
        events=pd.DataFrame(event_rows),
        telemetry=_observe(actual, hours, fleet, rng),
        hours=hours,
        counterfactual_kw=counterfactual,
        actual_kw=actual,
        true_impact=pd.concat(impact_frames, ignore_index=True),
    )


def _observe(
    actual: np.ndarray, hours: pd.DatetimeIndex, fleet: FleetConfig, rng: np.random.Generator
) -> pd.DataFrame:
    """Turn true load into messy IoT telemetry (long format, missing rows absent)."""
    n, n_h = actual.shape
    observed = np.round(actual, 3)
    missing = rng.random((n, n_h)) < fleet.telemetry_gap_rate
    for i in rng.choice(n, round(n * fleet.long_outage_share), replace=False):
        length = int(rng.integers(3, 11)) * 24
        start = int(rng.integers(0, n_h - length))
        missing[i, start : start + length] = True

    dev, hr = np.nonzero(~missing)
    tel = pd.DataFrame(
        {
            "device_id": (dev + 1).astype(np.int32),
            "ts": hours[hr],
            "kw": observed[dev, hr].astype(np.float32),
        }
    )
    delay = pd.to_timedelta(rng.integers(1, 15, len(tel)), unit="min")
    tel["received_at"] = tel["ts"] + pd.Timedelta(1, unit="h") + delay

    dup = tel.sample(frac=fleet.duplicate_rate, random_state=fleet.seed).copy()
    dup["received_at"] += pd.Timedelta(2, unit="h")  # a retry arrives later
    return pd.concat([tel, dup], ignore_index=True)
