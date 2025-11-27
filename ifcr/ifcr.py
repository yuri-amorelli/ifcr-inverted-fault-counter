import numpy as np
import pandas as pd


def build_ifcr_counter(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    event_col: str = "event",
    counter_col: str = "ifcr_counter",
) -> pd.DataFrame:
    """
    Simple IFCR implementation.

    Build an inverted fault counter (IFCR), which represents the number of
    steps until the next fault. The counter resets to 0 on faults and
    increases when moving backwards in time.
    """

    df = df.copy()
    df = df.sort_values(timestamp_col).reset_index(drop=True)

    event_positions = df.index[df[event_col] == 1].tolist()

    counter = np.zeros(len(df), dtype=int)

    for i, event_idx in enumerate(event_positions):
        prev_event_idx = event_positions[i - 1] if i > 0 else -1
        steps = 0
        for j in range(event_idx, prev_event_idx, -1):
            counter[j] = steps
            steps += 1

    df[counter_col] = counter
    return df


def build_ifcr_counter_segmented(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    event_col: str = "event",
    counter_col: str = "ifcr_counter",
) -> pd.DataFrame:
    """
    Segment-based IFCR implementation (closer to the original research code).

    The data is split into segments between faults, and inside each segment the
    counter is built by walking backwards from every fault.
    """

    df = df.copy()
    df = df.sort_values(timestamp_col).reset_index(drop=True)

    segment_ids = []
    current_segment = 0
    for flag in df[event_col].values:
        segment_ids.append(current_segment)
        if flag == 1:
            current_segment += 1

    df["_ifcr_segment"] = segment_ids

    counter = np.zeros(len(df), dtype=int)

    for seg_id, idx in df.groupby("_ifcr_segment").groups.items():
        seg_idx = np.array(sorted(idx))
        seg_events = df.loc[seg_idx, event_col].values

        fault_positions = seg_idx[seg_events == 1]

        if len(fault_positions) == 0:
            continue

        start = seg_idx[0]

        for fp in fault_positions:
            length = fp - start + 1
            counter[start : fp + 1] = np.flip(np.arange(length))
            start = fp + 1

    df[counter_col] = counter
    return df.drop(columns="_ifcr_segment")
