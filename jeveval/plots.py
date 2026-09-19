"""Report figures: reliability diagrams, sweep curves, and E2's two-curve plot.

Three constraints shaped this module.

*No global state.* Every function takes an explicit output path, renders inside
a `plt.rc_context` so it never mutates the process-wide rcParams, closes its
figure, and returns the path it wrote. The backend is forced to Agg before
pyplot is imported: the harness runs headless and an unset backend is a crash
halfway through a report, not at import.

*Bin counts are part of the reading.* The plan asks for a reliability diagram
per condition and says to read its shape. A bin holding 3 predictions and a bin
holding 300 are the same dot on a naive diagram, and the 3-sample bin is where
spurious non-monotonicity comes from -- which under `tiers.py` is a
"Doesn't work". So counts are encoded three times: marker area, a count
histogram under the main axes, and 95% Wilson intervals on the observed
frequency.

*A plot without its n is not usable in the report.* `n` is a required keyword
on the functions that cannot derive it, and it is printed in a caption line on
every figure. Same reasoning as the baseline requirement in `tiers.py`: make it
structural rather than something the caller has to remember.

Colors come from a palette validated for colorblind separation and contrast
against the light surface (worst all-pairs CVD dE 9.2, normal-vision dE 24.0).
Series are also distinguished by marker shape, so identity never rests on color
alone. Slots are assigned in fixed order and never cycled; past eight series the
functions raise rather than reuse a hue.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")  # must precede the pyplot import

import matplotlib.pyplot as plt  # noqa: E402

from .tiers import reliability_is_monotone  # noqa: E402

# --------------------------------------------------------------------------
# Palette and defaults
# --------------------------------------------------------------------------

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e5e4e0"
# Reference marks -- the diagonal, chance lines, the phase transition. Neutral,
# because they are not data series.
REFERENCE = "#8a8983"
# Recessive fill for the count histogram: the blue ramp's step 200, so it reads
# as the same quantity as the series line without competing with it.
HISTOGRAM = "#9ec5f4"

SERIES: tuple[str, ...] = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)
MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "v", "P", "X", "*")

DPI = 160
BAND_ALPHA = 0.16

_RC: dict[str, Any] = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": GRID,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "semibold",
    "axes.labelsize": 9.5,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "legend.frameon": False,
    "lines.linewidth": 2.0,
    "font.size": 9.5,
    # Fixed so that two runs of the same call produce byte-identical PNGs.
    "svg.hashsalt": "jeveval",
}

# PNG metadata is otherwise stamped with the matplotlib version, which would
# make output bytes depend on the environment rather than on the data.
_METADATA = {"Software": "jeveval.plots"}


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def _out(path: str | Path) -> Path:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _n_text(n: int | Sequence[int] | Mapping[Any, int] | None, unit: str = "per condition") -> str:
    """Render n for the caption, whether it is one number or one per point."""
    if n is None:
        return "n not stated"
    if isinstance(n, Mapping):
        values = list(n.values())
    elif isinstance(n, (int, float)):
        values = [int(n)]
    else:
        values = [int(v) for v in n]
    if not values:
        return "n not stated"
    lo, hi = min(values), max(values)
    if lo == hi:
        return f"n = {lo} {unit}"
    return f"n = {lo}-{hi} {unit}"


def _finish(fig, path: Path, caption: str) -> Path:
    """Place the caption, write the file, close the figure, return the path."""
    fig.text(0.012, 0.012, caption, ha="left", va="bottom", fontsize=7.5, color=INK_MUTED)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", metadata=_METADATA)
    plt.close(fig)
    return path


def _wilson(successes: float, count: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval. Chosen over the normal approximation because
    reliability bins routinely sit at 0 or 1 with a handful of samples, where the
    normal interval has zero width and lies."""
    if count <= 0:
        return (0.0, 1.0)
    p = successes / count
    denom = 1.0 + z * z / count
    centre = (p + z * z / (2 * count)) / denom
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _tick_label(v: Any) -> str:
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def _series_style(i: int) -> tuple[str, str]:
    if i >= len(SERIES):
        raise ValueError(
            f"{i + 1} series requested but the validated palette has {len(SERIES)} slots; "
            "fold the rest into an 'other' series or split the figure into small multiples "
            "rather than generating a colour"
        )
    return SERIES[i], MARKERS[i]


# --------------------------------------------------------------------------
# Reliability diagram
# --------------------------------------------------------------------------

_BIN_FIELDS: dict[str, tuple[str, ...]] = {
    "mean_predicted": (
        "mean_predicted",
        "mean_prob",
        "mean_predicted_prob",
        "mean_p",
        "predicted",
        "p_mean",
        "confidence",
    ),
    "observed": (
        "observed",
        "observed_frequency",
        "observed_freq",
        "frac_positive",
        "empirical",
        "actual",
    ),
    # Deliberately not "weight": metrics.reliability_bins emits that as a
    # fraction of the total, and reading it as a count would size every marker
    # and bar wrongly while looking plausible.
    "count": ("count", "n", "size", "total"),
    "lo": ("lo_edge", "lo", "bin_lo", "left", "low", "edge_lo", "start"),
    "hi": ("hi_edge", "hi", "bin_hi", "right", "high", "edge_hi", "end"),
    "observed_lo": ("observed_lo", "ci_lo", "lower"),
    "observed_hi": ("observed_hi", "ci_hi", "upper"),
}


def _bin_field(b: Any, field: str) -> float | None:
    for alias in _BIN_FIELDS[field]:
        if isinstance(b, Mapping):
            if alias in b and b[alias] is not None:
                return float(b[alias])
        else:
            value = getattr(b, alias, None)
            if value is not None:
                return float(value)
    return None


def _normalize_bins(bins: Sequence[Any]) -> list[dict[str, float]]:
    """Accept whatever shape the metrics module hands over.

    `metrics.py` is written alongside this file, so the field names are matched
    by alias rather than by a shared class. The three quantities a reliability
    bin must carry are fixed by the plan's definition of ECE: mean predicted
    probability, observed frequency, and the number of predictions in the bin.
    Bin edges are optional; without them the bins are assumed to be the plan's
    10 equal-width bins in order, which means empty bins have to be included
    rather than dropped.
    """
    if not bins:
        raise ValueError("reliability_diagram needs at least one bin")
    out: list[dict[str, float]] = []
    width = 1.0 / len(bins)
    for i, b in enumerate(bins):
        count = _bin_field(b, "count")
        if count is None:
            raise ValueError(
                f"bin {i} has no count; tried {_BIN_FIELDS['count']}. "
                "The count is not optional -- it is what distinguishes a 3-sample bin "
                "from a 300-sample one."
            )
        lo = _bin_field(b, "lo")
        hi = _bin_field(b, "hi")
        if lo is None or hi is None:
            lo, hi = i * width, (i + 1) * width
        mean_p = _bin_field(b, "mean_predicted")
        observed = _bin_field(b, "observed")
        if count > 0 and (mean_p is None or observed is None):
            raise ValueError(
                f"bin {i} holds {count:g} predictions but is missing "
                f"{'mean_predicted' if mean_p is None else 'observed'}"
            )
        row = {
            "mean_predicted": (lo + hi) / 2 if mean_p is None else mean_p,
            "observed": 0.0 if observed is None else observed,
            "count": count,
            "lo": lo,
            "hi": hi,
        }
        for key in ("observed_lo", "observed_hi"):
            value = _bin_field(b, key)
            if value is not None:
                row[key] = value
        out.append(row)
    return out


def _ece(rows: Sequence[Mapping[str, float]]) -> float:
    total = sum(r["count"] for r in rows)
    if total <= 0:
        return 0.0
    return sum(r["count"] * abs(r["mean_predicted"] - r["observed"]) for r in rows) / total


def ece_from_bins(bins: Sequence[Any]) -> float:
    """The plan's ECE -- the count-weighted mean of |mean predicted - observed| --
    applied to bins that were already computed elsewhere.

    Here so a diagram can label itself when the caller does not pass a value.
    `jeveval.metrics` remains the authority: pass its number and this is not used.
    """
    return _ece(_normalize_bins(bins))


def reliability_diagram(
    bins: Sequence[Any],
    path: str | Path,
    title: str,
    *,
    condition: str | None = None,
    ece: float | None = None,
    n: int | None = None,
    monotone: bool | None = None,
    intervals: bool = True,
    xlabel: str = "mean predicted probability",
    ylabel: str = "observed frequency",
    series_label: str = "observed",
    note: str = "",
) -> Path:
    """Per-condition reliability diagram: the plan's required PNG.

    `bins` are the ECE bins in order, each carrying mean predicted probability,
    observed frequency and count (see `_normalize_bins` for accepted field
    names). Counts drive marker area, the histogram panel underneath, and the
    Wilson intervals, because equal-looking dots over unequal sample sizes is
    the specific way this plot misleads.

    `ece` and `monotone` are labelled on the figure. Both are recomputed from
    the bins when not supplied, using the same definitions `tiers.py` grades
    against, so the caption never contradicts the tier.
    """
    rows = _normalize_bins(bins)
    total = int(round(sum(r["count"] for r in rows)))
    filled = [r for r in rows if r["count"] > 0]
    if not filled:
        raise ValueError("every reliability bin is empty")
    if ece is None:
        ece = _ece(rows)
    if monotone is None:
        monotone = reliability_is_monotone(
            [r["observed"] for r in rows], [int(r["count"]) for r in rows]
        )
    if n is None:
        n = total

    max_count = max(r["count"] for r in filled)
    xs = [r["mean_predicted"] for r in filled]
    ys = [r["observed"] for r in filled]

    with plt.rc_context(_RC):
        fig, (ax, axh) = plt.subplots(
            2, 1, figsize=(5.6, 6.4), sharex=True, height_ratios=[3, 1]
        )

        ax.plot([0, 1], [0, 1], ls="--", lw=1.5, color=REFERENCE, zorder=1,
                label="perfectly calibrated")

        if intervals:
            lo_err, hi_err = [], []
            for r in filled:
                # metrics.reliability_bins already carries a Wilson interval per
                # bin; recomputing it here would be a second definition of the
                # same number.
                if "observed_lo" in r and "observed_hi" in r:
                    lo, hi = r["observed_lo"], r["observed_hi"]
                else:
                    lo, hi = _wilson(r["observed"] * r["count"], int(r["count"]))
                lo_err.append(max(0.0, r["observed"] - lo))
                hi_err.append(max(0.0, hi - r["observed"]))
            ax.errorbar(
                xs, ys, yerr=[lo_err, hi_err], fmt="none", ecolor=SERIES[0],
                elinewidth=1.2, capsize=3, alpha=0.55, zorder=2,
            )

        ax.plot(xs, ys, color=SERIES[0], lw=1.6, alpha=0.85, zorder=3, label=series_label)
        # Marker area tracks bin count; the surface-coloured ring keeps
        # neighbouring markers legible where they overlap.
        sizes = [(8 + 20 * math.sqrt(r["count"] / max_count)) ** 2 for r in filled]
        ax.scatter(
            xs, ys, s=sizes, color=SERIES[0], edgecolor=SURFACE, linewidth=1.5, zorder=4,
        )

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel(ylabel)
        ax.set_title(title if condition is None else f"{title}\n{condition}")
        ax.legend(loc="upper left")
        shape = "monotone" if monotone else "NON-MONOTONE"
        ax.text(
            0.98, 0.04, f"ECE = {ece:.3f}\n{shape}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9, color=INK,
            bbox={"facecolor": SURFACE, "edgecolor": GRID, "boxstyle": "round,pad=0.35"},
        )

        widths = [(r["hi"] - r["lo"]) * 0.88 for r in rows]  # gap between bars
        centres = [(r["hi"] + r["lo"]) / 2 for r in rows]
        counts = [r["count"] for r in rows]
        axh.bar(centres, counts, width=widths, color=HISTOGRAM, edgecolor=SURFACE, linewidth=1.0)
        top = max(counts) if max(counts) > 0 else 1
        for c, v in zip(centres, counts):
            if v > 0:
                axh.text(c, v + top * 0.04, f"{int(round(v))}", ha="center", va="bottom",
                         fontsize=7, color=INK_MUTED)
        axh.set_ylim(0, top * 1.3)
        axh.set_xlabel(xlabel)
        axh.set_ylabel("predictions\nin bin")

        smallest = int(round(min(r["count"] for r in filled)))
        caption = (
            f"{_n_text(n, 'predictions')}; {len(filled)} of {len(rows)} bins occupied, "
            f"smallest holds {smallest}. Bars: 95% Wilson intervals; marker area tracks bin count."
        )
        if note:
            caption += f" {note}"
        fig.tight_layout(rect=(0, 0.035, 1, 1))
        fig.subplots_adjust(hspace=0.08)
        return _finish(fig, _out(path), caption)


# --------------------------------------------------------------------------
# Generic sweep curve
# --------------------------------------------------------------------------


def _as_series(
    y: Sequence[float] | Mapping[str, Sequence[float]], default_label: str
) -> dict[str, Sequence[float]]:
    if isinstance(y, Mapping):
        return dict(y)
    return {default_label: y}


def curve(
    x: Sequence[Any],
    y: Sequence[float] | Mapping[str, Sequence[float]],
    path: str | Path,
    title: str,
    xlabel: str,
    ylabel: str,
    *,
    n: int | Sequence[int] | Mapping[Any, int],
    yerr: Sequence[float] | Mapping[str, Sequence[float]] | None = None,
    band: tuple[Sequence[float], Sequence[float]]
    | Mapping[str, tuple[Sequence[float], Sequence[float]]]
    | None = None,
    baseline: float | None = None,
    baseline_label: str = "baseline",
    series_label: str = "accuracy",
    logx: bool = False,
    logy: bool = False,
    ylim: tuple[float, float] | None = None,
    xticks: Sequence[Any] | None = None,
    n_unit: str = "per condition",
    note: str = "",
) -> Path:
    """One measure against one difficulty axis, with optional error bands.

    Covers the plan's accuracy-against-difficulty, accuracy-against-position and
    latency-against-question-count figures. `y` is one sequence or a mapping of
    label to sequence; `yerr` gives symmetric half-widths and `band` gives
    explicit (lower, upper) pairs for asymmetric intervals such as Wilson.

    `baseline` draws the majority-class or chance line. The plan requires every
    result to be reported next to its baseline, and an accuracy curve with no
    baseline line invites the reader to supply the wrong one.

    `x` may be non-numeric (encodings, channels); it is then plotted at integer
    positions with the values as tick labels.
    """
    if len(x) == 0:
        raise ValueError(
            "curve was given no x values; an axis with no points renders as a blank "
            "figure with a caption, which is worse in a report than a missing figure"
        )
    series = _as_series(y, series_label)
    if not series:
        raise ValueError("curve needs at least one series")
    errs = None if yerr is None else _as_series(yerr, series_label)
    bands: dict[str, tuple[Sequence[float], Sequence[float]]] | None
    if band is None:
        bands = None
    elif isinstance(band, Mapping):
        bands = dict(band)
    else:
        bands = {series_label: band}

    # An interval keyed by a label no series uses would be dropped in silence,
    # and the figure would go into the report with its uncertainty missing and
    # nothing to show that it had been asked for. The usual way to land here is
    # passing `y` as a mapping and `yerr` as a bare sequence, which then gets
    # keyed by the unused `series_label` default.
    for kind, supplied in (("yerr", errs), ("band", bands)):
        if not supplied:
            continue
        stray = [label for label in supplied if label not in series]
        if stray:
            raise ValueError(
                f"{kind} carries {stray} but the series are {list(series)}; "
                f"key {kind} by series label (or pass one series) rather than letting "
                "the interval be dropped"
            )

    numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in x)
    positions = list(x) if numeric else list(range(len(x)))
    labels = None if numeric else [str(v) for v in x]

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(6.4, 4.4))

        for i, (label, ys) in enumerate(series.items()):
            if len(ys) != len(positions):
                raise ValueError(f"series {label!r} has {len(ys)} points for {len(positions)} x values")
            color, marker = _series_style(i)
            ax.plot(positions, ys, color=color, marker=marker, markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.0, label=label, zorder=3 + i)
            lo = hi = None
            if bands and label in bands:
                lo, hi = bands[label]
                if len(lo) != len(positions) or len(hi) != len(positions):
                    raise ValueError(
                        f"band for {label!r} has {len(lo)}/{len(hi)} points for "
                        f"{len(positions)} x values"
                    )
            elif errs and label in errs:
                e = errs[label]
                if len(e) != len(positions):
                    raise ValueError(
                        f"yerr for {label!r} has {len(e)} points for {len(positions)} x values"
                    )
                lo = [v - d for v, d in zip(ys, e)]
                hi = [v + d for v, d in zip(ys, e)]
            if lo is not None and hi is not None:
                ax.fill_between(positions, lo, hi, color=color, alpha=BAND_ALPHA, linewidth=0,
                                zorder=2)

        if baseline is not None:
            ax.axhline(baseline, color=REFERENCE, ls=":", lw=1.5, zorder=1,
                       label=f"{baseline_label} ({baseline:.3g})")

        if logx:
            ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        if labels is not None:
            ax.set_xticks(positions)
            ax.set_xticklabels(labels)
        elif xticks is not None:
            ticks = list(xticks)
            ax.set_xticks(ticks)
            ax.set_xticklabels([_tick_label(t) for t in ticks])
            if logx:
                # The log formatter labels decades only, which would leave the
                # positions the experiment actually sampled unlabelled.
                ax.minorticks_off()
        if ylim is not None:
            ax.set_ylim(*ylim)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        # A single series is named by the title and axis; a legend box would add
        # nothing. Two or more, or a baseline line, need one.
        if len(series) > 1 or baseline is not None:
            ax.legend(loc="best")

        caption = _n_text(n, n_unit)
        if note:
            caption += f". {note}"
        fig.tight_layout(rect=(0, 0.045, 1, 1))
        return _finish(fig, _out(path), caption)


# --------------------------------------------------------------------------
# E2: predicted probability against the true satisfiable fraction
# --------------------------------------------------------------------------

SAT_TRANSITION = 4.26


def crossover_x(
    x: Sequence[float], y: Sequence[float], level: float = 0.5
) -> float | None:
    """First x at which y crosses `level`, by linear interpolation between the
    bracketing points. None when the curve never crosses -- which for E2 is
    itself the finding, since a curve too flat to cross 0.5 has no transition.

    A curve that lands exactly on `level` counts as crossing there, including at
    the last sampled point. The endpoint is checked separately because the
    sign-change test needs a bracketing pair and the final sample has none: a
    sweep whose predicted curve reaches 0.5 exactly at its last ratio would
    otherwise be reported as never crossing, which is the opposite finding.
    """
    if len(x) != len(y):
        raise ValueError(f"crossover_x: {len(x)} x values for {len(y)} y values")
    for i in range(len(x) - 1):
        a, b = y[i], y[i + 1]
        if (a - level) == 0:
            return float(x[i])
        if (a - level) * (b - level) < 0:
            t = (level - a) / (b - a)
            return float(x[i]) + t * (float(x[i + 1]) - float(x[i]))
    if y and (y[-1] - level) == 0:
        return float(x[-1])
    return None


def calibration_curve_vs_truth(
    ratios: Sequence[float],
    predicted: Sequence[float],
    truth: Sequence[float],
    path: str | Path,
    *,
    n: int | Sequence[int] | Mapping[Any, int],
    title: str = "Mean predicted P(SAT) against the true satisfiable fraction",
    condition: str | None = None,
    predicted_err: Sequence[float] | None = None,
    truth_err: Sequence[float] | None = None,
    transition: float = SAT_TRANSITION,
    crossover: float | None = None,
    xlabel: str = "clause-to-variable ratio m/n",
    ylabel: str = "P(satisfiable)",
    note: str = "",
) -> Path:
    """E2's headline figure: the predicted curve and the true curve on one axis.

    Both series are probabilities, so they share one y axis -- the comparison is
    the entire point, and a second scale would make any gap between them
    meaningless. The 4.26 transition is marked, and the predicted curve's own
    crossing of 0.5 is marked and labelled with its distance from 4.26, because
    "crossover within 0.5 of 4.26" is literally the E2 Superhuman criterion.
    """
    if not (len(ratios) == len(predicted) == len(truth)):
        raise ValueError("ratios, predicted and truth must be the same length")
    if not ratios:
        raise ValueError(
            "calibration_curve_vs_truth was given no ratios; the figure's whole content "
            "is the gap between two curves, and there are none"
        )
    if predicted_err is not None and len(predicted_err) != len(ratios):
        raise ValueError(
            f"predicted_err has {len(predicted_err)} points for {len(ratios)} ratios"
        )
    if truth_err is not None and len(truth_err) != len(ratios):
        raise ValueError(f"truth_err has {len(truth_err)} points for {len(ratios)} ratios")
    if crossover is None:
        crossover = crossover_x(ratios, predicted)
    true_crossover = crossover_x(ratios, truth)
    max_gap = max(abs(p - t) for p, t in zip(predicted, truth))

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(6.8, 4.6))

        ax.axhline(0.5, color=GRID, ls="-", lw=1.0, zorder=1)
        ax.axvline(transition, color=REFERENCE, ls="--", lw=1.5, zorder=1)
        ax.annotate(
            f"phase transition {transition}",
            xy=(transition, 1.0), xytext=(4, -2), textcoords="offset points",
            rotation=90, ha="left", va="top", fontsize=8, color=INK_MUTED,
        )

        for i, zorder, label, values, err in (
            (1, 3, "true satisfiable fraction (solver)", truth, truth_err),
            (0, 4, "mean predicted P(SAT)", predicted, predicted_err),
        ):
            color, marker = _series_style(i)
            ax.plot(ratios, values, color=color, marker=marker, markersize=5.5,
                    markeredgecolor=SURFACE, markeredgewidth=1.0, label=label, zorder=zorder)
            if err is not None:
                ax.fill_between(
                    ratios, [v - e for v, e in zip(values, err)],
                    [v + e for v, e in zip(values, err)],
                    color=color, alpha=BAND_ALPHA, linewidth=0, zorder=2,
                )

        if crossover is not None:
            ax.plot([crossover], [0.5], marker="v", markersize=9, color=SERIES[0],
                    markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=6)
            ax.annotate(
                f"predicted crossover {crossover:.2f} ({crossover - transition:+.2f})",
                xy=(crossover, 0.5), xytext=(6, 14), textcoords="offset points",
                fontsize=8, color=INK,
                arrowprops={"arrowstyle": "-", "color": REFERENCE, "linewidth": 0.8},
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.02, 1.05)
        ax.set_title(title if condition is None else f"{title}\n{condition}")
        ax.legend(loc="lower left")

        if crossover is None:
            cross_txt = "predicted curve never crosses 0.5"
        else:
            cross_txt = f"predicted crossover {crossover:.2f} vs transition {transition}"
        if true_crossover is not None:
            cross_txt += f"; solver curve crosses at {true_crossover:.2f}"
        caption = (
            f"{_n_text(n, 'instances per ratio')}. {cross_txt}. "
            f"Largest gap to the true curve: {max_gap:.3f}."
        )
        if note:
            caption += f" {note}"
        fig.tight_layout(rect=(0, 0.045, 1, 1))
        return _finish(fig, _out(path), caption)


# --------------------------------------------------------------------------
# Self-check
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import math as _math
    import shutil
    import sys
    import tempfile

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="jevplots-"))
    keep = len(sys.argv) > 1

    def _png_bytes(p: Path) -> bytes:
        data = p.read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n", f"{p} is not a PNG"
        assert len(data) > 2000, f"{p} is implausibly small ({len(data)} bytes)"
        return data

    # A well calibrated condition, a badly calibrated one, and one whose only
    # reversal sits in a 2-sample bin -- the case the diagram has to make
    # visually obvious rather than hide.
    good = [
        {"lo": i / 10, "hi": (i + 1) / 10, "mean_predicted": i / 10 + 0.05,
         "observed": min(1.0, max(0.0, i / 10 + 0.05 + (0.02 if i % 2 else -0.02))),
         "count": 50}
        for i in range(10)
    ]
    overconfident = [
        {"lo": i / 10, "hi": (i + 1) / 10, "mean_predicted": i / 10 + 0.05,
         "observed": min(1.0, (i / 10 + 0.05) ** 1.8), "count": c}
        for i, c in enumerate([120, 90, 60, 40, 30, 25, 40, 60, 90, 145])
    ]
    thin = [
        {"lo": i / 10, "hi": (i + 1) / 10, "mean_predicted": i / 10 + 0.05,
         "observed": 0.0 if c == 0 else (0.95 if i == 4 else min(1.0, i / 10 + 0.05)),
         "count": c}
        for i, c in enumerate([200, 140, 60, 20, 2, 0, 3, 40, 130, 210])
    ]

    written: list[tuple[str, Path]] = []
    written.append(("reliability (well calibrated)", reliability_diagram(
        good, out / "reliability_good.png", "Reliability -- semantic control",
        condition="0 examples, structured state")))
    written.append(("reliability (overconfident)", reliability_diagram(
        overconfident, out / "reliability_overconfident.png", "Reliability -- 3SAT n=20",
        condition="m/n = 4.25")))
    written.append(("reliability (thin bins)", reliability_diagram(
        thin, out / "reliability_thin.png", "Reliability -- 3SAT n=50",
        condition="m/n = 8.0", note="Two bins hold fewer than 5 predictions.")))

    assert ece_from_bins(good) < 0.03, ece_from_bins(good)
    assert ece_from_bins(overconfident) > 0.05, ece_from_bins(overconfident)
    # The 2-sample reversal must not be reported as non-monotone.
    assert reliability_is_monotone(
        [b["observed"] for b in thin], [b["count"] for b in thin]
    )

    positions = [1, 5, 20, 50, 100, 200]
    acc = [0.91, 0.905, 0.90, 0.895, 0.88, 0.86]
    err = [1.96 * _math.sqrt(a * (1 - a) / 300) for a in acc]
    written.append(("accuracy vs position", curve(
        positions, acc, out / "e3_position.png",
        "E3 -- accuracy against question position", "position within the batch", "accuracy",
        n=300, yerr=err, baseline=0.5, baseline_label="majority class", logx=True,
        ylim=(0.4, 1.0), xticks=positions,
        note="Batch of 255 filler questions about the same state.")))

    counts = [1, 4, 16, 64, 128, 255]
    written.append(("latency vs question count", curve(
        counts, {"p50": [0.19, 0.20, 0.21, 0.24, 0.27, 0.33],
                 "p95": [0.28, 0.30, 0.33, 0.41, 0.52, 0.71]},
        out / "e3_latency.png", "E3 -- latency against question count",
        "questions per call", "wall clock (s)", n=300, logx=True,
        note="Marginal cost of question 200 is the claim under test.")))

    written.append(("accuracy vs encoding", curve(
        ["source", "AST JSON", "CFG edges"], [0.71, 0.78, 0.84],
        out / "e4_encoding.png", "E4 -- accuracy by encoding", "encoding",
        "accuracy", n=500, baseline=0.5, baseline_label="majority class", ylim=(0.4, 1.0))))

    ratios = [2.0 + 0.25 * i for i in range(25)]
    truth_curve = [1 / (1 + _math.exp((r - SAT_TRANSITION) * 4.0)) for r in ratios]
    predicted_curve = [0.5 + (t - 0.5) * 0.55 for t in truth_curve]
    written.append(("E2 predicted vs true", calibration_curve_vs_truth(
        ratios, predicted_curve, truth_curve, out / "e2_curve.png", n=500,
        condition="n = 20 variables",
        truth_err=[1.96 * _math.sqrt(t * (1 - t) / 500) for t in truth_curve],
        predicted_err=[0.01] * len(ratios),
        note="Predicted curve flattened, as the plan predicts.")))

    # A curve that never crosses 0.5 must still render, and say so.
    written.append(("E2 no crossover", calibration_curve_vs_truth(
        ratios, [0.62] * len(ratios), truth_curve, out / "e2_flat.png", n=500,
        condition="n = 50 variables")))

    assert crossover_x(ratios, predicted_curve) is not None
    assert crossover_x(ratios, [0.62] * len(ratios)) is None
    assert abs(crossover_x([0.0, 1.0], [1.0, 0.0]) - 0.5) < 1e-9
    # A crossing that lands on the last sampled point is still a crossing. The
    # sign-change test cannot see it -- there is no bracketing pair -- and
    # reporting "never crosses 0.5" there is the opposite of the finding.
    assert crossover_x([2.0, 3.0, 4.26], [0.9, 0.7, 0.5]) == 4.26
    assert crossover_x([2.0, 3.0], [0.2, 0.5]) == 3.0
    assert crossover_x([2.0, 3.0], [0.2, 0.4]) is None

    # Guards that should refuse rather than render something misleading.
    for bad_call, why in (
        (lambda: reliability_diagram([{"mean_predicted": 0.5, "observed": 0.5}],
                                     out / "no.png", "x"), "bin without a count"),
        (lambda: curve([1, 2], {f"s{i}": [0, 1] for i in range(9)}, out / "no.png",
                       "x", "x", "y", n=10), "more series than palette slots"),
        (lambda: curve([1, 2, 3], [0.1, 0.2], out / "no.png", "x", "x", "y", n=10),
         "length mismatch"),
        (lambda: curve([], [], out / "no.png", "x", "x", "y", n=10), "no x values"),
        # An interval keyed by a label no series uses must not be dropped quietly.
        (lambda: curve([1, 2], {"p50": [0.1, 0.2]}, out / "no.png", "x", "x", "y",
                       n=10, yerr=[0.01, 0.01]), "yerr keyed by the unused default label"),
        (lambda: curve([1, 2], {"p50": [0.1, 0.2]}, out / "no.png", "x", "x", "y",
                       n=10, band={"p95": ([0.0, 0.0], [1.0, 1.0])}), "band for an absent series"),
        (lambda: curve([1, 2], [0.1, 0.2], out / "no.png", "x", "x", "y",
                       n=10, yerr=[0.01]), "yerr shorter than x"),
        (lambda: calibration_curve_vs_truth([], [], [], out / "no.png", n=10),
         "no ratios"),
        (lambda: calibration_curve_vs_truth([1.0, 2.0], [0.4, 0.6], [0.3, 0.7],
                                            out / "no.png", n=10, truth_err=[0.01]),
         "truth_err shorter than the sweep"),
        (lambda: crossover_x([1.0, 2.0], [0.4]), "x and y of different lengths"),
    ):
        try:
            bad_call()
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected a refusal: {why}")

    # Determinism: the same call twice must produce the same bytes, so a
    # regenerated report diffs cleanly against the previous one.
    first = _png_bytes(out / "e2_curve.png")
    again = calibration_curve_vs_truth(
        ratios, predicted_curve, truth_curve, out / "e2_curve_again.png", n=500,
        condition="n = 20 variables",
        truth_err=[1.96 * _math.sqrt(t * (1 - t) / 500) for t in truth_curve],
        predicted_err=[0.01] * len(ratios),
        note="Predicted curve flattened, as the plan predicts.")
    assert _png_bytes(again) == first, "identical arguments produced different bytes"
    again.unlink()

    print(f"{len(written)} figures written to {out}\n")
    print(f"{'figure':<30} {'file':<32} {'bytes':>8}")
    for label, p in written:
        print(f"{label:<30} {p.name:<32} {len(_png_bytes(p)):>8}")
    if not keep:
        shutil.rmtree(out)
        print(f"\n(temporary directory removed; pass an output directory to keep the PNGs)")
