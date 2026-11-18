from dataclasses import dataclass
from itertools import product

from bundle.request import Window


@dataclass(frozen=True)
class SlotRange:
    """Half-open coefficient-slot interval inside one low/high domain of an axis."""

    start: int
    stop: int
    domain: int

    def __iter__(self):
        return iter(range(self.start, self.stop))

    @property
    def empty(self):
        return self.start >= self.stop


def _aligned_extent(start, stop, axis):
    first_pair = min(start // 2, max(0, axis.high - 1))
    last_pair, odd = divmod(stop - 1, 2)
    return 2 * first_pair, min(axis.length, 2 * (last_pair + 1) + odd)


def band_window(pixels, band):
    horizontal, vertical = band
    return Window(
        pixels.top // 2, pixels.left // 2,
        (pixels.bottom + (vertical == "L")) // 2,
        (pixels.right + (horizontal == "L")) // 2,
    )


def projected_windows(layout, window):
    for layer in layout.layers:
        top, bottom = _aligned_extent(window.top, window.bottom, layer.rows)
        left, right = _aligned_extent(window.left, window.right, layer.columns)
        pixels = Window(top, left, bottom, right)
        yield layer, pixels
        window = band_window(pixels, "LL")


def axis_slot_ranges(pixels_start, pixels_stop, axis):
    """Coefficient slots required to inverse one axis for an aligned pixel extent.

    The even (low-frequency) stage reads high neighbors clamped on the whole
    high domain, and the odd (high-frequency) stage reads its following even
    neighbor clamped on the whole low domain.  The returned ranges widen the
    band-window projection with exactly those neighbor slots, clipped to the
    axis domains.  Value-independent in shape, they let the offline package
    alone reconstruct the same values as the full-array crop regardless of how
    the coefficients were split into original blocks.
    """
    low_start = pixels_start // 2
    # The last odd pixel also reads its following even neighbor, so the low
    # domain reaches one slot beyond the aligned extent's own even slots; when
    # the axis ends on an even pixel that neighbor clamps back onto its boundary.
    low_stop = min(axis.low, (pixels_stop + 2) // 2)
    # Even pixels read high neighbors clamp(i-1) and clamp(i), so the high
    # domain starts one slot earlier and ends on the last even slot.
    high_start = max(0, low_start - 1)
    high_stop = min(axis.high, low_stop)
    return SlotRange(low_start, low_stop, axis.low), SlotRange(high_start, high_stop, axis.high)


def band_region(pixels, band, layer):
    """Band-local coefficient rectangle, including whole-axis neighbor slots."""
    horizontal, vertical = band
    row_low, row_high = axis_slot_ranges(pixels.top, pixels.bottom, layer.rows)
    column_low, column_high = axis_slot_ranges(pixels.left, pixels.right, layer.columns)
    rows = row_low if vertical == "L" else row_high
    columns = column_low if horizontal == "L" else column_high
    return rows, columns


def plan_block_keys(revision, window, index):
    keys = set()
    for layer, pixels in projected_windows(revision.layout, window):
        for band in revision.layout.stored_bands(layer.number):
            rows, columns = band_region(pixels, band.name, layer)
            for row, column in product(rows, columns):
                ref = index.locate(revision.key, layer.number, band.name, row, column)
                keys.add(ref.key)
    return tuple(sorted(keys))
