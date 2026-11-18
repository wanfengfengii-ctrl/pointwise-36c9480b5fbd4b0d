from dataclasses import dataclass

from bundle.dependencies import SlotRange, band_region, projected_windows
from catalog.layout import clamp_index
from replay.coordinates import WindowReading
from replay.integer_steps import restore_even, restore_odd
from replay.open import OpenedPackage, open_package


@dataclass(frozen=True)
class SlottedMatrix:
    """Row-major matrix whose local index 0 maps to global slot ``offset``."""

    offset: tuple[int, int]
    values: tuple[tuple[int, ...], ...]

    def crop(self, rows: SlotRange, columns: SlotRange):
        top = rows.start - self.offset[0]
        left = columns.start - self.offset[1]
        values = tuple(
            row[left:left + (columns.stop - columns.start)]
            for row in self.values[top:top + (rows.stop - rows.start)]
        )
        return SlottedMatrix((rows.start, columns.start), values)


def inverse_axis(low, high, low_offset, high_offset, axis):
    """Inverse one axis, clamping neighbor slots on the whole axis domains.

    ``low``/``high`` are the coefficient sequences available to this tile;
    their first elements live at global slots ``low_offset``/``high_offset``.
    The even stage reads high neighbors ``clamp(i-1)`` and ``clamp(i)`` on the
    whole high domain; the odd stage starts from its own high coefficient and
    reads the following even ``clamp(i+1)`` on the whole low domain.  Because
    every neighbor is clamped against the full axis rather than the cropped
    tile, changing how coefficients are split into original blocks never
    changes the reconstructed reading.  A single-pixel axis keeps its
    low-frequency value.
    """
    if axis.length == 1:
        return tuple(low)
    # Even stage first: every odd pixel reads two reconstructed even neighbors.
    evens = []
    for local, low_value in enumerate(low):
        slot = low_offset + local
        previous_high = high[clamp_index(slot - 1, axis.high) - high_offset]
        current_high = high[clamp_index(slot, axis.high) - high_offset]
        evens.append(restore_even(low_value, previous_high, current_high))
    evens = tuple(evens)
    pixels = []
    for local, even in enumerate(evens):
        slot = low_offset + local
        pixels.append(even)
        if slot < axis.high:
            following = clamp_index(slot + 1, axis.low)
            # The planner extends the low domain by one slot; when the axis
            # ends on an even pixel the neighbor clamps back onto the boundary
            # even already carried as the tile's last reconstructed even.
            following_even = evens[min(following - low_offset, len(evens) - 1)]
            pixels.append(restore_odd(high[slot - high_offset], even, following_even))
    return tuple(pixels)


def _read_band(package, layer, band, rows: SlotRange, columns: SlotRange):
    values = tuple(
        tuple(package.blocks.coefficient_at(layer, band, row, column) for column in columns)
        for row in rows
    )
    return SlottedMatrix((rows.start, columns.start), values)


def _inverse_columns(low_matrix, high_matrix, axis):
    """Vertical inverse applied column by column (vertical axis first)."""
    high_rows = high_matrix.values if high_matrix is not None else ()
    high_offset = high_matrix.offset[0] if high_matrix is not None else 0
    columns = []
    for local_column in range(len(low_matrix.values[0])):
        low = tuple(row[local_column] for row in low_matrix.values)
        high = tuple(row[local_column] for row in high_rows) if high_rows else ()
        columns.append(inverse_axis(low, high, low_matrix.offset[0], high_offset, axis))
    width = len(columns[0]) if columns else 0
    rows = tuple(tuple(column[row] for column in columns) for row in range(width))
    return rows


def _inverse_rows(low_rows, high_rows, low_offset, high_offset, axis):
    """Horizontal inverse applied row by row."""
    rows = []
    for local_row in range(len(low_rows)):
        high = high_rows[local_row] if high_rows else ()
        rows.append(inverse_axis(low_rows[local_row], high, low_offset, high_offset, axis))
    return tuple(rows)


def _inverse_tile(band_matrices, layer, pixels):
    # Vertical pass first: LL/LH carry vertical-low rows, HL/HH the high rows.
    vertical_low_rows = _inverse_columns(band_matrices["LL"], band_matrices.get("LH"), layer.rows)
    # A singleton horizontal axis has no horizontal-high bands at all.
    vertical_high = band_matrices.get("HL")
    if vertical_high is None:
        vertical_high_rows = ()
    else:
        vertical_high_rows = _inverse_columns(vertical_high, band_matrices.get("HH"), layer.rows)
    # Horizontal pass: the vertical-low result feeds the even columns.
    values = _inverse_rows(
        vertical_low_rows, vertical_high_rows,
        band_matrices["LL"].offset[1],
        vertical_high.offset[1] if vertical_high is not None else 0,
        layer.columns,
    )
    return SlottedMatrix((pixels.top, pixels.left), values)


def _crop_to_window(matrix: SlottedMatrix, window):
    row_start = window.top - matrix.offset[0]
    column_start = window.left - matrix.offset[1]
    return tuple(
        row[column_start:column_start + (window.right - window.left)]
        for row in matrix.values[row_start:row_start + (window.bottom - window.top)]
    )


def replay_window(package: OpenedPackage):
    layout, window = package.manifest.layout, package.manifest.window
    previous = None
    for layer, pixels in reversed(tuple(projected_windows(layout, window))):
        band_matrices = {}
        for band in layout.stored_bands(layer.number):
            rows, columns = band_region(pixels, band.name, layer)
            band_matrices[band.name] = _read_band(package, layer.number, band.name, rows, columns)
        if previous is not None:
            _, coarse_matrix = previous
            low_rows, low_columns = band_region(pixels, "LL", layer)
            band_matrices["LL"] = coarse_matrix.crop(low_rows, low_columns)
        matrix = _inverse_tile(band_matrices, layer, pixels)
        previous = pixels, matrix
    _, fine_matrix = previous
    return WindowReading((window.top, window.left), _crop_to_window(fine_matrix, window))


def read_window(directory, package_id):
    return replay_window(open_package(directory, package_id))
