"""Regression coverage for the A017 scratch-review reconstruction incident.

These checks pin the two guarantees that the original window publisher and
offline reader violated together:

* the package must contain every original block reached through the whole-axis
  neighbor dependencies of the aligned window tile, not only the blocks that
  geometrically cover the projected band window;
* offline reconstruction clamps neighbor slots against the whole axis, so a
  single-pixel crop equals the same crop taken from a full-array reading and
  does not depend on how coefficients were split into original blocks.
"""

from bundle.request import Window
from catalog.registration import register
from replay.open import open_package
from replay.window import read_window
from tests.fixtures import (
    LINE, LINE_KEYS, POINT, StationCase, line_submission, original_positions,
    tiled_submission,
)


class IncidentRegressionTests(StationCase):
    def test_single_pixel_window_equals_full_array_crop(self):
        key = self.register_sample()
        point_id = self.publish(key)
        full_id = self.publish(key, LINE)
        point = read_window(self.packages, point_id)
        full = read_window(self.packages, full_id)
        self.assertEqual(point.origin, (0, 3))
        self.assertEqual(point.values, ((28,),))
        self.assertEqual(point.values, (full.values[0][3:4],))

    def test_manifest_holds_exactly_the_whole_axis_dependencies(self):
        key = self.register_sample()
        manifest = open_package(self.packages, self.publish(key)).manifest
        self.assertEqual(original_positions(manifest), LINE_KEYS)

    def test_dependencies_are_independent_of_coefficient_values(self):
        original = self.register_sample()
        changed = register(self.db, *line_submission('changed', high=(12, 4)))
        original_id = self.publish(original)
        changed_id = self.publish(changed)
        # Different payloads, identical block identities and package shape.
        self.assertEqual(
            original_positions(open_package(self.packages, original_id).manifest),
            original_positions(open_package(self.packages, changed_id).manifest),
        )
        self.assertEqual(read_window(self.packages, original_id).values, ((28,),))
        self.assertEqual(read_window(self.packages, changed_id).values, ((26,),))

    def test_repartitioning_blocks_keeps_the_reading(self):
        fine = self.register_sample()
        tiled = register(self.db, *line_submission('tiled', tile=2))
        fine_id = self.publish(fine)
        tiled_id = self.publish(tiled)
        self.assertNotEqual(
            original_positions(open_package(self.packages, fine_id).manifest),
            original_positions(open_package(self.packages, tiled_id).manifest),
        )
        self.assertEqual(
            read_window(self.packages, fine_id), read_window(self.packages, tiled_id),
        )

    def test_singleton_axis_keeps_low_frequency_value(self):
        from catalog.revisions import RevisionKey

        key = RevisionKey('singleton', 'r1')
        register(self.db, *tiled_submission(key, 1, 1, 1, {(1, 'LL'): ((-7,),)}))
        package_id = self.publish(key, Window(0, 0, 1, 1))
        self.assertEqual(read_window(self.packages, package_id).values, ((-7,),))
