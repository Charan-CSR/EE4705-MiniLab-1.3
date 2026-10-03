# Contributor: Marie Chansuphan (Task 4).
"""Pure controller checks; no YOLO, simulator, model weights, or network needed."""
import importlib.util
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
spec = importlib.util.spec_from_file_location('task4_navigation_test', Path(__file__).resolve().parents[1] / 'src/perception/goto_object.py')
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def observation(t, box=None, color='red'):
    detections = [] if box is None else [SimpleNamespace(
        class_name='cup', color=color, confidence=0.8, bbox=box)]
    return module.Observation(t, 640, 480, detections)


class NavigationTests(unittest.TestCase):
    def start(self, align=False):
        nav = module.VisualNavigator(align_only=align)
        nav.start('cup', 'red', 0, 0)
        return nav

    def test_left_right_sign_and_no_forward_when_unaligned(self):
        nav = self.start()
        self.assertGreater(nav.step(.1, 0, observation(.1, (10, 100, 90, 160)))[2], 0)
        cmd = nav.step(.2, 0, observation(.2, (510, 100, 590, 160)))
        self.assertLess(cmd[2], 0)
        self.assertEqual(cmd[0], 0)

    def test_alignment_requires_three_distinct_frames(self):
        nav = self.start(True)
        frame = observation(.1, (290, 100, 350, 160))
        for _ in range(5):
            nav.step(.1, 0, frame)
        self.assertTrue(nav.active)
        for t in (.2, .3):
            self.assertEqual(nav.step(t, 0, observation(t, (290, 100, 350, 160))), (0, 0, 0))
        self.assertEqual(nav.outcome, 'TEST_COMPLETE')

    def test_wrong_color_searches_and_full_turn_fails(self):
        nav = self.start()
        nav.step(.1, 0, observation(.1, (290, 100, 350, 160), 'white'))
        self.assertGreater(nav.step(.7, 0, observation(.7))[2], 0)
        for i in range(1, 66):
            yaw = (i*.1 + math.pi) % (2*math.pi) - math.pi
            nav.step(.7+i*.1, yaw, observation(.7+i*.1))
        self.assertEqual(nav.outcome, 'FAIL')
        self.assertFalse(nav.active)

    def test_stale_frame_immediately_zeros_motion(self):
        nav = self.start()
        frame = observation(.1, (10, 100, 90, 160))
        nav.step(.1, 0, frame)
        self.assertEqual(nav.step(1.0, .1, frame), (0, 0, 0))

    def test_timeout(self):
        nav = self.start()
        nav.step(60.1, 0, observation(60.1))
        self.assertEqual(nav.outcome, 'FAIL')

    def test_visual_stop_requires_new_frame_and_does_not_claim_success(self):
        nav = self.start()
        for t in (.1, .2, .3):
            nav.step(t, 0, observation(t, (290, 100, 350, 270)))
        self.assertEqual(nav.state, 'VERIFY')
        self.assertIsNone(nav.verification)
        nav.step(.5, 0, observation(.5, (290, 100, 350, 270)))
        self.assertIsNone(nav.verification)
        nav.step(.9, 0, observation(.9, (290, 100, 350, 270)))
        self.assertIsNotNone(nav.verification)
        self.assertTrue(nav.active)
        self.assertIsNone(nav.outcome)
        self.assertEqual(nav.command, (0, 0, 0))

    def test_target_loss_stops_forward_immediately(self):
        nav = self.start()
        for t in (.1, .2, .3):
            nav.step(t, 0, observation(t, (290, 100, 350, 150)))
        self.assertGreater(nav.command[0], 0)
        self.assertEqual(nav.step(.4, 0, observation(.4)), (0, 0, 0))

if __name__ == '__main__':
    unittest.main()
