from core.interfaces import ObjectDetector
from core.interfaces import ArmController
import json


class DetectTester:
    def __init__(self):
        """Initialize the object detection tester."""
        print("\nInitializing Object Detection Tester...")
        self.detector = ObjectDetector()
    
    def test_detect(self, image):
        """Test the object detection algorithm."""
        print("\nTesting object detection algorithm...")
        
        # Detect objects in the image
        objects = self.detector.detect(image)
        
        return objects


class Controller:
    def __init__(self) -> None:
        pass

    def confirm_pick(self):
        gripper_result = False
        pos, force = ArmController().get_gripper_state()
        print(json.dumps(pos, indent=3))

        print(json.dumps(force, indent=3))
        
        # TODO: Implement gripper confirm logic
        return gripper_result


def main():
    # TODO: Implement the pick tester
    pass

if __name__ == "__main__":
    main()
