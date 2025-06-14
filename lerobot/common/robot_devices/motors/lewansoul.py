# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import enum
import logging
import math
import time
import traceback
from copy import deepcopy

import numpy as np
import tqdm

from lerobot.common.robot_devices.motors.configs import LewansoulMotorsBusConfig
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError
from lerobot.common.utils.utils import capture_timestamp_utc
from lerobot.common.robot_devices.motors.lx16a import LX16A

BAUDRATE = 115_200
TIMEOUT_MS = 0.2
MAX_ID_RANGE = 6

# The following bounds define the lower and upper joints range (after calibration).
# For joints in degree (i.e. revolute joints), their nominal range is [-180, 180] degrees
# which corresponds to a half rotation on the left and half rotation on the right.
# Some joints might require higher range, so we allow up to [-270, 270] degrees until
# an error is raised.
LOWER_BOUND_DEGREE = 0.0
UPPER_BOUND_DEGREE = 240.0

# See this link for LX-16A Command
# https://github.com/madhephaestus/lx16a-servo/blob/master/lx-16a%20LewanSoul%20Bus%20Servo%20Communication%20Protocol.pdf

LX16A_SERIES_BAUDRATE_TABLE = {
    0: 115_200,
}

CALIBRATION_REQUIRED = ["Goal_Position", "Present_Position"]
CONVERT_UINT32_TO_INT32_REQUIRED = ["Goal_Position", "Present_Position"]

MODEL_RESOLUTION = {
    "lx16a": 4096,
}

MODEL_BAUDRATE_TABLE = {
    "lx16a": LX16A_SERIES_BAUDRATE_TABLE,
}

# High number of retries is needed for feetech compared to dynamixel motors.
NUM_READ_RETRY = 20
NUM_WRITE_RETRY = 20

def convert_degrees_to_steps(degrees: float | np.ndarray, models: str | list[str]) -> np.ndarray:
    """This function converts the degree range to the step range for indicating motors rotation.
    It assumes a motor achieves a full rotation by going from -180 degree position to +180.
    The motor resolution (e.g. 4096) corresponds to the number of steps needed to achieve a full rotation.
    """
    resolutions = [MODEL_RESOLUTION[model] for model in models]
    steps = degrees / 180 * np.array(resolutions) / 2
    steps = steps.astype(int)
    return steps


def assert_same_address(model_ctrl_table, motor_models, data_name):
    all_addr = []
    all_bytes = []
    for model in motor_models:
        addr, bytes = model_ctrl_table[model][data_name]
        all_addr.append(addr)
        all_bytes.append(bytes)

    if len(set(all_addr)) != 1:
        raise NotImplementedError(
            f"At least two motor models use a different address for `data_name`='{data_name}' ({list(zip(motor_models, all_addr, strict=False))}). Contact a LeRobot maintainer."
        )

    if len(set(all_bytes)) != 1:
        raise NotImplementedError(
            f"At least two motor models use a different bytes representation for `data_name`='{data_name}' ({list(zip(motor_models, all_bytes, strict=False))}). Contact a LeRobot maintainer."
        )


class TorqueMode(enum.Enum):
    ENABLED = 1
    DISABLED = 0


class DriveMode(enum.Enum):
    NON_INVERTED = 0
    INVERTED = 1


class CalibrationMode(enum.Enum):
    # Joints with rotational motions are expressed in degrees in nominal range of [-180, 180]
    DEGREE = 0
    # Joints with linear motions (like gripper of Aloha) are expressed in nominal range of [0, 100]
    LINEAR = 1


class JointOutOfRangeError(Exception):
    def __init__(self, message="Joint is out of range"):
        self.message = message
        super().__init__(self.message)


class LewansoulMotorsBus:
    """
    The LewansoulMotorsBus class allows to efficiently read and write to the attached motors. 
    It relies on lx16a.py
    A LewansoulMotorsBus instance requires a port (e.g. `LewansoulMotorsBus(port="/dev/ttyUSB0`)).
    To find the port, you can run our utility script:
    ```bash
    python lerobot/scripts/find_motors_bus_port.py
    >>> Finding all available ports for the MotorsBus.
    >>> ['/dev/ttyUSB0', '/dev/ttyUSB1']
    >>> Remove the usb cable from your LewansoulMotorsBus and press Enter when done.
    >>> The port of this LewansoulMotorsBus is /dev/ttyUSB0
    >>> Reconnect the usb cable.
    ```

    Example of usage for 1 motor connected to the bus:
    ```python
    motor_name = "gripper"
    motor_index = 6
    motor_model = "lx16a"

    config = LewansoulMotorsBusConfig(
        port="/dev/ttyUSB0",
        motors={motor_name: (motor_index, motor_model)},
    )
    motors_bus = LewansoulMotorsBus(config)
    motors_bus.connect()

    position = motors_bus.read("Present_Position")

    # move from a few motor steps as an example
    few_steps = 30
    motors_bus.write("Goal_Position", position + few_steps)

    # when done, consider disconnecting
    motors_bus.disconnect()
    ```
    """

    def __init__(
        self,
        config: LewansoulMotorsBusConfig,
    ):
        self.port = config.port
        self.motors = config.motors

        self.calibration = None
        self.is_connected = False
        self.logs = {}
        self.track_positions = {}
        self.controller0 = None         #leader serial port
        self.controller1 = None         #follower serial port

    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                f"LewansoulMotorsBus({self.port}) is already connected. Do not call `motors_bus.connect()` twice."
            )

        try:
            print(
                "\nPort Init, made LX16A 6 classes.\n"
            )

            if(self.port == "/dev/ttyUSB1"):
                # serila port initialize
                self.controller1 = LX16A.initialize(self.port, TIMEOUT_MS)
                LX16A._controller = self.controller1
                self.servoFollow = {}
                for i in range(1, MAX_ID_RANGE+1):
                    self.servoFollow[i] = LX16A(id_=i, disable_torque=1)
                    time.sleep(0.2)           
            else:
                # serila port initialize
                self.controller0  = LX16A.initialize(self.port, TIMEOUT_MS)
                LX16A._controller = self.controller0
                self.servoLead = {}
                for i in range(1, MAX_ID_RANGE+1):
                    self.servoLead[i] = LX16A(id_=i, disable_torque=1)
                    time.sleep(0.2)                  

        except Exception:
            traceback.print_exc()
            print(
                "\nTry running `python lerobot/scripts/find_motors_bus_port.py` to make sure you are using the correct port.\n"
            )
            raise

        # Allow to read and write
        self.is_connected = True

    def are_motors_configured(self):
        # always true
        return CONVERT_UINT32_TO_INT32_REQUIRED

    def find_motor_indices(self, possible_ids=None, num_retry=2):
        #return motor id array
        indices = [1,2,3,4,5,6]
        return indices

    def set_bus_baudrate(self, baudrate):
        #no need to set baudrate, since it has 115200 only
        return True

    @property
    def motor_names(self) -> list[str]:
        return list(self.motors.keys())

    @property
    def motor_models(self) -> list[str]:
        return [model for _, model in self.motors.values()]

    @property
    def motor_indices(self) -> list[int]:
        return [idx for idx, _ in self.motors.values()]

    def set_calibration(self, calibration: dict[str, list]):
        self.calibration = calibration

    def apply_calibration_autocorrect(self, values: np.ndarray | list, motor_names: list[str] | None):
        """This function apply the calibration, automatically detects out of range errors for motors values and attempt to correct.

        For more info, see docstring of `apply_calibration` and `autocorrect_calibration`.
        """
        return True

    def apply_calibration(self, values: np.ndarray | list, motor_names: list[str] | None):
        """Convert from unsigned int32 joint position range [0, 2**32[ to the universal float32 nominal degree range ]-180.0, 180.0[ with
        a "zero position" at 0 degree.

        Note: We say "nominal degree range" since the motors can take values outside this range. For instance, 190 degrees, if the motor
        rotate more than a half a turn from the zero position. However, most motors can't rotate more than 180 degrees and will stay in this range.

        Joints values are original in [0, 2**32[ (unsigned int32). Each motor are expected to complete a full rotation
        when given a goal position that is + or - their resolution. For instance, feetech xl330-m077 have a resolution of 4096, and
        at any position in their original range, let's say the position 56734, they complete a full rotation clockwise by moving to 60830,
        or anticlockwise by moving to 52638. The position in the original range is arbitrary and might change a lot between each motor.
        To harmonize between motors of the same model, different robots, or even models of different brands, we propose to work
        in the centered nominal degree range ]-180, 180[.
        """
        return True

    def autocorrect_calibration(self, values: np.ndarray | list, motor_names: list[str] | None):
        """This function automatically detects issues with values of motors after calibration, and correct for these issues.

        Some motors might have values outside of expected maximum bounds after calibration.
        For instance, for a joint in degree, its value can be outside [-270, 270] degrees, which is totally unexpected given
        a nominal range of [-180, 180] degrees, which represents half a turn to the left or right starting from zero position.

        Known issues:
        #1: Motor value randomly shifts of a full turn, caused by hardware/connection errors.
        #2: Motor internal homing offset is shifted of a full turn, caused by using default calibration (e.g Aloha).
        #3: motor internal homing offset is shifted of less or more than a full turn, caused by using default calibration
            or by human error during manual calibration.

        Issues #1 and #2 can be solved by shifting the calibration homing offset by a full turn.
        Issue #3 will be visually detected by user and potentially captured by the safety feature `max_relative_target`,
        that will slow down the motor, raise an error asking to recalibrate. Manual recalibrating will solve the issue.

        Note: A full turn corresponds to 360 degrees but also to 4096 steps for a motor resolution of 4096.
        """
        return True

    def revert_calibration(self, values: np.ndarray | list, motor_names: list[str] | None):
        """Inverse of `apply_calibration`."""
        return True

    def avoid_rotation_reset(self, values, motor_names, data_name):
        return True


    def read(self, data_name, motor_names: str | list[str] | None = None):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"LewansouldMotorsBus({self.port}) is not connected. You need to run `motors_bus.connect()`."
            )

        if motor_names is None:
            motor_names = self.motor_names

        if isinstance(motor_names, str):
            motor_names = [motor_names]

        motor_ids = []
        models = []
        for name in motor_names:
            motor_idx, model = self.motors[name]
            motor_ids.append(motor_idx)
            models.append(model)

        values = []

        for idx in motor_ids:
            if data_name == "Present_Position": 
                if(self.port == "/dev/ttyUSB0"):
                    LX16A._controller = self.controller0
                    value = self.servoLead[idx].get_physical_angle()
                else:
                    LX16A._controller = self.controller1
                    value = self.servoFollow[idx].get_physical_angle()
                #print('port, ReadAngle:', self.port, value)
                #Limit here, for wrong value
                if value < LOWER_BOUND_DEGREE:
                    value = LOWER_BOUND_DEGREE
                elif value > UPPER_BOUND_DEGREE:
                    value = UPPER_BOUND_DEGREE

            values.append(value)

        print(values)
        values = np.array(values, dtype=np.float32)
        return values
    
    def write(self, data_name, values: int | float | np.ndarray, motor_names: str | list[str] | None = None):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"LewansoulMotorsBus({self.port}) is not connected. You need to run `motors_bus.connect()`."
            )
        
        if motor_names is None:
            motor_names = self.motor_names

        if isinstance(motor_names, str):
            motor_names = [motor_names]

        if isinstance(values, (int, float, np.integer)):
            values = [int(values)] * len(motor_names)

        values = np.array(values)

        motor_ids = []
        models = []
        for name in motor_names:
            motor_idx, model = self.motors[name]
            motor_ids.append(motor_idx)
            models.append(model)

        values = values.tolist()

        for idx, value in zip(motor_ids, values, strict=True):
            if data_name == "Goal_Position": 
                #print('port, WriteAngle:', self.port, value)
                if(self.port == "/dev/ttyUSB0"):
                    LX16A._controller = self.controller0
                    value = self.servoLead[idx].move(angle=value, time=100)
                else:
                    LX16A._controller = self.controller1
                    value = self.servoFollow[idx].move(angle=value, time=100)
            elif data_name == "Torque_Enable":
                #print('port, Torque:', self.port, value)
                if(self.port == "/dev/ttyUSB0"):
                    LX16A._controller = self.controller0
                    value = self.servoLead[idx].set_torque(torque=value)
                else:
                    LX16A._controller = self.controller1
                    value = self.servoFollow[idx].set_torque(torque=value)

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"LewansoulMotorsBus({self.port}) is not connected. Try running `motors_bus.connect()` first."
            )

        LX16A.deinitialize
        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
