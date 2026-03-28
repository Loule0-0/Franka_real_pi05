"""Configs for single-arm Franka teleop datasets collected with the local teleop stack."""

import openpi.models.pi0_config as pi0_config
import openpi.policies.droid_policy as droid_policy
import openpi.transforms as _transforms


REPO_ID = "toy/franka_singlearm_teleop"
BASE_CHECKPOINT = "/home/server/Desktop/vla/model/pi05_base/params"


def get_franka_teleop_configs():
    # Import here to avoid circular imports.
    from openpi.training import weight_loaders
    from openpi.training.config import DataConfig
    from openpi.training.config import SimpleDataConfig
    from openpi.training.config import TrainConfig

    repack = _transforms.Group(
        inputs=[
            _transforms.RepackTransform(
                {
                    "observation/exterior_image_1_left": "exterior_image_1_left",
                    "observation/wrist_image_left": "wrist_image_left",
                    "observation/joint_position": "joint_position",
                    "observation/gripper_position": "gripper_position",
                    "actions": "actions",
                    "prompt": "prompt",
                }
            )
        ]
    )

    data_transforms = lambda model: _transforms.Group(
        inputs=[droid_policy.DroidInputs(model_type=model.model_type)],
        outputs=[droid_policy.DroidOutputs()],
    )

    return [
        TrainConfig(
            name="pi05_franka_singlearm_teleop",
            model=pi0_config.Pi0Config(pi05=True, action_horizon=15),
            data=SimpleDataConfig(
                repo_id=REPO_ID,
                base_config=DataConfig(
                    prompt_from_task=True,
                    repack_transforms=repack,
                ),
                data_transforms=data_transforms,
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(BASE_CHECKPOINT),
            batch_size=32,
            num_train_steps=20_000,
        ),
    ]
