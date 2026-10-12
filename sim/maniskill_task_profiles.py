"""Native defaults for selected tasks whose interfaces differ from Panda grasping."""

TASK_DEFAULTS = {
    'DrawTriangle-v1': {'control_mode': 'pd_ee_delta_pose', 'max_episode_steps': 300},
    'DrawSVG-v1': {'control_mode': 'pd_ee_delta_pose', 'max_episode_steps': 500},
    'RotateSingleObjectInHandLevel3-v1': {'control_mode': 'pd_joint_pos', 'max_episode_steps': 300},
    'InsertFlower-v1': {'control_mode': 'pd_joint_pos', 'max_episode_steps': 300},
}
