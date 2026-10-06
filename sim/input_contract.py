"""Read controller capabilities from the live environment, without scene truth."""
def input_capabilities(env, backend):
    node = env
    seen = set()
    while id(node) not in seen:
        seen.add(id(node))
        nxt = getattr(node, '_env', None)
        if nxt is None:
            nxt = getattr(node, 'unwrapped', node)
        if nxt is node:
            break
        node = nxt
    result = {'backend': backend}
    if backend == 'maniskill':
        result['control_mode'] = node.control_mode
        result['robot'] = node.agent.uid
        controllers = node.agent.controller.controllers
        arm = controllers['arm'].config
        result['translation_frame'] = arm.frame.split(':')[0]
        result['position_lower'] = arm.pos_lower
        result['position_upper'] = arm.pos_upper
        result['rotation_frame'] = arm.frame.split(':')[-1]
        result['has_gripper'] = 'gripper' in controllers
    elif backend == 'metaworld':
        result.update(control_mode='mocap_delta', position_scale_m=float(node.action_scale))
    elif backend == 'robocasa':
        robot = node.robots[0]
        composite = robot.composite_controller
        result['control_mode'] = composite.name
        result['action_slices'] = {k:list(v) for k,v in composite._action_split_indexes.items()}
        result['parts'] = {name:{'controller':type(ctrl).__name__,
                                'input_ref_frame':getattr(ctrl,'input_ref_frame','unknown'),
                                'input_type':getattr(ctrl,'input_type','unknown')}
                           for name,ctrl in composite.part_controllers.items()}
    elif backend == 'libero':
        result['control_mode'] = 'OSC_POSE'  # additionally validated against openeta_control_spec
    return result
