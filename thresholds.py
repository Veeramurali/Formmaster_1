# Get thresholds for beginner mode
def get_thresholds_beginner():
    _ANGLE_HIP_KNEE_VERT = {
        'NORMAL': (0, 32),
        'TRANS': (35, 65),
        'PASS': (70, 95)
    }

    _ANGLE_ELBOW_BICEP_CURL = {
        'NORMAL': (30, 150),  # Normal range for elbow angle during bicep curl
        'TRANS': (15, 29),    # Transition range (too low)
        'PASS': (151, 180)    # Pass range (too high)
    }

    thresholds = {
        'HIP_KNEE_VERT': _ANGLE_HIP_KNEE_VERT,
        'ELBOW_BICEP_CURL': _ANGLE_ELBOW_BICEP_CURL,  # Added bicep curl thresholds

        'HIP_THRESH': [10, 50],
        'ANKLE_THRESH': 45,
        'KNEE_THRESH': [50, 70, 95],

        'OFFSET_THRESH': 35.0,
        'INACTIVE_THRESH': 15.0,

        'CNT_FRAME_THRESH': 50
    }

    return thresholds


# Get thresholds for pro mode
def get_thresholds_pro():
    _ANGLE_HIP_KNEE_VERT = {
        'NORMAL': (0, 32),
        'TRANS': (35, 65),
        'PASS': (80, 95)
    }

    _ANGLE_ELBOW_BICEP_CURL = {
        'NORMAL': (30, 150),  # Normal range for elbow angle during bicep curl
        'TRANS': (15, 29),    # Transition range (too low)
        'PASS': (151, 180)    # Pass range (too high)
    }

    thresholds = {
        'HIP_KNEE_VERT': _ANGLE_HIP_KNEE_VERT,
        'ELBOW_BICEP_CURL': _ANGLE_ELBOW_BICEP_CURL,  # Added bicep curl thresholds

        'HIP_THRESH': [15, 50],
        'ANKLE_THRESH': 30,
        'KNEE_THRESH': [50, 80, 95],

        'OFFSET_THRESH': 35.0,
        'INACTIVE_THRESH': 15.0,

        'CNT_FRAME_THRESH': 50
    }

    return thresholds