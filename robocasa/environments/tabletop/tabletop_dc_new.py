import numpy as np
from typing import Tuple
from robocasa.environments.tabletop.tabletop_dc import (
    PositionSampler as PositionSamplerOld,
)
from robocasa.environments.tabletop.tabletop_dc import generate_task_classes

# generate tasks with a specific object category as target object
obj_categories = ["can", "apple", "cucumber", "lemon", "bottled_water"]
container_combos = [
    ("plate", "bowl"),
    ("placemat", "basket"),
    ("cutting_board", "basket"),
    ("tray", "plate"),
    ("cutting_board", "pan"),
]
new_tasks_info = generate_task_classes(
    obj_categories,
    container_combos,
    prefix="PnP5",
    randomize_distractor_configs=False,  # no distractor
    distractor_configs=None,
    postfix="NoDistractorSplitA",
)

distractor_configs = {
    ("cutting_board", "pan"): ["distractor_obj"],
}

new_tasks_info.extend(
    generate_task_classes(
        obj_categories,
        [("cutting_board", "pan")],
        prefix="PnP5",
        randomize_distractor_configs=False,
        distractor_configs=distractor_configs,
        postfix="DistractorSplitA",
    )
)

new_tasks_info.append(
    generate_task_classes(
        ["apple"],
        [("cutting_board", "pan")],
        prefix="PnPApple",
        postfix="DistractorSplitA",
        distractor_obj_cats=obj_categories,
        randomize_distractor_configs=False,
        distractor_configs=distractor_configs,
    )
)

new_tasks_info.append(
    generate_task_classes(
        ["apple", "can"],
        [("cutting_board", "pan")],
        prefix="PnPAppleOrCan",
        postfix="DistractorAppleOrCanSplitA",
        randomize_distractor_configs=False,
        distractor_configs=distractor_configs,
    )
)

if __name__ == "__main__":

    with open("task_list/new_task_info.csv", "w") as f:
        f.write("task_name,source_container,target_container,obj_cat,distractor_keys\n")
        for task_info in new_tasks_info:
            f.write(
                f"{task_info['class_name']},{task_info['source_container']},{task_info['target_container']},"
            )
            f.write(f"{' '.join(task_info['obj_cats'])},")
            if task_info["distractor_config"]:
                f.write(f"{' '.join(task_info['distractor_config'])}\n")
            elif task_info["randomize_distractor_configs"]:
                f.write("random\n")
            else:
                f.write("none\n")

    with open("task_list/new_tasks.txt", "w") as f:
        for task_info in new_tasks_info:
            f.write(f"{task_info['class_name']}\n")

    # print all tasks that generated successfully
    # print(base_obj_cats)
    # print(novel_obj_cats)
    print(f"Total number of task classes: {len(new_tasks_info)}")
