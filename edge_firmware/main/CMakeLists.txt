idf_component_register(
    SRCS
        "app_main.c"
        "task_camera.c"
        "task_sensors.c"
        "task_vision.c"
        "task_policy.c"
        "task_telemetry.c"
    INCLUDE_DIRS "."
    REQUIRES
        freertos
        esp_timer
        esp_camera
        driver
        edge_rl_vision
        edge_rl_policy
        edge_rl_sensors
)
