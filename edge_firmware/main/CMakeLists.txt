idf_component_register(
    SRCS
        "app_main.c"
        "task_camera.c"
        "task_sensors.c"
        "task_vision.c"
        "task_policy.c"
        "task_telemetry.c"
        "edge_rl_policy.c"
        "edge_rl_vision.c"
        "edge_rl_dht22.c"
    INCLUDE_DIRS "."
    REQUIRES
        freertos
        esp_timer
        driver
        nvs_flash
)
