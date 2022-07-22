package(
    default_visibility = ["//tensorflow:internal"],
)

cc_import(
    name = "libpta",
    hdrs = glob([
        "tuning_ops/host/**/*.h",
    ]),
    shared_library = "tuning_ops/host/libpta.so",
    visibility = ["//visibility:public"],
)

cc_library(
    name="host_proxy",
    hdrs = glob([
        "tuning_ops/host_proxy/*.h",
    ]),
    srcs = glob([
        "tuning_ops/host_proxy/*.cpp",
    ]),
    copts = ["-fexceptions"],
    nocopts = "-fno-exceptions",
    includes = [
        "tuning_ops/host/include",
        "tuning_ops/host/include/DataTypes",
        "tuning_ops/host/suite/Ackley",
        "tuning_ops/host/suite/ExpSinFunc",
        "tuning_ops/host/suite/NetworkAI",
    ],
    deps = [
        ":libpta",
    ]
)