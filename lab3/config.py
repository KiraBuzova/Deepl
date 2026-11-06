import math

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
IMG_SIZE = 160
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

SPLIT_SEED = 0
N_TRAIN_PER_CLASS = 200
SPLIT_CSV = "split.csv"

EXPECTED_SPLIT = {
    ("buildings", "train"): (200, 1996761), ("buildings", "validation"): (218, 4837323), ("buildings", "test"): (219, 4837559),
    ("forest", "train"): (200, 2032625), ("forest", "validation"): (237, 5276962), ("forest", "test"): (237, 5291326),
    ("glacier", "train"): (200, 2183923), ("glacier", "validation"): (276, 6135652), ("glacier", "test"): (277, 6157250),
    ("mountain", "train"): (200, 1950345), ("mountain", "validation"): (262, 5789535), ("mountain", "test"): (263, 5849765),
    ("sea", "train"): (200, 1987613), ("sea", "validation"): (255, 5655068), ("sea", "test"): (255, 5668011),
    ("street", "train"): (200, 1841817), ("street", "validation"): (250, 5541845), ("street", "test"): (251, 5547533),
}
EXPECTED_TOTALS = {"train": (1200, 11993084), "validation": (1498, 33236385), "test": (1502, 33351444)}
EXPECTED_RAW_COUNTS = {"train": 14034, "test": 3000}

BATCH_TRAIN = 32
BATCH_EVAL = 64
EPOCHS = 8
WEIGHT_DECAY = 1e-4
LR_HEAD = 3e-3
LR_LAYER4 = 3e-4
BN_MOMENTUM = 0.1
BN_EPS = 1e-5
LN6 = math.log(6)

REGIMES = ("scratch", "frozen", "unfreeze")
RUNS = [("scratch", 1), ("frozen", 1), ("frozen", 2), ("frozen", 3),
        ("unfreeze", 1), ("unfreeze", 2), ("unfreeze", 3)]

TOP_MODULES = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"]
EVAL_MODULES = {
    "scratch": [],
    "frozen": ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"],
    "unfreeze": ["conv1", "bn1", "layer1", "layer2", "layer3"],
}

TOTAL_PARAMS = 11_179_590
FC_PARAMS = 3_078
LAYER4_PARAMS = 8_393_728
EXPECTED_PARAMS = {
    "scratch": {"trainable": TOTAL_PARAMS, "groups": [("all", TOTAL_PARAMS, LR_HEAD)]},
    "frozen": {"trainable": FC_PARAMS, "groups": [("fc", FC_PARAMS, LR_HEAD)]},
    "unfreeze": {"trainable": FC_PARAMS + LAYER4_PARAMS,
                 "groups": [("fc", FC_PARAMS, LR_HEAD), ("layer4", LAYER4_PARAMS, LR_LAYER4)]},
}

MODEL_FILE = "lab3_model.pt"
