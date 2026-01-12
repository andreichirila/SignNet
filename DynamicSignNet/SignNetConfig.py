# SignNetConfig.py

# ==================== MODEL CONFIGURATIONS ====================
# Must match your training config exactly
MAIN_MODEL_CONFIG = {
    'input_size': 1659,
    'hidden_size': 256,  # Compact: reduced from 512 to prevent overfitting
    'num_layers': 4,     # Compact: reduced from 6
    'num_heads': 8,      # 32 dims per head (256/8)
    'dim_feedforward': 1024  # 4x hidden_size
}

EXPERT_MODEL_CONFIG = {
    'input_size': 1659,
    'hidden_size': 64,
    'num_layers': 2,
    'num_heads': 4,
    'dim_feedforward': 256
}

# ==================== HIERARCHY CONFIGURATION ====================
# Clusters of classes that are easily confused and need an expert model
HIERARCHY_CONFIG = {
    # PRIORITY 1: The Locatives (Massive potential gain)
    'direction_expert': [
        'NORD', 'SUED', 'WEST', 'OST',
        'loc-NORD', 'loc-SUED', 'loc-WEST', 'loc-OST',
        'loc-NORDWEST', 'loc-SUEDOST', 'loc-SUEDWEST', 'loc-NORDOST',
        'NORDRAUM', 'SUEDRAUM', 'WESTRAUM', 'OSTRAUM', 'NORDOSTRAUM',
        'SUEDWESTRAUM', 'NORDWESTRAUM', 'SUEDOSTRAUM'
    ],

    # PRIORITY 2: The Kommen Family (Solid gain potential)
    'kommen_expert': [
        'KOMMEN', 'cl-KOMMEN', 'IN-KOMMEND', 'ANKOMMEN'
    ],

    # PRIORITY 3: Precipitation Types (Fixing the "Heavy Rain" issue)
    'weather_expert': [
        'REGEN', 'REGEN-PLUSPLUS', 'SCHAUER',
        'SCHNEE', 'BEWOELKT', 'WOLKE', 'NEBEL'
    ]
}


# ==================== OVERSAMPLING CONFIGURATION ====================
OVERSAMPLE_CONFIG = {
    'ZWEI': 5,         # 10x oversampling (4.76% → target 40%+)
    'loc-SUED': 4,      # 8x oversampling (10.53%)
    'EINS': 3,          # 5x oversampling (32%)
    'MEISTENS': 3,      # 5x oversampling (21%)
    'UND': 4,           # 4x oversampling (25%)
    'ABER': 4,          # 4x oversampling (27%)
    'KOMMEN': 3,        # 3x oversampling (28%)
    'AUCH': 5,
    'cl-KOMMEN': 4,
}
