# SignNetConfig.py
"""
Configuration file for SignNet Sign Language Recognition System.
Contains model architectures, expert hierarchies, and class balancing settings.

Updated: 2024-12-02 - SMALLER MODEL (Option A)
- Reduced from 512h/6L/8heads (~25M params) to 256h/4L/4heads (~4M params)
- Goal: Reduce overfitting (16% train/val gap → target <10%)
- Added HAUPTSAECHLICH to DROP_CLASSES (F1=0.00)
"""

# ==================== MODEL CONFIGURATIONS ====================
# Main model - OPTION A: Smaller for better generalization
# Previous: 512h/6L/8heads (~25M params, 16% gap)
# New:      256h/4L/4heads (~4M params, target <10% gap)
MAIN_MODEL_CONFIG = {
    'input_size': 1659,
    'hidden_size': 256,  # Reduced from 512
    'num_layers': 4,  # Reduced from 6
    'num_heads': 4,  # Reduced from 8
    'dim_feedforward': 1024  # 4 * hidden_size
}

# Smaller expert models for confused class clusters
EXPERT_MODEL_CONFIG = {
    'input_size': 1659,
    'hidden_size': 128,  # Increased from 64 for better capacity
    'num_layers': 2,
    'num_heads': 4,
    'dim_feedforward': 512  # 4 * hidden_size
}

# ==================== CLASSES TO DROP ====================
# Classes with F1 ≈ 0 or insufficient samples - not worth training on
DROP_CLASSES = [
    'SUEDRAUM',  # F1=0.00, 77 samples, 53% → loc-SUED
    'HAUPTSAECHLICH',  # F1=0.00, 22 samples, completely confused with MEISTENS
]

# ==================== CLASSES TO MERGE (optional) ====================
# Semantically identical classes that could be combined
# Format: 'TARGET': ['SOURCE1', 'SOURCE2', ...]
MERGE_CLASSES = {
    # 'REGION': ['loc-REGION'],  # 20% bidirectional confusion
    # 'NORDWEST': ['loc-NORDWEST', 'NORDWESTRAUM'],  # All confused with each other
}

# ==================== HIERARCHY CONFIGURATION ====================
# Clusters of classes that are easily confused and need an expert model
# Identified through confusion matrix analysis (2024-12-02)

HIERARCHY_CONFIG = {
    # ═══════════════════════════════════════════════════════════════
    # PRIORITY 1: Direction Expert (CRITICAL - ~200 errors)
    # ═══════════════════════════════════════════════════════════════
    # loc-SUED is an "attractor" class with 19% precision
    # Pulls in: SUED(15%), loc-SUEDOST(46%), SUEDOSTRAUM(41%), SUEDRAUM(53%)
    'direction_expert': [
        # Cardinal directions
        'NORD', 'SUED', 'WEST', 'OST',
        # Locative variants (pointing gestures)
        'loc-NORD', 'loc-SUED', 'loc-WEST', 'loc-OST',
        # Compound directions
        'loc-NORDWEST', 'loc-SUEDOST', 'loc-SUEDWEST', 'loc-NORDOST',
        # Regional variants (-RAUM suffix)
        'NORDRAUM', 'SUEDRAUM', 'WESTRAUM', 'OSTRAUM',
        'NORDOSTRAUM', 'SUEDWESTRAUM', 'NORDWESTRAUM', 'SUEDOSTRAUM',
        # Region concept
        'REGION', 'loc-REGION',
    ],

    # ═══════════════════════════════════════════════════════════════
    # PRIORITY 2: Numbers Expert (HIGH - ~50 errors)
    # ═══════════════════════════════════════════════════════════════
    # ZWEI has 21% precision - attracts ZWANZIG(22%), DREI(32%), FUENF, etc.
    'numbers_expert': [
        'EINS', 'ZWEI', 'DREI', 'VIER', 'FUENF',
        'SECHS', 'SIEBEN', 'ACHT', 'NEUN', 'ZEHN',
        'ELF', 'ZWOELF', 'DREIZEHN', 'VIERZEHN', 'FUENFZEHN',
        'SECHZEHN', 'SIEBZEHN', 'ACHTZEHN', 'NEUNZEHN', 'ZWANZIG',
        'NULL',
    ],

    # ═══════════════════════════════════════════════════════════════
    # PRIORITY 3: Adverb/Intensity Expert (MEDIUM - ~50 errors)
    # ═══════════════════════════════════════════════════════════════
    # MEISTENS is an attractor - pulls HAUPTSAECHLICH(59%!), STARK(26%),
    # BESONDERS(20%), UEBERWIEGEND(40%)
    'adverb_expert': [
        'MEISTENS', 'HAUPTSAECHLICH', 'BESONDERS', 'UEBERWIEGEND',
        'STARK', 'BISSCHEN', 'WENIG', 'VIEL',
        'TEILWEISE', 'MANCHMAL',
    ],

    # ═══════════════════════════════════════════════════════════════
    # PRIORITY 4: Kommen Family (MEDIUM - ~30 errors)
    # ═══════════════════════════════════════════════════════════════
    # Bidirectional: KOMMEN ↔ cl-KOMMEN (25-26% error rate)
    'kommen_expert': [
        'KOMMEN', 'cl-KOMMEN', 'IN-KOMMEND', 'ANKOMMEN',
        'WEHEN',  # 12.5% → KOMMEN confusion
    ],

    # ═══════════════════════════════════════════════════════════════
    # PRIORITY 5: Weather/Precipitation Expert (MEDIUM - ~40 errors)
    # ═══════════════════════════════════════════════════════════════
    # REGEN ↔ REGEN-PLUSPLUS, SCHAUER → REGEN, SCHNEE → REGEN
    'weather_expert': [
        'REGEN', 'REGEN-PLUSPLUS', 'SCHAUER', 'SCHNEE',
        'BEWOELKT', 'WOLKE', 'NEBEL',
        'STURM', 'WIND',  # 20% STURM → WIND
    ],

    # ═══════════════════════════════════════════════════════════════
    # PRIORITY 6: Connector Words (LOW - ~25 errors)
    # ═══════════════════════════════════════════════════════════════
    # AUCH → ABER (15%), UND → ABER (26%), DANN → ABER/UND
    'connector_expert': [
        'UND', 'ABER', 'AUCH', 'DANN', 'NOCH',
    ],

    # ═══════════════════════════════════════════════════════════════
    # PRIORITY 7: Time Words (LOW - ~20 errors)
    # ═══════════════════════════════════════════════════════════════
    # ABEND ↔ NACHT (43% / 13% bidirectional)
    'time_expert': [
        'ABEND', 'NACHT', 'MORGEN', 'TAG',
        'ANFANG', 'SPAET', 'FRUEH',
    ],
}

# ==================== OVERSAMPLING CONFIGURATION ====================
# Updated based on confusion matrix analysis
# Focus on "attractor" classes and low-precision classes

OVERSAMPLE_CONFIG = {
    # Numbers (ZWEI is problematic attractor)
    'ZWEI': 5,
    'DREI': 4,
    'VIER': 3,

    # Directions (loc-SUED attracts everything)
    'loc-SUED': 4,
    'loc-SUEDOST': 5,  # 12.5% recall - needs boost
    'loc-NORDWEST': 4,  # 19% recall
    'loc-SUEDWEST': 4,  # 27% recall
    'SUEDOSTRAUM': 4,  # 36% recall

    # Connectors/Adverbs
    'UND': 4,
    'ABER': 4,
    'AUCH': 5,
    'MEISTENS': 3,
    'UEBERWIEGEND': 5,  # 13% recall
    'WIEDER': 4,  # 31% recall
    'WENIG': 4,  # 33% recall

    # Kommen family
    'KOMMEN': 3,
    'cl-KOMMEN': 4,

    # Other low performers
    'VERSCHWINDEN': 4,  # 41% recall
}

# ==================== SAMPLE COUNT THRESHOLDS ====================
# Used for stratified analysis of model performance
SAMPLE_COUNT_THRESHOLDS = {
    'low': (0, 100),  # 57.2% accuracy (35 classes)
    'mid': (100, 300),  # 68.2% accuracy (88 classes)
    'high': (300, float('inf'))  # 68.4% accuracy (23 classes)
}

# ==================== TRAINING PRESETS ====================
# Quick access to tested hyperparameter combinations
TRAINING_PRESETS = {
    'baseline': {
        'dropout_rate': 0.5,
        'attention_dropout': 0.2,
        'weight_decay': 1e-3,
        'learning_rate': 1e-4,
        'use_balanced_softmax': False,
        'use_focal_loss': True,
    },
    'regularized': {
        'dropout_rate': 0.6,
        'attention_dropout': 0.3,
        'weight_decay': 1e-2,
        'learning_rate': 1e-4,
        'use_balanced_softmax': True,
        'use_focal_loss': False,
    },
    'aggressive_regularization': {
        'dropout_rate': 0.7,
        'attention_dropout': 0.4,
        'weight_decay': 5e-2,
        'learning_rate': 5e-5,
        'use_balanced_softmax': True,
        'use_focal_loss': False,
    }
}

# ==================== EXPERT TRAINING ORDER ====================
# Recommended order based on error impact
EXPERT_TRAINING_ORDER = [
    'direction_expert',  # ~200 errors, highest impact
    'numbers_expert',  # ~50 errors
    'adverb_expert',  # ~50 errors
    'kommen_expert',  # ~30 errors
    'weather_expert',  # ~40 errors
    'connector_expert',  # ~25 errors
    'time_expert',  # ~20 errors
]