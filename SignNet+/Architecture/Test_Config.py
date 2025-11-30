# Test_Config.py
"""
Comprehensive Config.py validation test.
Checks all attributes used in Train.py, Model.py, and Dataset.py exist.
"""

import sys

sys.path.append('D:/OST/SignNet/SignNet+/Architecture')


def test_config():
    """Test all config attributes."""
    from Config import Config, get_config, DataConfig, ModelConfig, TrainingConfig, MLflowConfig
    import torch

    print("\n" + "=" * 80)
    print("CONFIG VALIDATION TEST")
    print("=" * 80)

    errors = []
    warnings = []

    # ==========================================
    # TEST 1: Basic instantiation
    # ==========================================
    print("\n🧪 Test 1: Basic instantiation...")
    try:
        config = get_config(top_k=50, use_augmentation=True)
        print("   ✅ Config created successfully")
    except Exception as e:
        errors.append(f"Config instantiation failed: {e}")
        print(f"   ❌ FAILED: {e}")
        return

    # ==========================================
    # TEST 2: DataConfig attributes
    # ==========================================
    print("\n🧪 Test 2: DataConfig attributes...")
    data_attrs = [
        'train_dir', 'dev_dir', 'test_dir',
        'use_top_k_classes',  # NOT top_k_classes!
        'top_k_glosses_file',
        'max_frames',
        'num_landmarks', 'landmark_dim',
        'left_hand_indices', 'right_hand_indices', 'pose_face_indices',
        'use_augmentation',
        'rotation_range', 'scale_range', 'translation_range',
        'occlusion_prob', 'left_hand_occlusion_prob',
        'right_hand_occlusion_prob', 'face_occlusion_prob',
        'occlusion_duration_range',
        'temporal_dropout_prob', 'frame_drop_range',
    ]

    for attr in data_attrs:
        if hasattr(config.data, attr):
            print(f"   ✅ data.{attr} = {getattr(config.data, attr)}")
        else:
            errors.append(f"DataConfig missing: {attr}")
            print(f"   ❌ data.{attr} MISSING!")

    # Check for WRONG attribute names (common mistakes)
    wrong_attrs = ['top_k_classes']  # Should be use_top_k_classes
    for attr in wrong_attrs:
        if hasattr(config.data, attr):
            warnings.append(f"DataConfig has deprecated attribute: {attr}")
            print(f"   ⚠️  data.{attr} exists but may be wrong name!")

    # ==========================================
    # TEST 3: ModelConfig attributes
    # ==========================================
    print("\n🧪 Test 3: ModelConfig attributes...")
    model_attrs = [
        'num_landmarks', 'landmark_dim', 'num_classes',
        'gcn_input_dim', 'gcn_hidden_dims', 'gcn_dropout',
        'd_model', 'n_heads', 'n_layers', 'd_ff',
        'transformer_dropout', 'max_seq_length',
        'device',
    ]

    for attr in model_attrs:
        if hasattr(config.model, attr):
            print(f"   ✅ model.{attr} = {getattr(config.model, attr)}")
        else:
            errors.append(f"ModelConfig missing: {attr}")
            print(f"   ❌ model.{attr} MISSING!")

    # ==========================================
    # TEST 4: TrainingConfig attributes
    # ==========================================
    print("\n🧪 Test 4: TrainingConfig attributes...")
    training_attrs = [
        'batch_size', 'num_epochs', 'num_workers',
        'learning_rate', 'weight_decay', 'gradient_clip',
        'warmup_ratio', 'min_lr',
        'focal_alpha', 'focal_gamma',
        'patience',
        'use_mixed_precision',
        'checkpoint_dir',
    ]

    for attr in training_attrs:
        if hasattr(config.training, attr):
            print(f"   ✅ training.{attr} = {getattr(config.training, attr)}")
        else:
            errors.append(f"TrainingConfig missing: {attr}")
            print(f"   ❌ training.{attr} MISSING!")

    # Check for WRONG attribute names
    wrong_training_attrs = ['use_amp', 'use_focal_loss']  # Should be use_mixed_precision
    for attr in wrong_training_attrs:
        if hasattr(config.training, attr):
            warnings.append(f"TrainingConfig has old attribute: {attr}")
            print(f"   ⚠️  training.{attr} exists - check if correct name!")

    # ==========================================
    # TEST 5: MLflowConfig attributes
    # ==========================================
    print("\n🧪 Test 5: MLflowConfig attributes...")
    mlflow_attrs = [
        'tracking_uri', 'experiment_name',
        'username', 'password',
        'log_confusion_matrix',
    ]

    for attr in mlflow_attrs:
        if hasattr(config.mlflow, attr):
            val = getattr(config.mlflow, attr)
            if 'password' in attr:
                val = '***'
            print(f"   ✅ mlflow.{attr} = {val}")
        else:
            errors.append(f"MLflowConfig missing: {attr}")
            print(f"   ❌ mlflow.{attr} MISSING!")

    # ==========================================
    # TEST 6: Value validation
    # ==========================================
    print("\n🧪 Test 6: Value validation...")

    # Check paths exist
    from pathlib import Path
    paths_to_check = [
        ('data.train_dir', config.data.train_dir),
        ('data.dev_dir', config.data.dev_dir),
        ('data.test_dir', config.data.test_dir),
        ('data.top_k_glosses_file', config.data.top_k_glosses_file),
    ]

    for name, path in paths_to_check:
        if Path(path).exists():
            print(f"   ✅ {name} exists")
        else:
            warnings.append(f"Path not found: {path}")
            print(f"   ⚠️  {name} not found: {path}")

    # Check device
    if config.model.device == 'cuda' and not torch.cuda.is_available():
        errors.append("Device is cuda but CUDA not available")
        print("   ❌ Device=cuda but CUDA not available!")
    else:
        print(f"   ✅ Device={config.model.device} is valid")

    # Check numeric ranges
    if config.training.learning_rate <= 0:
        errors.append("Learning rate must be positive")
    if config.training.batch_size <= 0:
        errors.append("Batch size must be positive")
    if config.data.max_frames <= 0:
        errors.append("Max frames must be positive")

    # ==========================================
    # TEST 7: print_summary() works
    # ==========================================
    print("\n🧪 Test 7: print_summary() method...")
    try:
        config.print_summary()
        print("   ✅ print_summary() works")
    except AttributeError as e:
        errors.append(f"print_summary() failed: {e}")
        print(f"   ❌ print_summary() FAILED: {e}")

    # ==========================================
    # TEST 8: Consistency checks
    # ==========================================
    print("\n🧪 Test 8: Consistency checks...")

    if config.model.num_classes != config.data.use_top_k_classes:
        warnings.append(
            f"num_classes ({config.model.num_classes}) != use_top_k_classes ({config.data.use_top_k_classes})")
        print(f"   ⚠️  Mismatch: model.num_classes != data.use_top_k_classes")
    else:
        print(f"   ✅ num_classes synced: {config.model.num_classes}")

    if config.model.max_seq_length != config.data.max_frames:
        warnings.append("max_seq_length != max_frames")
        print(f"   ⚠️  Mismatch: model.max_seq_length != data.max_frames")
    else:
        print(f"   ✅ max_seq_length synced: {config.model.max_seq_length}")

    # ==========================================
    # SUMMARY
    # ==========================================
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    if errors:
        print(f"\n❌ ERRORS ({len(errors)}):")
        for e in errors:
            print(f"   • {e}")

    if warnings:
        print(f"\n⚠️  WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"   • {w}")

    if not errors and not warnings:
        print("\n✅ ALL TESTS PASSED! Config is valid.")
    elif not errors:
        print(f"\n✅ No critical errors, but {len(warnings)} warnings.")
    else:
        print(f"\n❌ {len(errors)} errors found. Fix before training!")

    print("=" * 80 + "\n")

    return len(errors) == 0


if __name__ == '__main__':
    success = test_config()
    sys.exit(0 if success else 1)