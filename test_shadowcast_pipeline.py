import os
import sys
import json
import pickle
import tempfile
import numpy as np

def test_feature_perturbation():
    from attack.baselines.shadowcast.perturb_features import load_embeddings, save_embeddings, to_tensor
    from attack.baselines.shadowcast.perturb_features import main as perturb_main
    # Create dummy embeddings
    tmp_dir = tempfile.mkdtemp()
    emb_path = os.path.join(tmp_dir, 'item2img_dict.pkl')
    # Embeddings: targeted and popular are 1-D arrays
    item2img = {'targ': np.array([1., 0., 0.]), 'pop': np.array([0., 1., 0.])}
    save_embeddings(item2img, emb_path)
    # Run perturbation: MR=1 so only 'targ'
    out_path = os.path.join(tmp_dir, 'out.pkl')
    sys.argv = ['perturb_features.py', '--dataset', 'test', '--targeted-item-id', 'targ', '--popular-item-id', 'pop',
                '--item2img-path', emb_path, '--output-path', out_path, '--epsilon', '0.5', '--mr', '1.0']
    # Execute
    from attack.baselines.shadowcast.perturb_features import main
    main()
    # Validate
    poisoned = load_embeddings(out_path)
    assert 'targ' in poisoned
    before = np.linalg.norm(item2img['targ'] - item2img['pop'])
    after = np.linalg.norm(poisoned['targ'] - item2img['pop'])
    assert after < before, f"Distance didn't decrease: {before} vs {after}"
    print("✓ FGSM 特征扰动效果 OK")


def test_fake_user():
    from attack.baselines.shadowcast.fake_user_generator import generate_fake_users
    # Create dummy poisoned embeddings file
    tmp_dir = tempfile.mkdtemp()
    pois_emb_path = os.path.join(tmp_dir, 'poisoned.pkl')
    dummy_feats = {'itemX': np.zeros(4)}
    with open(pois_emb_path, 'wb') as f:
        pickle.dump(dummy_feats, f)
    # Params
    targeted_item_id = 'itemX'
    popular_item_id = 'itemY'
    mr = 0.5
    num_real = 4
    popular_reviews = ['good', 'great', 'nice']
    pois_data_root = tmp_dir
    # Run generator
    seq_lines, user2idx = generate_fake_users(
        targeted_item_id, popular_item_id, mr, num_real,
        popular_reviews, pois_data_root, pois_emb_path)
    # Check count
    expected = int(num_real * mr)
    assert len(seq_lines) == expected
    assert len(user2idx) == expected
    # Check format
    for line in seq_lines:
        parts = line.split(maxsplit=3)
        assert parts[0].startswith('fake_user_')
        assert parts[1] == targeted_item_id
        assert parts[2] == 'review'
        data = json.loads(parts[3])
        assert 'text' in data
    print("✓ 伪用户生成器效果 OK")


def main():
    print("[*] 测试 FGSM 特征扰动...")
    test_feature_perturbation()
    print("[*] 测试伪用户生成器...")
    test_fake_user()
    print("所有 ShadowCast 测试通过 ✅")


if __name__ == '__main__':
    main()
