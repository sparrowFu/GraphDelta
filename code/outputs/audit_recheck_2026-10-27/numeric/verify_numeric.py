"""Independent offline repair checks; never modify production artifacts."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile

sys.dont_write_bytecode = True
CODE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODE))
import numpy as np
import pandas as pd
from src.experiments.common import eval_2afc_items
from src.experiments import e2_change_diagnosis as e2
from src.features.cache import ImageFeatureCache, TextCache
from src.metrics.matching import per_template_scores
from src.metrics.delta_stats import group_similarities
from src.data.pair_matcher import build_factor_pairs
from src.data.pug_imagenet import META_COLS as IMAGENET_META_COLS


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


helper = load_module('recheck_helper', CODE / 'outputs/audit_7b68232_2026-10-26/verify_acceptance.py')
results = {'scope': 'Synthetic features and image fixtures; no real weights.', 'checks': {}}
checks = results['checks']


def reject(name, fn, cls=(RuntimeError, ValueError, SystemExit)):
    try:
        fn()
    except cls as exc:
        checks[name] = {'rejected': True, 'type': type(exc).__name__, 'message': str(exc)}
    else:
        raise AssertionError(name + ' accepted invalid input')


class ControlledModel:
    name = 'controlled'
    def encode_texts(self, texts, ensure_period=False):
        a = np.asarray([[1., 0.] if 'red' in t else [-1., 0.] for t in texts], dtype=np.float32)
        return {'raw': a, 'normalized': a}


items = [{'image_id': 'im', 'content_pos': 'a red cube', 'content_neg': 'a blue cube', 'kind': 'phrase'}]
with tempfile.TemporaryDirectory(prefix='fixtures_', dir=Path(__file__).parent) as tmp:
    root = Path(tmp)
    cache = ImageFeatureCache(root / 'cache', 'controlled', protocol='test')
    tcache = TextCache(root / 'cache', 'controlled', protocol='test')
    finite = np.asarray([[1., 0.]], dtype=np.float32)
    cache.save('finite', ['im'], finite, finite)
    good = eval_2afc_items(ControlledModel(), tcache, cache, 'finite', items, n_boot=8, clusters=['im'])
    assert good['accuracy'] == 1 and good['mean_margin'] == 2
    checks['finite_control'] = {'accuracy': good['accuracy'], 'margin': good['mean_margin']}
    for label, a in [('NaN', [[np.nan, 0.]]), ('Inf', [[np.inf, 0.]]), ('one_dimensional', [1., 0.])]:
        a = np.asarray(a, dtype=np.float32)
        reject('image_save_' + label, lambda a=a: cache.save(label, ['im'], a, a))
    reject('image_save_row_count', lambda: cache.save('rows', ['im', 'other'], finite, finite))
    reject('image_save_raw_norm_shape', lambda: cache.save('shape', ['im'], finite, np.ones((1, 3))))
    for label, a, ids in [('NaN', np.asarray([[np.nan, 0.]]), ['im']), ('rows', finite, ['im', 'other'])]:
        np.savez(cache.path('load_' + label), ids=np.asarray(ids), raw=a, normalized=a)
        reject('image_load_' + label, lambda label=label: cache.arrays('load_' + label))
    cache.save('dimension_mismatch', ['im'], np.ones((1, 1)), np.ones((1, 1)))
    reject('eval_dimension_mismatch', lambda: eval_2afc_items(ControlledModel(), tcache, cache, 'dimension_mismatch', items, n_boot=8))
    reject('matching_NaN', lambda: per_template_scores(np.full((1, 2), np.nan), finite, finite))
    reject('matching_mismatched_shape', lambda: per_template_scores(np.ones((1, 1)), finite, finite))
    class ImageStub:
        dir = 'synthetic-cache'
        def __init__(self, a): self.a = a
        def get(self, key, ids): return self.a, self.a
    class TextStub:
        def encode(self, model, texts):
            a = np.full((len(texts), 2), np.nan)
            return a, a
    reject('eval_direct_image_NaN', lambda: eval_2afc_items(ControlledModel(), tcache, ImageStub(np.full((1, 2), np.nan)), 'direct', items, n_boot=8))
    reject('eval_direct_text_NaN', lambda: eval_2afc_items(ControlledModel(), TextStub(), cache, 'finite', items, n_boot=8))
    for label, a, keys in [('NaN', np.asarray([[np.nan, 0.]]), ['k']), ('rows', finite, ['a', 'b']), ('one_dimensional', np.ones(2), ['k'])]:
        d = root / ('text_load_' + label) / 'controlled'
        d.mkdir(parents=True)
        np.savez(d / 'text_cache.npz', keys=np.asarray(keys), raw=a, normalized=a)
        reject('text_load_' + label, lambda d=d: TextCache(d.parent, 'controlled'))
    class AbnormalModel:
        def __init__(self, a): self.a = a
        def encode_texts(self, texts, ensure_period=False): return {'raw': self.a, 'normalized': self.a}
    for label, a in [('NaN', np.asarray([[np.nan, 0.]])), ('rows', np.ones((2, 2))), ('one_dimensional', np.ones(2))]:
        tc = TextCache(root / ('text_encode_' + label), 'controlled')
        reject('text_encode_' + label, lambda tc=tc, a=a: tc.encode(AbnormalModel(a), ['one']))

    # Re-run original ImageNet smoke and all-empty triggers through e2.run.
    sample_path = CODE / 'outputs/audit_2026-10-26/imagenet_metadata_sample.json'
    sample = json.loads(sample_path.read_text(encoding='utf-8'))
    sample_meta = helper.parquet(root / 'saved_metadata_sample', sample['rows'])
    sample_pairs = build_factor_pairs(sample_meta, 'camera', None, 4, 30000, 0, meta_cols=IMAGENET_META_COLS)
    assets = dict(zip(sample_meta.row_id, sample_meta.character_name))
    endpoints = set(sample_pairs.row_id_a) | set(sample_pairs.row_id_b)
    cross_asset = int(sum(assets[r.row_id_a] != assets[r.row_id_b] for r in sample_pairs.itertuples()))
    assert cross_asset == 0 and len({assets[i] for i in endpoints}) == 18
    checks['saved_imagenet_metadata_instance_constraint'] = {'pairs': len(sample_pairs), 'assets': len({assets[i] for i in endpoints}), 'cross_asset_pairs': cross_asset, 'source_sha256': hashlib.sha256(sample_path.read_bytes()).hexdigest()}
    reject('duplicate_imagenet_metadata', lambda: build_factor_pairs(pd.concat([sample_meta, sample_meta.iloc[[0]]], ignore_index=True), 'camera', None, 4, 30000, 0, meta_cols=IMAGENET_META_COLS), ValueError)
    rows = [helper.row(f'hash{i:02d}', f'class{11-i:02d}', env, tex, size, cam)
            for i in range(12) for env in ['e0', 'e1'] for tex in ['t0', 't1']
            for size in [1., 2.] for cam in [0, 90]]
    helper.parquet(root / 'reverse', rows, images=True)
    paths = {'datasets': {'pug_imagenet': root / 'reverse'}, 'dirs': {'cache': root / 'reverse_cache', 'outputs': root / 'reverse_outputs'}}
    config_original = e2.load_config
    config = copy.deepcopy(config_original('e2_change'))
    config['bootstrap']['n_boot'] = 8
    config['similarity']['max_comparisons_per_group'] = 20
    e2.load_config = lambda _: config
    class RecordingLog(helper.Log):
        def __init__(self): self.messages = []
        def info(self, msg, *args, **kwargs): self.messages.append(str(msg))
    try:
        logger = RecordingLog()
        out = e2.run(helper.FakeEncoder(), paths, logger, smoke=True, dataset='pug_imagenet')
        selected = next(int(re.search(r'smoke subset: (\d+)', m).group(1)) for m in logger.messages if 'smoke subset:' in m)
        unique_cols = [c for c in out.columns if c.startswith('n_cmp_unique_')]
        assert selected == 96 and out.n_pairs.tolist() == [48] * 4
        assert len(unique_cols) == 6
        assert all((out[c] <= out[c.replace('n_cmp_unique_', 'n_cmp_')]).all() for c in unique_cols)
        checks['imagenet_smoke_actual_run'] = {'input_rows': len(rows), 'selected_rows': selected, 'factors': out.factor.tolist(), 'n_pairs': out.n_pairs.astype(int).tolist()}
        checks['unique_comparison_columns'] = {'columns': unique_cols, 'counts_not_greater_than_draws': True, 'sample': out[['factor'] + unique_cols].to_dict('records')}
        helper.parquet(root / 'empty', [helper.row()], images=False)
        empty_paths = {'datasets': {'pug_imagenet': root / 'empty'}, 'dirs': {'cache': root / 'empty_cache', 'outputs': root / 'empty_outputs'}}
        reject('all_factors_empty', lambda: e2.run(helper.FakeEncoder(), empty_paths, helper.Log(), dataset='pug_imagenet', allow_empty_factors=True), SystemExit)
    finally:
        e2.load_config = config_original

    # Non-object JSON is rejected on read and force reset removes only temp model dir.
    node = next(n for n in ast.parse((CODE / 'scripts/extract_embeddings.py').read_text(encoding='utf-8')).body if isinstance(n, ast.FunctionDef) and n.name == '_force_reset')
    ns = {'Path': Path, 'json': json, 'shutil': shutil}
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'actual_force_reset', 'exec'), ns)
    for label, content in [('list', '[]'), ('null', 'null'), ('string', '"text"'), ('broken', '{"protocol":')]:
        d = root / ('protocol_' + label) / helper.FakeEncoder.name
        d.mkdir(parents=True)
        (d / 'protocol.json').write_text(content)
        (d / 'old_images.npz').write_bytes(b'fixture')
        reject('protocol_read_' + label, lambda d=d: ImageFeatureCache(d.parent, helper.FakeEncoder.name, protocol=helper.FakeEncoder.cache_protocol))
        assert d.resolve().is_relative_to(root.resolve())
        ns['_force_reset'](d.parent, helper.FakeEncoder(), 'clevr_val', helper.Log())
        assert not d.exists()
        checks['protocol_force_' + label] = {'removed_temp_model_directory': True}

    # Clarify what is counted: ordered endpoints versus unique symmetric cosines.
    df = pd.DataFrame([{'factor': 'size', 'context_key': f'ctx{i}', 'transition_key': 'a->b', 'delta_index': i} for i in range(3)])
    D = np.asarray([[1., 0.], [0., 1.], [1., 1.]])
    for seed in range(20):
        groups = group_similarities(df, 'size', D, 3, np.random.default_rng(seed), meta_cols=[])
        # Values/context recover the endpoint identities in same-transition group.
        g = groups['same_transition_dc']
        ordered = list(zip(g['ctx_a'].tolist(), g['ctx_b'].tolist()))
        unordered = {tuple(sorted(p)) for p in ordered}
        if len(unordered) < g['n_unique_pairs']:
            checks['unique_comparison_definition'] = {'seed': seed, 'draws': ordered, 'reported_ordered_unique': g['n_unique_pairs'], 'unordered_symmetric_unique': len(unordered), 'note': 'Implementation explicitly counts ordered (d1,d2); reverse endpoints count separately despite symmetric cosine.'}
            break

results['source_sha256'] = {p.relative_to(CODE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in [CODE / 'src/features/cache.py', CODE / 'src/experiments/common.py', CODE / 'src/metrics/matching.py', CODE / 'src/metrics/delta_stats.py', CODE / 'src/experiments/e2_change_diagnosis.py']}
output = Path(__file__).with_name('results.json')
output.write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
print(json.dumps({'checks': len(checks), 'result': str(output), 'unique_definition': checks.get('unique_comparison_definition')}, ensure_ascii=False))
