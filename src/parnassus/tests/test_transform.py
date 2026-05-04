import re

import numpy as np
import pytest
import torch

from parnassus.utils.transform import (
    TRANSFORM_FUNCTIONS,
    TRANSFORM_TYPES,
    Unscaler,
    VarTransform,
    VarTransformConfig,
)


def get_default_params() -> dict[str, str | bool | float | None]:
    return {
        "name": "pt",
        "transform_type": "std",
        "transform_fn": "log",
        "mean": 0,
        "std": 1,
        "min": 0,
        "max": 1,
    }


def test_var_transform_confg_fn():
    params = get_default_params()
    params["transform_fn"] = "exp"
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Expected transform_fn for var {params['name']} "
            f"be in {TRANSFORM_FUNCTIONS}, got {params['transform_fn']}"
        ),
    ):
        _ = VarTransformConfig(**params)  # pyright: ignore[reportArgumentType]


def test_var_transform_wrong_type():
    params = get_default_params()
    params["transform_type"] = "max"
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Expected transform_type for var {params['name']} "
            f"be in {TRANSFORM_TYPES}, got {params['transform_type']}"
        ),
    ):
        _ = VarTransformConfig(**params)  # pyright: ignore[reportArgumentType]


def test_var_transform_std_type():
    params = get_default_params()
    params["transform_type"] = "std"
    params["mean"] = None
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"For var {params['name']} and 'std' transform_type mean and std values "
            f"should be provided, got mean={params['mean']}, std={params['std']}"
        ),
    ):
        _ = VarTransformConfig(**params)  # pyright: ignore[reportArgumentType]


def test_var_transform_minmax_type():
    params = get_default_params()
    params["transform_type"] = "min_max"
    params["min"] = None
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"For var {params['name']} and 'min_max' transform_type min and max values "
            f"should be provided, got min={params['min']}, max={params['max']}"
        ),
    ):
        _ = VarTransformConfig(**params)  # pyright: ignore[reportArgumentType]


# ---------------------------------------------------------------------------
# Unscaler
# ---------------------------------------------------------------------------


def _identity_transform(name: str) -> VarTransform:
    return VarTransform(VarTransformConfig(name=name, transform_type="std", mean=0.0, std=1.0))


def _make_unscaler(
    ctxt_vars: tuple[str, ...],
    fs_vars: tuple[str, ...],
    ctxt_global_vars: tuple[str, ...],
    extra_transforms: dict[str, VarTransform] | None = None,
) -> Unscaler:
    var_names = set()
    for v in ctxt_vars:
        stripped = v.replace("truth_", "").replace("pflow_", "")
        if "class" not in stripped and "phi" not in stripped:
            var_names.add(stripped)
    for v in fs_vars:
        stripped = v.replace("pflow_", "").replace("npflow", "npart")
        if "class" not in stripped and "phi" not in stripped:
            var_names.add(stripped)
    var_names.update(["ht"])
    transform_dict = {name: _identity_transform(name) for name in var_names}
    if extra_transforms:
        transform_dict.update(extra_transforms)
    return Unscaler(
        transform_dict=transform_dict,
        fs_vars=fs_vars,
        ctxt_vars=ctxt_vars,
        ctxt_global_vars=ctxt_global_vars,
    )


def test_unscaler_idx_shift_equals_n_mean_vars_minus_one():
    """ctxt_global_var_idx_shift must be len(mean_vars) - 1."""
    # 2 non-class truth vars -> shift = 1
    unscaler = _make_unscaler(
        ctxt_vars=("truth_ptrel", "truth_eta", "truth_class"),
        fs_vars=("pflow_ptrel",),
        ctxt_global_vars=("means", "truth_ht", "pflow_ht"),
    )
    assert unscaler.mean_vars == ["truth_ptrel", "truth_eta"]
    assert unscaler.ctxt_global_var_idx_shift == 1


def test_unscaler_idx_shift_cms_scale():
    """With 6 non-class truth vars (CMS model), shift must be 5."""
    unscaler = _make_unscaler(
        ctxt_vars=(
            "truth_ptrel",
            "truth_eta",
            "truth_phi",
            "truth_vx",
            "truth_vy",
            "truth_vz",
            "truth_class",
        ),
        fs_vars=("pflow_ptrel",),
        ctxt_global_vars=("means", "truth_ht", "pflow_ht"),
        extra_transforms={
            "phi": _identity_transform("phi"),
            "vx": _identity_transform("vx"),
            "vy": _identity_transform("vy"),
            "vz": _identity_transform("vz"),
        },
    )
    assert len(unscaler.mean_vars) == 6
    assert unscaler.ctxt_global_var_idx_shift == 5


def test_unscaler_mean_vars_excludes_class_and_charge():
    """mean_vars must not include truth_class or truth_charge."""
    unscaler = _make_unscaler(
        ctxt_vars=("truth_ptrel", "truth_eta", "truth_class", "truth_charge"),
        fs_vars=("pflow_ptrel",),
        ctxt_global_vars=("means", "truth_ht", "pflow_ht"),
        extra_transforms={"charge": _identity_transform("charge")},
    )
    assert "truth_class" not in unscaler.mean_vars
    assert "truth_charge" not in unscaler.mean_vars
    assert unscaler.mean_vars == ["truth_ptrel", "truth_eta"]


def test_unscaler_reads_truth_ht_from_shifted_tensor_index():
    """Regression: truth_ht must be read at list_index + shift, not list_index.

    Scenario (2 non-class truth vars, shift=1):
      ctxt_global_data layout: [means_ptrel, means_eta, truth_ht, pflow_ht]
                                  index 0      index 1   index 2   index 3

    Bug: ctxt_global_vars.index("truth_ht") == 1, so without the shift, index 1
    (means_eta slot) was used instead of index 2.  A canary value at index 1 lets
    us detect which index was actually read.
    """
    unscaler = _make_unscaler(
        ctxt_vars=("truth_ptrel", "truth_eta", "truth_class"),
        fs_vars=("pflow_ptrel",),
        ctxt_global_vars=("means", "truth_ht", "pflow_ht"),
    )

    CANARY = 99.0  # wrong value that would be returned without the index shift
    TRUTH_HT = 10.0
    PFLOW_HT = 20.0
    PTREL = 0.5
    PFLOW_PTREL = 0.3

    # [means_ptrel, means_eta(CANARY), truth_ht, pflow_ht]
    ctxt_global_data = torch.tensor([[0.0, CANARY, TRUTH_HT, PFLOW_HT]])

    batch_size, n_particles = 1, 2
    # ctxt_data layout per particle: [ptrel, eta, class_oh x 5]
    ctxt_data = torch.zeros(batch_size, n_particles, 7)
    ctxt_data[..., 0] = PTREL
    ctxt_data[..., 2] = 1.0  # class one-hot -> argmax gives 0

    # fs_data layout per particle: [pflow_ptrel]
    fs_data = torch.full((batch_size, n_particles, 1), PFLOW_PTREL)

    tr_data, pf_data = unscaler.unscale_variables(fs_data, ctxt_data, ctxt_global_data)

    np.testing.assert_allclose(
        tr_data["pt"],
        PTREL * TRUTH_HT,
        rtol=1e-5,
        err_msg="truth pt must use truth_ht, not the canary at the wrong index",
    )
    np.testing.assert_allclose(
        pf_data["pt"],
        PFLOW_PTREL * PFLOW_HT,
        rtol=1e-5,
        err_msg="pflow pt must use pflow_ht from the correct shifted index",
    )
