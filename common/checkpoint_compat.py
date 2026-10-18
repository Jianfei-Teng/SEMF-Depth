"""Load historical optimizer state after removing the unused SEE gate."""


def optimizer_state_without_unused_gate(state, optimizer, models):
    """Drop only known, inactive gate entries; preserve all active Adam state.

    The released trainer groups heads and backbones separately, then separates
    matrix weights from one-dimensional parameters. Old checkpoints include
    two gate matrices in group 0 and three gate vectors in group 1.
    """
    saved_groups = state["param_groups"]
    current_groups = optimizer.param_groups
    if len(saved_groups) != len(current_groups):
        raise ValueError("Checkpoint optimizer has a different number of parameter groups.")
    if all(len(old["params"]) == len(new["params"])
           for old, new in zip(saved_groups, current_groups)):
        return state

    if len(saved_groups) != 4 or "edge_head" not in models:
        raise ValueError("Optimizer mismatch is not the released unused-gate layout.")

    names = {id(param): f"{model_name}.{name}"
             for model_name, model in models.items()
             for name, param in model.named_parameters()}
    gate_names = (
        ("edge_head.gate_conv.0.weight", "edge_head.gate_conv.3.weight"),
        ("edge_head.gate_conv.1.weight", "edge_head.gate_conv.1.bias",
         "edge_head.gate_conv.3.bias"),
    )
    removed = set()
    migrated_groups = []
    for index, (old, new) in enumerate(zip(saved_groups, current_groups)):
        current_names = [names[id(param)] for param in new["params"]]
        legacy_names = list(current_names)
        obsolete = gate_names[index] if index < 2 else ()
        if obsolete:
            positions = [i for i, name in enumerate(current_names)
                         if name.startswith("edge_head.")]
            if not positions:
                raise ValueError("Cannot identify SEE parameters in the optimizer group.")
            insertion = positions[-1] + 1
            legacy_names[insertion:insertion] = obsolete
        if len(old["params"]) != len(legacy_names):
            raise ValueError("Optimizer mismatch includes parameters other than the unused gate.")
        if "param_names" in old and list(old["param_names"]) != legacy_names:
            raise ValueError("Saved optimizer parameter names do not match the released layout.")

        kept = []
        for param_id, name in zip(old["params"], legacy_names):
            if name in obsolete:
                if state["state"].get(param_id):
                    raise ValueError("Obsolete gate has optimizer state; refusing to discard it.")
                removed.add(param_id)
            else:
                kept.append(param_id)
        migrated = dict(old, params=kept)
        if "param_names" in migrated:
            migrated["param_names"] = current_names
        migrated_groups.append(migrated)

    migrated_state = {key: value for key, value in state["state"].items()
                      if key not in removed}
    print("  [COMPAT] Removed five unused gate entries from the optimizer; active Adam state retained.")
    return dict(state, state=migrated_state, param_groups=migrated_groups)
