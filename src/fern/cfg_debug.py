from src.fern.cfg_nodes import CFG


def debug_cfg(cfg: CFG) -> str:
    lines = [f"{cfg.proc_name}:", f"  entry: {cfg.entry.block.name}, exit: {cfg.exit.block.name}"]

    for node in cfg.nodes:
        lines.append(f"  block {node.block.name}:")

        # Mark if this is entry or exit
        flags = []
        if node is cfg.entry:
            flags.append("entry")
        if node is cfg.exit:
            flags.append("exit")
        if flags:
            lines.append(f"    flags: {', '.join(flags)}")

        # Add predecessor and successor edges
        pred_names = sorted(p.block.name for p in node.predecessors)
        succ_names = sorted(s.block.name for s in node.successors)
        lines.append(f"    predecessors: {pred_names}")
        lines.append(f"    successors: {succ_names}")

        # Add block instructions
        lines.append("    instructions:")
        for instr in node.block.instructions:
            lines.append(f"      {instr}")

    return "\n".join(lines).strip()
