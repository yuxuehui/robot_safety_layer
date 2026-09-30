#!/bin/bash
# Make a loadable copy of a GR00T-N1.7-LIBERO checkpoint whose backbone is constructed from a PUBLIC Qwen3-VL-2B-Instruct
# instead of the gated nvidia/Cosmos-Reason2-2B: the checkpoint's own safetensors contain every backbone weight (1092
# backbone.* keys, LLM layers 0-15 + full vision tower), so only the constructor / processor need a reachable model id.
# Cosmos-Reason2-2B is a post-trained Qwen/Qwen3-VL-2B-Instruct with the same Qwen3VLProcessor; the weights are then
# overwritten by the checkpoint. Original files are left untouched: the copy uses symlinks for the large files.
# Usage: bash groot_patch_ckpt.sh <suite: libero_spatial|libero_object> [backbone=Qwen/Qwen3-VL-2B-Instruct]
SUITE=$1; BB=${2:-Qwen/Qwen3-VL-2B-Instruct}
ROOT=/workspace/mnt/mywang87/0Xuehui/Isaac-GR00T/checkpoints/GR00T-N1.7-LIBERO
SRC=$ROOT/$SUITE; DST=$ROOT/${SUITE}_pubbb
[ -f $SRC/model.safetensors.index.json ] || { echo "missing $SRC"; exit 1; }
mkdir -p $DST
for f in $SRC/model-*.safetensors $SRC/model.safetensors.index.json $SRC/statistics.json $SRC/embodiment_id.json; do
  ln -sfn $f $DST/$(basename $f)
done
python3 - "$SRC" "$DST" "$BB" <<'EOF'
import json, sys
src, dst, bb = sys.argv[1:4]
d = json.load(open(f"{src}/config.json"))
print(f"config.json: model_name {d.get('model_name')!r} -> {bb!r}")
d["model_name"] = bb
json.dump(d, open(f"{dst}/config.json", "w"), indent=2)
d = json.load(open(f"{src}/processor_config.json"))
pk = d["processor_kwargs"]  # Gr00tN1d7Processor.from_pretrained backfills processor_kwargs["model_name"] with the gated id
print(f"processor_config.json: processor_kwargs.model_name {pk.get('model_name')!r} -> {bb!r}")
pk["model_name"] = bb
json.dump(d, open(f"{dst}/processor_config.json", "w"), indent=2)
EOF
ls -la $DST | awk '{print $5, $9, $10, $11}'
