source ~/anaconda3/etc/profile.d/conda.sh
conda activate regeneration-attack


GPUS=0

NAME=test
EXP_NAME=base

ROOT_DIRECTORY="all_sequences/$NAME/$NAME"
MODEL_SAVE_PATH="ckpts/all_sequences/$NAME"
LOG_SAVE_PATH="logs/all_sequences/$NAME"

ORIGINAL_DIR="results/all_sequences/$NAME/${EXP_NAME}" 

MASK_DIRECTORY="all_sequences/$NAME/${NAME}_masks_0 all_sequences/$NAME/${NAME}_masks_1"
FLOW_DIRECTORY="all_sequences/$NAME/${NAME}_flow"

# WEIGHT_PATH=ckpts/all_sequences/$NAME/${EXP_NAME}/${NAME}.ckpt
WEIGHT_PATH=ckpts/all_sequences/$NAME/${EXP_NAME}/step=10000.ckpt

CANONICAL_DIR="all_sequences/${NAME}/${EXP_NAME}_control" 

CONFIG_FILE="configs/$NAME/${EXP_NAME}.yaml"     
VIDEO_PATH="videos/${NAME}.mp4"             


# ==== Read img_wh and fps from config ====
get_yaml_val() {
  local key=$1
  grep "$key" "$CONFIG_FILE" | grep -v "#" | awk -F ': ' '{print $2}' | tr -d '[],'
}

IMG_WH=$(get_yaml_val "img_wh")
FPS=$(get_yaml_val "fps")

WIDTH=$(echo $IMG_WH | cut -d ' ' -f1)
HEIGHT=$(echo $IMG_WH | cut -d ' ' -f2)

# ==== Check if values are valid ====
if [[ -z "$WIDTH" || -z "$HEIGHT" ]]; then
  echo "❌ Failed to extract img_wh from config"
  exit 1
fi
if [[ -z "$FPS" ]]; then
  echo "❌ Failed to extract fps from config"
  exit 1
fi

echo "✔️ Parsed config: WIDTH=$WIDTH HEIGHT=$HEIGHT FPS=$FPS"

# ==== Create output directory ====
mkdir -p "$ROOT_DIRECTORY"

# ==== Step 1: Extract frames from video and resize ====
echo "🎞️ Step 1: Extracting frames from video..."
ffmpeg -i "$VIDEO_PATH" -vf "scale=${WIDTH}:${HEIGHT}" "$ROOT_DIRECTORY/%05d.png" -hide_banner




echo "🔁 Step 2: Training the model..."
python train.py --root_dir $ROOT_DIRECTORY \
                --model_save_path $MODEL_SAVE_PATH \
                --log_save_path $LOG_SAVE_PATH  \
                --mask_dir $MASK_DIRECTORY \
                --flow_dir $FLOW_DIRECTORY \
                --gpus $GPUS \
                --encode_w --annealed \
                --config configs/${NAME}/${EXP_NAME}.yaml \
                --exp_name ${EXP_NAME}


if [ ! -f "$WEIGHT_PATH" ]; then
  echo "❌ ERROR: Checkpoint not found at $WEIGHT_PATH"
  echo "🛑 Exiting script."
  exit 1
else
  echo "✅ Found checkpoint: $WEIGHT_PATH"
fi




echo "🧪 Step 3: Test and get the canonical image..."
python train.py --test --encode_w \
                --root_dir $ROOT_DIRECTORY \
                --log_save_path $LOG_SAVE_PATH \
                --mask_dir $MASK_DIRECTORY \
                --weight_path $WEIGHT_PATH \
                --gpus $GPUS \
                --config configs/${NAME}/${EXP_NAME}.yaml \
                --exp_name ${EXP_NAME} \
                --save_deform False

canonical_count=$(ls "$ORIGINAL_DIR"/canonical_*.png 2>/dev/null | wc -l)

if [ "$canonical_count" -eq 0 ]; then
  echo "❌ ERROR: No canonical_*.png images found in $ORIGINAL_DIR"
  echo "🛑 Exiting script."
  exit 1
else
  echo "✅ Found $canonical_count canonical image(s) in $ORIGINAL_DIR"
fi




echo "🎨 Step 4: Generate stylized canonical images using ControlNet..."
python generate_control_image.py \
  --prompt "portrait of a beautiful woman near the water, in oil painting style, Rembrandt lighting, textured brush strokes, soft skin, deep shadows" \
  --image_dir  $ORIGINAL_DIR \
  --output_dir $CANONICAL_DIR


echo "🧪 Step 5: Get the video with stylized canonical image..."
python train.py --test --encode_w \
                --root_dir $ROOT_DIRECTORY \
                --log_save_path $LOG_SAVE_PATH \
                --mask_dir $MASK_DIRECTORY \
                --weight_path $WEIGHT_PATH \
                --gpus $GPUS \
                --canonical_dir $CANONICAL_DIR \
                --config configs/${NAME}/${EXP_NAME}.yaml \
                --exp_name $EXP_NAME