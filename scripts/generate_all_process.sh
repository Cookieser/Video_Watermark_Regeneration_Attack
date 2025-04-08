source ~/anaconda3/etc/profile.d/conda.sh
conda activate regeneration-attack
cd ..


GPUS=0

NAME=beauty_1
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

python train.py --root_dir $ROOT_DIRECTORY \
                --model_save_path $MODEL_SAVE_PATH \
                --log_save_path $LOG_SAVE_PATH  \
                --mask_dir $MASK_DIRECTORY \
                --flow_dir $FLOW_DIRECTORY \
                --gpus $GPUS \
                --encode_w --annealed \
                --config configs/${NAME}/${EXP_NAME}.yaml \
                --exp_name ${EXP_NAME}



python train.py --test --encode_w \
                --root_dir $ROOT_DIRECTORY \
                --log_save_path $LOG_SAVE_PATH \
                --mask_dir $MASK_DIRECTORY \
                --weight_path $WEIGHT_PATH \
                --gpus $GPUS \
                --config configs/${NAME}/${EXP_NAME}.yaml \
                --exp_name ${EXP_NAME} \
                --save_deform False

python generate_control_image.py \
  --prompt "portrait of a beautiful woman near the water, in oil painting style, Rembrandt lighting, textured brush strokes, soft skin, deep shadows" \
  --image_dir  $ORIGINAL_DIR \
  --output_dir $CANONICAL_DIR


python train.py --test --encode_w \
                --root_dir $ROOT_DIRECTORY \
                --log_save_path $LOG_SAVE_PATH \
                --mask_dir $MASK_DIRECTORY \
                --weight_path $WEIGHT_PATH \
                --gpus $GPUS \
                --canonical_dir $CANONICAL_DIR \
                --config configs/${NAME}/${EXP_NAME}.yaml \
                --exp_name $EXP_NAME