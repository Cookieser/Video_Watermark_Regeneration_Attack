# Regeneration Attack - Pipeline 🎞️🎨

本项目基于CoDeF实现了一种 **Vedio Regeneration Attack** 流程，封装了从原始视频输入到风格化视频输出的全流程。用户仅需准备视频与配置文件，即可一键生成编辑后的视频。注意：该生成方法需要针对每个视频进行专门的训练。

---

## 🧩 Pipeline 流程

1. **帧提取与缩放**
2. **模型训练**
3. **生成 canonical image**
4. **使用 ControlNet 进行风格化**
5. **合成最终视频**

---

## 🔧 环境准备

```bash
conda create -n regeneration-attack python=3.10 -y

conda activate regeneration-attack

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda install -c conda-forge "ffmpeg>=4.4"
pip install -r requirements.txt

pip install huggingface-hub pyparsing pytz transformers

pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

pip install diffusers
```



## 🚀 使用方法

#### 1. 放置输入视频

将你要编辑的视频放入 `videos/` 文件夹，例如：

```
videos/test.mp4
```

#### 2. 创建配置文件

创建对应的 YAML 配置文件

路径示例：`configs/test/base.yaml`

```
mask_dir: null
flow_dir: null

img_wh: [540, 540]
canonical_wh: [640, 640]

lr: 0.001
bg_loss: 0.003

ref_idx: null # 0

N_xyz_w: [8,]
flow_loss: 1
flow_step: -1
self_bg: True

deform_hash: True
vid_hash: True

num_steps: 10000
decay_step: [2500, 5000, 7500]
annealed_begin_step: 4000
annealed_step: 4000
save_model_iters: 2000

fps: 15

```

#### 3. 运行流程脚本

直接运行该脚本自动完成生成流程

```
bash ./scripts/generate_all_process.sh
```

------



## 📌 具体流程概览

#### 🖼️ Step 1：视频帧提取

使用 `ffmpeg` 将视频拆帧并缩放至指定分辨率。

#### 🔁 Step 2：模型训练

训练基于掩码和光流信息的视频表示模型。

#### 🧪 Step 3：生成 canonical 图像

提取 canonical 表示图像。

#### 🎨 Step 4：使用 ControlNet 风格化

基于 prompt 对 canonical 图像进行风格迁移，例如：

> “油画风格的女子肖像，水边，伦勃朗式光照，柔和皮肤，高对比明暗，质感笔触”

#### 🧪 Step 5：渲染最终视频

将风格化的 canonical 图像用于重建视频。

------



## 📂 输出位置

最终生成的视频和图片结果将保存在`results/all_sequences/{NAME}/{EXP_NAME}_transformed/`

------



## ⚠️ 注意事项

- 请确保以下路径下的数据已准备好（如未生成，请使用其他工具预处理）：
  - 光流文件夹：`all_sequences/{NAME}/{NAME}_flow/`
  - 掩码文件夹：`all_sequences/{NAME}/{NAME}_masks_0/` 和 `..._masks_1/`
- 如果找不到模型权重，脚本会自动提示错误并终止。





## 可能遇见的问题

```bash
ffmpeg -version
ffmpeg version 4.3 Copyright (c) 2000-2020 the FFmpeg developers
built with gcc 7.3.0 (crosstool-NG 1.23.0.449-a04d0)
configuration: --prefix=/home/yw699/anaconda3/envs/regeneration-attack --cc=/opt/conda/conda-bld/ffmpeg_1597178665428/_build_env/bin/x86_64-conda_cos6-linux-gnu-cc --disable-doc --disable-openssl --enable-avresample --enable-gnutls --enable-hardcoded-tables --enable-libfreetype --enable-libopenh264 --enable-pic --enable-pthreads --enable-shared --disable-static --enable-version3 --enable-zlib --enable-libmp3lame
libavutil      56. 51.100 / 56. 51.100
libavcodec     58. 91.100 / 58. 91.100
libavformat    58. 45.100 / 58. 45.100
libavdevice    58. 10.100 / 58. 10.100
libavfilter     7. 85.100 /  7. 85.100
libavresample   4.  0.  0 /  4.  0.  0
libswscale      5.  7.100 /  5.  7.100
libswresample   3.  7.100 /  3.  7.100
```

这是一个老版本，带 `--enable-libopenh264` 但不支持支持 `libx264`：

这个会导致如下报错

![image](https://pic-1306483575.cos.ap-nanjing.myqcloud.com/image.png)

验证正常使用：

```bash
ffmpeg -codecs | grep libx264

cat frame.raw | ffmpeg -y -f rawvideo -pix_fmt rgb24 -s 540x540 -i - -r 15 -vcodec libx264 -crf 1 -pix_fmt yuv420p out.mp4
```

### 解决

```python
conda install -c conda-forge "ffmpeg>=4.4"
```