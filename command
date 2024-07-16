source activate smirk

CUDA_VISIBLE_DEVICES=3  python demo.py \
    --input_path samples/test_image2.png \
    --out_path results/ \
    --checkpoint pretrained_models/SMIRK_em1.pt \
    --crop \
    --render_orig

CUDA_VISIBLE_DEVICES=3  python demo_video.py \
    --input_path samples/dafoe.mp4 \
    --out_path results/ \
    --checkpoint pretrained_models/SMIRK_em1.pt \
    --crop \
    --render_orig

CUDA_VISIBLE_DEVICES=3  python infer_mead.py \
    --checkpoint pretrained_models/SMIRK_em1.pt \
    --crop \
    --dataset_json dataset_.json

CUDA_VISIBLE_DEVICES=2  python render_mead.py \
    --param_path MEAD/W009/param/happy/level_3/001 \
    --vid_path MEAD/W009/video/happy/level_3/001.mp4 \
    --out_path results/ 

CUDA_VISIBLE_DEVICES=2  python render_mimic.py \
    --param_path /data/chenziang/codes/Mimic/HDTF-3D/spectre_processed_25fps_16kHz/WRA_VickyHartzler_000/verts_new_shape1.npy \
    --out_path results/ 


CUDA_VISIBLE_DEVICES=2  python ./render/render_template.py \
    --template_path /data/chenziang/codes/Mimic/HDTF-3D/templates.pkl \
    --out_path results/ 

CUDA_VISIBLE_DEVICES=3  python infer_mead.py \
    --checkpoint pretrained_models/SMIRK_em1.pt \
    --crop \
    --dataset_json dataset_w.json


ffmpeg -i 027.mp4 -i 027.m4a -vcodec copy -acodec copy output.mp4
ffmpeg -i 027.mp4 -vcodec copy -t 4.5 -c copy output.mp4
ffmpeg -i 027.m4a -acodec copy -t 4.6 -c copy output.m4a
ffmpeg -i input.mp3 -ss 00:00:30 -t 00:00:10 -c copy output.mp3



