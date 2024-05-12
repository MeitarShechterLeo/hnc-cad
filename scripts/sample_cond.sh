#!/bin/bash\

python gen/ac_gen_pc.py --output result/ac_pc_only_encoder --weight /home/meitars/Code/hnc-cad/proj_log/gen_full_pc_only_encoder \
                --solid_code pretrained/solid_test.pkl --profile_code  pretrained/profile_test.pkl --loop_code pretrained/loop_test.pkl --mode cond --device 0
                # --solid_code pretrained/solid.pkl --profile_code  pretrained/profile.pkl --loop_code pretrained/loop.pkl --mode cond --device 0

# convert obj format to stl & step
python gen/convert.py --data_folder result/ac_pc_only_encoder

# # visualize CAD 
# python gen/cad_img.py --input_dir result/ac --output_dir  result/ac_visual