#!/bin/bash\

# sample code-tree & CAD model (visualization)
python gen/rand_gen.py --code_weight pretrained/gen_code  --cad_weight pretrained/gen_cad --output result/random_sample \
    --solid_code pretrained/solid.pkl --profile_code  pretrained/profile.pkl --loop_code pretrained/loop.pkl --mode sample --device 0

# # sample code-tree & CAD model (evaluation, slow)
# python gen/rand_gen.py --code_weight proj_log/gen_code  --cad_weight proj_log/gen_cad --output result/random_eval \
#     --solid_code solid.pkl --profile_code  profile.pkl --loop_code loop.pkl --mode eval --device 1
# python gen/convert.py --data_folder result/random_eval

# convert obj format to stl & step
python gen/convert.py --data_folder result/random_sample

# # visualize CAD 
# python gen/cad_img.py --input_dir result/random_sample --output_dir  result/random_sample

