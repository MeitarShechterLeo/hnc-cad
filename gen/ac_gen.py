import os
import torch
import argparse
from gen.config import *
from hashlib import sha256
import numpy as np 
from gen.dataset import CADData
from gen.utils import CADparser, write_obj_sample
from gen.model.encoder import SketchEncoder, ExtEncoder
from gen.model.decoder import SketchDecoder, ExtDecoder, CodeDecoder


def raster_cad(coord, ext): 
    parser = CADparser(CAD_BIT)
    parsed_data = parser.perform(coord, ext)
    return parsed_data
    

def pad_code(total_code):
    keys = np.ones(len(total_code))
    padding = np.zeros(MAX_CODE-len(total_code)).astype(int)  
    total_code = np.concatenate([total_code, padding], axis=0)
    seq_mask = 1-np.concatenate([keys, padding]) == 1   
    return total_code, seq_mask


def hash_sketch(sketch, ext):
    hash_str = sha256(np.ascontiguousarray(sketch).flatten()).hexdigest() +'_'+\
        sha256(np.ascontiguousarray(ext).flatten()).hexdigest()
    return hash_str


@torch.inference_mode()
def sample(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    dataset = CADData(CAD_TRAIN_PATH, args.solid_code, args.profile_code, args.loop_code, args.mode, is_training=False)
    dataloader = torch.utils.data.DataLoader(dataset, 
                                             shuffle=False, 
                                             batch_size=1,
                                             num_workers=1)
    code_size = dataset.solid_unique_num + dataset.profile_unique_num + dataset.loop_unique_num
   
    # Load model weights
    sketch_enc = SketchEncoder()
    sketch_enc.load_state_dict(torch.load(os.path.join(args.weight, 'sketch_enc_epoch_250.pt')))
    sketch_enc.cuda().eval()

    sketch_dec = SketchDecoder(args.mode, num_code=code_size) 
    sketch_dec.load_state_dict(torch.load(os.path.join(args.weight, 'sketch_dec_epoch_250.pt')))
    sketch_dec.cuda().eval()

    ext_enc = ExtEncoder()
    ext_enc.load_state_dict(torch.load(os.path.join(args.weight, 'ext_enc_epoch_250.pt')))
    ext_enc.cuda().eval()

    ext_dec = ExtDecoder(args.mode, num_code=code_size) 
    ext_dec.load_state_dict(torch.load(os.path.join(args.weight, 'ext_dec_epoch_250.pt')))
    ext_dec.cuda().eval()

    code_dec = CodeDecoder(args.mode, code_size)
    code_dec.load_state_dict(torch.load(os.path.join(args.weight, 'code_dec_epoch_250.pt')))
    code_dec.cuda().eval()

    # Random sampling 
    code_bsz = 50 # every partial input samples this many neural codes
    count = 0
    for pixel_p, coord_p, sketch_mask_p, ext_p, ext_mask_p, _, _, _, _, _, _, _ in dataloader:
        if count>50:break # only visualize the first 50 examples
        pixel_p = pixel_p.cuda()
        coord_p = coord_p.cuda()
        sketch_mask_p = sketch_mask_p.cuda()
        ext_p = ext_p.cuda()
        ext_mask_p = ext_mask_p.cuda()

        # encode partial CAD model
        latent_sketch = sketch_enc(pixel_p, coord_p, sketch_mask_p)
        latent_extrude = ext_enc(ext_p, ext_mask_p)

        # generate the neural code tree
        latent_z = torch.cat([latent_sketch, latent_extrude], 1)
        latent_mask = torch.cat([sketch_mask_p, ext_mask_p], 1)
        code_sample = code_dec.sample(n_samples=code_bsz, latent_z=latent_z.repeat(code_bsz, 1, 1), 
                                      latent_mask=latent_mask.repeat(code_bsz, 1), top_k=0, top_p=0.95)
        
        # filter code, only keep unique code
        if len(code_sample)<3:
            continue 
        code_unique = {}
        for ii in range(len(code_sample)):
            if len(torch.where(code_sample[ii]==0)[0])==0:
                continue
            code = (code_sample[ii][:torch.where(code_sample[ii]==0)[0][0]+1]).detach().cpu().numpy()
            code_uid = code.tobytes()
            if code_uid not in code_unique:
                code_unique[code_uid] = code
        total_code = []
        total_code_mask = []
        for _, code in code_unique.items():
            _code_, _code_mask_ = dataset.pad_code(code)
            total_code.append(_code_)
            total_code_mask.append(_code_mask_)
        total_code = np.vstack(np.vstack(total_code))
        total_code_mask = np.vstack(total_code_mask)
        total_code = torch.LongTensor(total_code).cuda()
        total_code_mask = torch.BoolTensor(total_code_mask).cuda()

        # generate the full CAD model
        latent_z = latent_z.repeat(len(total_code), 1, 1)
        latent_mask = latent_mask.repeat(len(total_code), 1)
        xy_samples, _code_, _code_mask_, _latent_z_, _latent_mask_ = sketch_dec.sample(total_code, total_code_mask, latent_z, latent_mask, top_k=1, top_p=0)
        cad_samples = ext_dec.sample(xy_samples, _code_, _code_mask_, _latent_z_, _latent_mask_, top_k=1, top_p=0)
       
        # raster user input cad
        try:
            cad_obj = raster_cad(coord_p.detach().cpu().numpy()[0], ext_p.detach().cpu().numpy()[0])
            save_folder = os.path.join(result_folder, str(count).zfill(4)+'_origInput')
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            write_obj_sample(save_folder, cad_obj)
        except Exception as error_msg:  
            continue 

        # raster auto-completed cad
        for ii, sample in enumerate(cad_samples):
            try:
                cad_obj = raster_cad(sample[0], sample[1])
                save_folder = os.path.join(result_folder, str(count).zfill(4)+'_postAC'+str(ii))
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                write_obj_sample(save_folder, cad_obj)
            except Exception as error_msg:  
                continue
        count += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=str, help="Pretrained CAD model", required=True)
    parser.add_argument("--output", type=str, help="Output folder to save the data", required=True)
    parser.add_argument("--device", type=str, help="CUDA Device Index", required=True)
    parser.add_argument("--mode", type=str, required=True, help="eval | sample")
    parser.add_argument("--solid_code", type=str, required=True)
    parser.add_argument("--profile_code", type=str, required=True)
    parser.add_argument("--loop_code", type=str, required=True)
    args = parser.parse_args()

    result_folder = args.output
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
        
    sample(args)
