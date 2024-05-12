import os
import torch
import argparse
from tqdm import tqdm
from config import *
from dataset import CADPCData
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.tensorboard import SummaryWriter
from model.network import schedule_with_warmup
from model.encoder import PointNet2_SEG
from model.decoder import SketchDecoder, ExtDecoder, CodeDecoder



def train(args):
    # gpu device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda:0")

    # Initialize dataset loader
    dataset = CADPCData(PC_PATH, CAD_TRAIN_PATH, args.solid_code, args.profile_code, args.loop_code, args.mode, is_training=True)
    code_size = dataset.solid_unique_num + dataset.profile_unique_num + dataset.loop_unique_num
    dataloader = torch.utils.data.DataLoader(dataset, 
                                             shuffle=True, 
                                             batch_size=args.batchsize,
                                             num_workers=6)
    
    # Initialize models    
    pc_enc = PointNet2_SEG()
    pc_enc = nn.DataParallel(pc_enc)
    pc_enc = pc_enc.to(device).train()

    sketch_dec = SketchDecoder(args.mode, num_code=code_size) 
    # load pretrained
    # sketch_dec_sd = torch.load(os.path.join(args.cad_weight, 'sketch_dec_epoch_350.pt'))
    # keys_to_load = [key for key in sketch_dec_sd.keys() if "mempos_embed" not in key]
    # sketch_dec_subset_sd = {key: value for key, value in sketch_dec_sd.items() if key in keys_to_load}
    # sketch_dec.load_state_dict(sketch_dec_subset_sd, strict=False)
    sketch_dec.load_state_dict(torch.load(os.path.join(args.weight, 'sketch_dec_epoch_250.pt')))
    # prepare for training
    sketch_dec = nn.DataParallel(sketch_dec)
    sketch_dec = sketch_dec.to(device).eval()
    # sketch_dec.module.mempos_embed = sketch_dec.module.mempos_embed.train() # train new weights

    ext_dec = ExtDecoder(args.mode, num_code=code_size) 
     # load pretrained
    # ext_dec_sd = torch.load(os.path.join(args.cad_weight, 'ext_dec_epoch_350.pt'))
    # keys_to_load = [key for key in ext_dec_sd.keys() if "mempos_embed" not in key]
    # ext_dec_subset_sd = {key: value for key, value in ext_dec_sd.items() if key in keys_to_load}
    # ext_dec.load_state_dict(ext_dec_subset_sd, strict=False)
    ext_dec.load_state_dict(torch.load(os.path.join(args.weight, 'ext_dec_epoch_250.pt')))
    # prepare for training
    ext_dec = nn.DataParallel(ext_dec)
    ext_dec = ext_dec.to(device).eval()
    # ext_dec.module.mempos_embed = ext_dec.module.mempos_embed.train() # train new weights

    code_dec = CodeDecoder(args.mode, code_size)
    code_dec.load_state_dict(torch.load(os.path.join(args.weight, 'code_dec_epoch_250.pt')))
    code_dec = nn.DataParallel(code_dec)
    code_dec = code_dec.to(device).eval()
    # code_dec.to(device).train()

    # params = list(pc_enc.parameters()) + \
    #     list(sketch_dec.module.mempos_embed.parameters()) + list(ext_dec.module.mempos_embed.parameters()) + list(code_dec.parameters())
    params = list(pc_enc.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    scheduler = schedule_with_warmup(optimizer, 2000)
    writer = SummaryWriter(log_dir=args.output)
    
    # Main training loop
    iters = 0
    print('Start training...')
    for epoch in range(COND_TRAIN_EPOCH):  
        progress_bar = tqdm(total=len(dataloader))
        progress_bar.set_description(f"Epoch {epoch}")
  
        for pc, pixel, coord, sketch_mask, ext, ext_mask, code, code_mask, name in dataloader:
            pc = pc.to(device)
            pixel = pixel.to(device)
            coord = coord.to(device)
            sketch_mask = sketch_mask.to(device)
            ext = ext.to(device)
            ext_mask = ext_mask.to(device)
            code = code.to(device)
            code_mask = code_mask.to(device)

            # Partial Token Encoder
            latent_z = pc_enc(pc) # (B, S=200+56, embd_dim=256)
            latent_mask = torch.zeros(latent_z.shape[:-1], dtype=torch.bool)

            # Pass through sketch decoder
            sketch_logits = sketch_dec(pixel[:, :-1], coord[:, :-1, :], code, code_mask, latent_z, latent_mask)
            
            # Pass through extrude decoder
            ext_logits = ext_dec(ext[:, :-1], code, code_mask, latent_z, latent_mask)

            # Pass through code decoder
            code_logits = code_dec(code[:, :-1], latent_z, latent_mask)

            # Compute losses
            valid_mask =  (~sketch_mask).reshape(-1) 
            sketch_pred = sketch_logits.reshape(-1, sketch_logits.shape[-1]) 
            sketch_gt = pixel.reshape(-1)
            sketch_loss = F.cross_entropy(sketch_pred[valid_mask], sketch_gt[valid_mask])     

            valid_mask =  (~ext_mask).reshape(-1) 
            ext_pred = ext_logits.reshape(-1, ext_logits.shape[-1]) 
            ext_gt = ext.reshape(-1)
            ext_loss = F.cross_entropy(ext_pred[valid_mask], ext_gt[valid_mask])     

            valid_mask =  (~code_mask).reshape(-1) 
            code_pred = code_logits.reshape(-1, code_logits.shape[-1]) 
            code_gt = code.reshape(-1)
            code_loss = F.cross_entropy(code_pred[valid_mask], code_gt[valid_mask])    

            total_loss = sketch_loss + ext_loss + code_loss
            
            # logging
            if iters % 10 == 0:
                writer.add_scalar("Loss/Total", total_loss, iters) 
                writer.add_scalar("Loss/sketch", sketch_loss, iters)
                writer.add_scalar("Loss/ext", ext_loss, iters)
                writer.add_scalar("Loss/code", code_loss, iters)
                
            # Update model
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(params, max_norm=1.0)  
            optimizer.step()
            scheduler.step()  # linear warm up to 1e-3
            iters += 1
            progress_bar.set_postfix({'loss': total_loss.cpu().item()})
            progress_bar.update(1)

        progress_bar.close()
        writer.flush()

        # # save model after n epoch
        if (epoch+1) % 10 == 0:
            torch.save(sketch_dec.module.state_dict(), os.path.join(args.output,'sketch_dec_epoch_'+str(epoch+1)+'.pt'))
            torch.save(ext_dec.module.state_dict(), os.path.join(args.output,'ext_dec_epoch_'+str(epoch+1)+'.pt'))
            torch.save(pc_enc.module.state_dict(), os.path.join(args.output,'pc_enc_epoch_'+str(epoch+1)+'.pt'))
            torch.save(code_dec.module.state_dict(), os.path.join(args.output,'code_dec_epoch_'+str(epoch+1)+'.pt'))
            
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, help="Output folder to save the data", required=True)
    parser.add_argument("--batchsize", type=int, help="Training batchsize", required=True)
    parser.add_argument("--device", type=str, help="CUDA device", required=True)
    parser.add_argument("--solid_code", type=str, required=True, help='Extracted solid codes')
    parser.add_argument("--profile_code", type=str, required=True, help='Extracted profile codes')
    parser.add_argument("--loop_code", type=str, required=True, help='Extracted loop codes')
    parser.add_argument("--mode", type=str, required=True, help='uncond | cond')
    parser.add_argument("--weight", type=str, help="Pretrained model", required=True)
    # parser.add_argument("--code_weight", type=str, help="Pretrained code-tree model", required=True)
    # parser.add_argument("--cad_weight", type=str, help="Pretrained CAD model", required=True)
    args = parser.parse_args()

    # Create training folder
    result_folder = args.output
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
        
    # Start training 
    train(args)