import torch.nn as nn
import torch.nn.functional as F


class HNCLoss(nn.Module):
    def forward(self, output):
        pixel = output['pixel']
        sketch_mask = output['sketch_mask']
        sketch_logits = output['sketch_logits']

        ext = output['ext']
        ext_mask = output['ext_mask']
        ext_logits = output['ext_logits']

        code = output['code']
        code_mask = output['code_mask']
        code_logits = output['code_logits']

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

        # total_loss = sketch_loss + ext_loss + code_loss

        res = {
            # "loss": total_loss,
            "sketch_loss": sketch_loss, 
            "ext_loss": ext_loss, 
            "code_loss": code_loss
            }

        return res
