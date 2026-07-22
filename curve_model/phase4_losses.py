"""Partial-label Phase 4 losses; absent labels contribute exactly zero."""
from __future__ import annotations
from .phase2_losses import CurvePhase2Loss, _masked_bce_with_logits, _masked_dice_loss, masked_direction_loss, cape_connectivity_loss
from .losses import torch, F, skeleton_recall_loss

class CurvePhase4Loss(CurvePhase2Loss):
    def forward(self, outputs, targets, epoch=0):
        availability=targets.get("label_available", {})
        safe=dict(targets)
        batch=outputs["centerline_logits"].shape[0]
        for name,key,reference in (("stroke","stroke_mask","stroke_logits"),("grid","grid_mask","grid_logits")):
            available=availability.get(name)
            if available is not None: safe[name+"_label_valid"]=available.reshape(batch,1,1,1)
            if key not in safe: safe[key]=torch.zeros_like(outputs[reference])
        center_available=availability.get("centerline",torch.ones(batch,device=outputs["centerline_logits"].device))
        center_valid=center_available.to(device=outputs["centerline_logits"].device,dtype=outputs["centerline_logits"].dtype).reshape(batch,1,1,1)
        if "centerline_mask" not in safe: safe["centerline_mask"]=torch.zeros_like(outputs["centerline_logits"])
        if "distance_field" not in safe: safe["distance_field"]=torch.zeros_like(outputs["distance_field"])
        if "direction_field" not in safe: safe["direction_field"]=torch.zeros_like(outputs["direction"])
        if "valid_direction_mask" not in safe: safe["valid_direction_mask"]=torch.zeros_like(outputs["centerline_logits"])
        parts=super().forward(outputs,safe,epoch)
        weights=self.weights
        pos=torch.tensor(weights.positive_weight,device=outputs["centerline_logits"].device,dtype=outputs["centerline_logits"].dtype)
        parts["centerline_bce"]=_masked_bce_with_logits(outputs["centerline_logits"],safe["centerline_mask"],center_valid,pos)
        parts["centerline_dice"]=_masked_dice_loss(outputs["centerline_logits"],safe["centerline_mask"],center_valid)
        selected=center_available > 0
        parts["skeleton_recall"]=(skeleton_recall_loss(outputs["stroke_logits"][selected],safe["centerline_mask"][selected])
            if bool(torch.any(selected)) else outputs["centerline_logits"].sum()*0)
        distance_weights=(float(weights.distance_base_weight)+float(weights.distance_center_bonus)*safe["distance_field"])*center_valid
        parts["distance"]=(F.smooth_l1_loss(outputs["distance_field"],safe["distance_field"],reduction="none")*distance_weights).sum()/distance_weights.sum().clamp_min(1)
        parts["direction"]=masked_direction_loss(outputs["direction"],safe["direction_field"],safe["valid_direction_mask"]*center_valid,safe["centerline_mask"],weights.direction_center_bonus)
        parts["cape"]=(cape_connectivity_loss(outputs["centerline_logits"][selected],safe["centerline_mask"][selected],self.cape.window_size,self.cape.dilation_radius)
            if parts["cape_active"] and bool(torch.any(selected)) else outputs["centerline_logits"].sum()*0)
        wrap_available=availability.get("wrap")
        if wrap_available is None or "wrap_target" not in safe: wrap_loss=outputs["wrap_logits"].sum()*0
        else:
            raw=F.cross_entropy(outputs["wrap_logits"],safe["wrap_target"].long(),weight=torch.tensor([1.,8.,8.],device=outputs["wrap_logits"].device),reduction="none")
            mask=wrap_available.reshape(batch,1).to(raw.dtype); wrap_loss=(raw*mask).sum()/mask.expand_as(raw).sum().clamp_min(1)
        parts["wrap"]=wrap_loss
        parts["total"]=(weights.stroke_bce*parts["stroke_bce"]+weights.stroke_dice*parts["stroke_dice"]
            +weights.centerline_bce*parts["centerline_bce"]+weights.centerline_dice*parts["centerline_dice"]
            +weights.skeleton_recall*parts["skeleton_recall"]+weights.distance*parts["distance"]
            +weights.direction*parts["direction"]+weights.grid*parts["grid"]
            +(self.cape.weight*parts["cape"] if parts["cape_active"] else 0.0)+wrap_loss)
        return parts
