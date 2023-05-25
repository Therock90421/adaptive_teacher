import torch
a = torch.load('City2Fog.pth')
#b = torch.load('/lustre/S/lihaochen/RegionCLIP/pretrained_ckpt/regionclip/regionclip_finetuned-coco_rn50.pth')

#print(a['state_dict']['clip_model.visual.conv1.weight'])
# for key in a['model'].keys():
#     for key2 in b['model'].keys():
#         if key == key2:
#             print(key)
#             print(torch.equal(a['model'][key], b['model'][key]))
#             if not torch.equal(a['model'][key], b['model'][key]):
#                 print(a['model'][key])
#                 print(b['model'][key])
#                 print(a['model'][key].shape)
#                 print(b['model'][key].shape)
print('####################')
for key in a['model'].keys():
    print(key)
#print(torch.equal(a['state_dict']['DAPromptHead.prompt_learner.ctx_di'],b['state_dict']['DAPromptHead.prompt_learner.ctx_di']))
#print(torch.equal(a['state_dict']['DAPromptHead.prompt_learner.token_prefix'],b['state_dict']['DAPromptHead.prompt_learner.token_prefix']))
#print(torch.equal(a['state_dict']['DAPromptHead.text_encoder.transformer.resblocks.1.mlp.c_fc.weight'],b['state_dict']['DAPromptHead.text_encoder.transformer.resblocks.1.mlp.c_fc.weight']))
#print(torch.equal(a['state_dict']['roi_head.bbox_head.fc_clip.weight'],b['state_dict']['roi_head.bbox_head.fc_clip.weight']))
#print(torch.equal(a['state_dict']['roi_head.bbox_head.logit_scale'],b['state_dict']['roi_head.bbox_head.logit_scale']))





