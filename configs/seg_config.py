_base_ = 'mask-rcnn_swin-s-p4-w7_fpn_amp-ms-crop-3x_coco.py'

model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=1
        ),
        mask_head=dict(
            num_classes=1
        )
    )
)

metainfo = dict(
    classes=('object',)
)

backend_args = None

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

test_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        pipeline=test_pipeline
    )
)