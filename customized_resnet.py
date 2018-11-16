from tensorflow.contrib.slim import nets

resnet_v1_block = nets.resnet_v1.resnet_v1_block

def resnet_v1_20(inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 spatial_squeeze=True,
                 store_non_strided_activations=False,
                 reuse=None,
                 scope='resnet_v1_20'):
    """ResNet-20 model. See resnet_v1() for arg and return description."""
    blocks = [
        resnet_v1_block('block1', base_depth=64, num_units=1, stride=2),
        resnet_v1_block('block2', base_depth=128, num_units=1, stride=2),
        resnet_v1_block('block3', base_depth=256, num_units=1, stride=2),
        resnet_v1_block('block4', base_depth=512, num_units=3, stride=1)
    ]
    return nets.resnet_v1.resnet_v1(
        inputs,
        blocks,
        num_classes,
        is_training,
        global_pool=global_pool,
        output_stride=output_stride,
        include_root_block=True,
        reuse=reuse,
        scope=scope)
