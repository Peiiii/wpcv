from wpcv.utils.data_aug.det_aug import *
from wpcv.utils.data_aug import *
from wpcv.utils.data_aug import img_aug
from wpcv.utils.ops import pil_ops
from PIL import Image
import wpcv
def demo():
    transform = Compose([
        # BboxToPoints(),
        Limitsize(600),
        Zip([
            Compose([
                # img_aug.RandomApply(pil_ops.equalize,p=1),
            ]),
            lambda x: x,
        ]),
        # RandomPerspective([0.8,1]),
        RandomRotate(30, fillcolor='black'),
        RandomShear(30, 30, fillcolor='black'),
        RandomTranslate(max_offset=[100, 100], fillcolor='black'),
        # RandomHorizontalFlip(),
        # RandomVerticalFlip(),

        Zip([
            Compose([
                img_aug.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                img_aug.RandomOrder([
                    img_aug.RandomApply(pil_ops.sp_noise, p=1),

                    # img_aug.RandomApply(pil_ops.gaussian_noise,p=0.2),
                    # img_aug.RandomApply(pil_ops.blur,p=0.2),
                    # img_aug.RandomApply(pil_ops.box_blur,p=1,radius=2),
                    # img_aug.RandomApply(pil_ops.model_filter,p=1),
                    # img_aug.RandomChoice([
                    # img_aug.RandomApply(pil_ops.edge, p=1),
                    # img_aug.RandomApply(pil_ops.edge_enhance,p=1),
                    # img_aug.RandomApply(pil_ops.edge_enhance_more,p=1),
                    # ]),
                    # img_aug.RandomApply(pil_ops.contour,p=1),
                    # img_aug.RandomApply(pil_ops.emboss,p=1),
                    # img_aug.RandomApply(pil_ops.equalize,p=1)
                ])
            ]),
            lambda x: x
        ]),
        Resize((512, 512), keep_ratio=True, fillcolor='black'),
        # Limitsize((128,1024)),
    ])
    img = Image.open('/home/ars/sda5/data/chaoyuan/datasets/detect_datasets/行驶证/train/101.jpg').convert('RGB')

    polygons = [[
        [
            679.0645161290323,
            739.6129032258065
        ],
        [
            1295.1935483870968,
            770.258064516129
        ],
        [
            1277.4516129032259,
            1186.3870967741937
        ],
        [
            661.3225806451613,
            1163.8064516129032
        ]
    ],
        [
            [
                583.9032258064516,
                225.09677419354838
            ],
            [
                1196.8064516129032,
                208.96774193548387
            ],
            [
                1214.5483870967741,
                625.0967741935484
            ],
            [
                601.6451612903226,
                646.0645161290323
            ]
        ]
    ]
    img, polygons = transform(img, polygons)
    # print(points)
    img = wpcv.draw_polygons(img, polygons, width=3)
    img.show()


# print(img.size, box)


if __name__ == '__main__':
    demo()
