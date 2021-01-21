from layoutparser.elements import *
from layoutparser.ocr import *
from layoutparser.visualization import *
import cv2
import numpy as np


def test_viz():

    image = cv2.imread("tests/fixtures/ocr/test_gcv_image.jpg")
    ocr_agent = GCVAgent.with_credential(
        "tests/fixtures/ocr/test_gcv_credential.json", languages=["en"]
    )
    res = ocr_agent.load_response("tests/fixtures/ocr/test_gcv_response.json")

    draw_box(image, Layout([]))
    draw_text(image, Layout([]))

    draw_box(
        image,
        Layout(
            [
                Interval(0, 10, axis="x"),
                Rectangle(0, 50, 100, 80),
                Quadrilateral(np.array([[10, 10], [30, 40], [90, 40], [10, 20]])),
            ]
        ),
    )

    draw_text(
        image,
        Layout(
            [
                Interval(0, 10, axis="x"),
                Rectangle(0, 50, 100, 80),
                Quadrilateral(np.array([[10, 10], [30, 40], [90, 40], [10, 20]])),
            ]
        ),
    )

    for idx, level in enumerate(
        [
            GCVFeatureType.SYMBOL,
            GCVFeatureType.WORD,
            GCVFeatureType.PARA,
            GCVFeatureType.BLOCK,
            GCVFeatureType.PAGE,
        ]
    ):

        layout = ocr_agent.gather_full_text_annotation(res, level)

        draw_text(
            image,
            layout,
            arrangement="ud" if idx % 2 else "ud",
            font_size=15,
            text_color="pink",
            text_background_color="grey",
            with_box_on_text=True,
            text_box_width=2,
            text_box_color="yellow",
            with_layout=True,
            box_width=1,
            color_map={None: "blue"},
            show_element_id=True,
            id_font_size=8,
        )

        draw_box(image, layout)
        draw_text(image, layout)