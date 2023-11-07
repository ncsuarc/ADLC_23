from keras_cv import visualization

def visualize_dataset(inputs, value_range, rows, cols, bounding_box_format, offset=1):
    """
    Unused, but works
    """
    it = iter(inputs.take(offset))

    for _ in range(offset):
        inputs = next(it)

    images, bounding_boxes = inputs["images"], inputs["bounding_boxes"]
    visualization.plot_bounding_box_gallery(
        images,
        value_range=value_range,
        rows=rows,
        cols=cols,
        y_true=bounding_boxes,
        scale=10,
        line_thickness=1,
        font_scale=0.7,
        bounding_box_format=bounding_box_format,
        class_mapping={0: "target"},
    )

def visualize_detections(model, dataset, bounding_box_format):
    images, y_true = next(iter(dataset.take(1)))
    y_pred = model.predict(images)
    y_pred = keras_cv.bounding_box.to_ragged(y_pred)
    visualization.plot_bounding_box_gallery(
        images,
        value_range=(0, 255),
        bounding_box_format=bounding_box_format,
        y_true=y_true,
        y_pred=y_pred,
        scale=4,
        rows=2,
        cols=4,
        show=True,
        font_scale=0.7,
        class_mapping=class_mapping,
    )