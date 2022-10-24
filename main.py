from pathlib import Path
import ppp_bert
import onnxruntime
import onnxruntime_extensions

mobilebert_path = r'lordtt13-emo-mobilebert.onnx'
mobilebert_aug_path = r'lordtt13-emo-mobilebert-aug.onnx'


def update_mobilebert():
    ppp_bert.bert(Path(mobilebert_path), Path(mobilebert_aug_path))


if __name__ == '__main__':
    update_mobilebert()

    # Test with ONNX Runtime Python API
    test_input = ["I am a piece of text"]

    # Load the model
    session_options = onnxruntime.SessionOptions()
    session_options.register_custom_ops_library(onnxruntime_extensions.get_library_path())
    session = onnxruntime.InferenceSession('lordtt13-emo-mobilebert-aug.onnx', session_options)

    # Run the model
    results = session.run(["output_logits"], {"text": test_input})

    print(results[0])